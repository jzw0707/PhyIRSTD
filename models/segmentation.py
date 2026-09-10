import torch
import torch.nn.functional as F
from util.misc import nested_tensor_from_tensor_list, interpolate
import einops


def object_presence_targets(targets, device):
    labels = []
    for target in targets:
        masks = target["native_masks"] if "native_masks" in target else target["masks"]
        labels.append(masks.flatten(1).any(dim=1))
    return torch.cat(labels).to(device=device, dtype=torch.float32)


def object_presence_loss(logits, targets):
    """Supervise the existing object-score head using per-frame mask presence."""
    logits = logits.reshape(-1).float()
    labels = object_presence_targets(targets, logits.device)
    if logits.shape != labels.shape:
        raise ValueError("Object-score logits must match the number of target frames")
    return F.binary_cross_entropy_with_logits(logits, labels)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)

    inputs = inputs.to(torch.float32)
    targets = targets.to(torch.float32)

    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # return loss.sum().to(torch.float16) / num_boxes
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.75, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    if num_boxes <= 0:
        raise ValueError("num_boxes must be positive")
    if alpha != -1 and not 0.0 <= alpha <= 1.0:
        raise ValueError("focal alpha must be -1 or between 0 and 1")
    if gamma < 0.0:
        raise ValueError("focal gamma must be non-negative")

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



def loss_masks(
    outputs,
    targets,
    num_frames=1,
    focal_alpha=0.75,
    focal_gamma=2.0,
):
    """Compute the losses related to the masks: the focal loss and the dice loss.
       targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """

    if any("native_masks" in target for target in targets):
        if not all("native_masks" in target for target in targets):
            raise ValueError("native_masks must be supplied for every video")
        if outputs.shape[0] != len(targets) * num_frames or outputs.shape[1] != 1:
            raise ValueError("native supervision requires one mask per video frame")
        per_video_losses = []
        for index, target in enumerate(targets):
            native_masks = target["native_masks"].to(outputs)
            if native_masks.shape[0] != num_frames:
                raise ValueError("native_masks must match the clip length")
            logits = outputs[index * num_frames:(index + 1) * num_frames]
            # Remove batch padding before mapping logits to annotation pixels.
            height, width = (int(value) for value in target["size"])
            logits = logits[..., :height, :width]
            logits = F.interpolate(
                logits, size=native_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            ).reshape(1, -1)
            labels = native_masks.reshape(1, -1)
            per_video_losses.append({
                "loss_mask": sigmoid_focal_loss(
                    logits, labels, 1, alpha=focal_alpha, gamma=focal_gamma,
                ),
                "loss_dice": dice_loss(logits, labels, 1),
            })
        return {
            name: torch.stack([loss[name] for loss in per_video_losses]).mean()
            for name in ("loss_mask", "loss_dice")
        }

    src_masks = outputs
    bs = src_masks.shape[0]
    target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], size_divisibility=32, split=False).decompose()
    target_masks = target_masks.to(src_masks)

    # downsample ground truth masks with ratio mask_out_stride
    src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
    if num_frames > 1:
        src_masks = einops.rearrange(src_masks, '(b t) c h w -> b t c h w', t=num_frames)
    src_masks = src_masks.flatten(1)  # [b, thw]

    target_masks = target_masks.flatten(1)  # [b, thw]
    if target_masks.shape[0] == 1 and src_masks.shape[0] != 1:
        src_masks = src_masks.flatten(0).unsqueeze(0)

    losses = {
        "loss_mask": sigmoid_focal_loss(
            src_masks,
            target_masks,
            bs / num_frames,
            alpha=focal_alpha,
            gamma=focal_gamma,
        ),
        "loss_dice": dice_loss(src_masks, target_masks, bs/num_frames),
    }
    return losses
