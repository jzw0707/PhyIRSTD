"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""

import math
import sys
from typing import Iterable
import torch
import util.misc as utils
from torch.nn import functional as F
from models.segmentation import (
    loss_masks,
    object_presence_loss,
    object_presence_targets,
)


def clip_optimizer_gradients(
    optimizer,
    max_norm,
    new_module_max_norm=None,
):
    """Clip each optimizer group independently and return pre-clip norms."""
    group_norms = {}
    norm_device = None
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = [
            parameter for parameter in group["params"] if parameter.grad is not None
        ]
        group_name = group.get("group_name", f"group_{group_index}")
        if parameters and norm_device is None:
            norm_device = parameters[0].grad.device
        group_max_norm = max_norm
        if group_name == "new_modules" and new_module_max_norm is not None:
            group_max_norm = new_module_max_norm

        if group_max_norm > 0:
            group_norm = torch.nn.utils.clip_grad_norm_(parameters, group_max_norm)
        else:
            group_norm = utils.get_total_grad_norm(parameters)
        group_norms[group_name] = group_norm

    if not group_norms:
        return torch.tensor(0.0), group_norms
    norm_device = norm_device or torch.device("cpu")
    total_norm = torch.linalg.vector_norm(
        torch.stack(
            [norm.detach().float().to(norm_device) for norm in group_norms.values()]
        )
    )
    return total_norm, group_norms


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    lr_scheduler=None,
    args=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.8f}"))
    for group_index, group in enumerate(optimizer.param_groups[1:], start=1):
        group_name = group.get("group_name", f"group_{group_index}")
        metric_logger.add_meter(
            f"lr_{group_name}",
            utils.SmoothedValue(window_size=1, fmt="{value:.8f}"),
        )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50

    accumulation_steps = getattr(args, "accumulation_steps", 1)
    if accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be positive")
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(data_loader)
    score_weight = getattr(args, "object_score_loss_weight", 0.0)
    presence_counts = torch.zeros(4, device=device)
    for step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        model.train()
        samples = samples.to(device)
        outputs = model(samples, targets)
        losses = {}
        seg_loss = loss_masks(
            torch.cat(outputs["masks"]),
            targets,
            num_frames=samples.tensors.shape[1],
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
        seg_loss["loss_mask"] = seg_loss["loss_mask"] * args.focal_loss_weight
        seg_loss["loss_dice"] = seg_loss["loss_dice"] * args.dice_loss_weight
        losses.update(seg_loss)
        if score_weight > 0:
            score_logits = torch.cat(outputs["object_score_logits"]).reshape(-1)
            losses["loss_object_score"] = score_weight * object_presence_loss(
                score_logits, targets
            )
            with torch.no_grad():
                present = object_presence_targets(targets, score_logits.device).bool()
                predicted = score_logits > 0
                presence_counts += torch.stack(
                    [
                        present.sum(),
                        (~present).sum(),
                        (present & ~predicted).sum(),
                        (~present & predicted).sum(),
                    ]
                )
        if "pred_rme_logits" in outputs:
            weight = torch.tensor([1.0, 2.0]).to(device)
            RME_loss = F.cross_entropy(
                torch.cat(outputs["pred_rme_logits"]),
                ignore_index=-1,
                target=torch.tensor(outputs["rme_label"]).long().to(device),
                weight=weight,
            )
            losses.update(
                {
                    "RME_loss": RME_loss
                    if not RME_loss.isnan()
                    else torch.tensor(0).to(device)
                }
            )

        loss_dict = losses
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        (losses / group_size).backward()

        if (step + 1) % accumulation_steps == 0 or step + 1 == total_batches:
            grad_total_norm, grad_group_norms = clip_optimizer_gradients(
                optimizer,
                max_norm,
                getattr(args, "new_module_clip_max_norm", None),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            metric_logger.update(grad_norm=grad_total_norm)
            metric_logger.update(
                **{
                    f"grad_norm_{group_name}": group_norm
                    for group_name, group_norm in grad_group_norms.items()
                }
            )

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        for group_index, group in enumerate(optimizer.param_groups[1:], start=1):
            group_name = group.get("group_name", f"group_{group_index}")
            metric_logger.update(**{f"lr_{group_name}": group["lr"]})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if score_weight > 0:
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(presence_counts)
        positives, negatives, false_negatives, false_positives = (
            presence_counts.tolist()
        )
        stats.update(
            {
                "object_positive_frames": positives,
                "object_negative_frames": negatives,
                "object_gate_false_negative_frames": false_negatives,
                "object_gate_false_positive_frames": false_positives,
                "object_gate_false_negative_rate": false_negatives / positives
                if positives
                else 0.0,
                "object_gate_false_positive_rate": false_positives / negatives
                if negatives
                else 0.0,
            }
        )
    return stats
