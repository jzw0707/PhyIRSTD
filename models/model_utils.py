from dataclasses import dataclass
from torch import Tensor
import torch
from models.sam2.modeling.sam2_utils import postprocess_masks

@dataclass
class DecoderOutput:
    """Class for keeping decoder outputs."""
    ious: Tensor = None
    low_res_masks: Tensor = None
    high_res_masks: Tensor = None
    obj_ptr: Tensor = None
    early_obj_ptr: Tensor = None
    object_score_logits: Tensor = None
    pix_feat_with_mem: Tensor = None
    masks: Tensor = None

    def compute_mask(self, image_size, original_size):
        assert self.low_res_masks is not None, 'before calling compute_mask you have to set \'low_res_masks\''
        self.masks = postprocess_masks(
            self.low_res_masks,
            image_size,
            input_size=original_size,
            original_size=original_size,
        )

    def move_to_cpu(self):
        self.low_res_masks = self.low_res_masks.cpu()
        self.high_res_masks =  self.high_res_masks.cpu()
        self.obj_ptr = self.obj_ptr.cpu()
        self.object_score_logits = self.object_score_logits.cpu()
        self.pix_feat_with_mem = self.pix_feat_with_mem.cpu()
        self.masks = self.masks.cpu()
        return self
    
    
@dataclass
class BackboneOutput:
    """Class for keeping backbone outputs."""
    B: int = None
    T: int = None
    orig_size: list = None
    vision_feats: list = None
    vision_pos_embeds: list = None
    feat_sizes: list = None
    dta_global_contexts: Tensor = None
    thermal_prompt_tokens: Tensor = None
    thermal_prior_masks: Tensor = None
    temporal_positions: Tensor = None

    def get_current_feats(self, idx):
        current_vision_feats = [x[:, idx:idx + 1, :] for x in self.vision_feats]
        return current_vision_feats

    def get_current_pos_embeds(self, idx):
        current_vision_pos_embeds = [x[:, idx:idx + 1, :] for x in self.vision_pos_embeds]
        return current_vision_pos_embeds
    
    def get_high_res_features(self, current_vision_feats):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], self.feat_sizes[:-1])
        ]
        return high_res_features

    def move_to_cpu(self):
        self.vision_feats = [x.cpu() for x in self.vision_feats]
        self.vision_pos_embeds = [x.cpu() for x in self.vision_pos_embeds]
        if self.dta_global_contexts is not None:
            self.dta_global_contexts = self.dta_global_contexts.cpu()
        if self.thermal_prompt_tokens is not None:
            self.thermal_prompt_tokens = self.thermal_prompt_tokens.cpu()
        if self.thermal_prior_masks is not None:
            self.thermal_prior_masks = self.thermal_prior_masks.cpu()
        if self.temporal_positions is not None:
            self.temporal_positions = self.temporal_positions.cpu()
        return self


def get_same_object_labels(masks, counterpart_masks, counter_logits=None):
    """
        masks (_type_): BT, H, W of torch.float32
    """
    from tools.metrics import eval_i_u
    masks, counterpart_masks = masks[0], counterpart_masks[0]
    inters, union = eval_i_u(masks.numpy() > 0, counterpart_masks.numpy() > 0)
    BT = len(masks)
    labels = torch.empty(BT)
    empty_mask_threshold = 50
    areas = (masks > 0).sum(dim=1).sum(dim=1)
    counter_areas = (counterpart_masks > 0).sum(dim=1).sum(dim=1)
    for j in range(BT):
        i, u = inters[j], union[j]
        a, ca = areas[j], counter_areas[j]
        if counter_logits is not None:
            logit = counter_logits[j].item()
            if logit < 0:
                labels[j] = -1
                continue
        if a <= empty_mask_threshold and ca <= empty_mask_threshold:
            # both empty, we consider as the same 'object'
            labels[j] = -1
            continue
        if i / min(a, ca) < 0.1:
            # includes the case where one of the 2 is empty
            labels[j] = 1
            continue
        # for now, treat all other cases as 'same object'
        labels[j] = 0
    return labels

class DDPWrapper:
    def __init__(self, ddp_module):
        self.ddp_module = ddp_module

    def __call__(self, *args, **kwargs):
        return self.ddp_module(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.ddp_module, name):
            attr = getattr(self.ddp_module, name)

            #  Make sure to return DDPWrapper instead of directly returning the attribute in case of a callable such as state.model.train()
            if callable(attr):

                def wrapper(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if result is self.ddp_module:
                        return self
                    return result

                return wrapper
            return attr

        if hasattr(self.ddp_module.module, name):
            return getattr(self.ddp_module.module, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
