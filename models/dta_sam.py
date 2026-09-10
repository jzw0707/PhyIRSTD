import torch
from torch import nn
from util.misc import nested_tensor_from_videos_list, NestedTensor
from hydra import compose, initialize
from models.sam2.modeling.sam2_utils import preprocess
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
from models.reliable_memory_encoder import ReliableMemoryEncoder
from models.dynamic_temporal_aggregator import DynamicTemporalAggregator
from models.multiscale_temporal_selection_unit import MultiscaleTemporalSelectionUnit
from models.thermal_prompt_generator import (
    ThermalAwarePromptGenerator,
    align_thermal_prior_to_sam,
)
from models.st_adapter import STAdapter
from models.config import ModelConfig
from models.model_utils import BackboneOutput, DecoderOutput, get_same_object_labels


class DTASAM(nn.Module):
    def __init__(self, image_size, sam, reliable_memory_encoder, args):
        super().__init__()
        cfg = ModelConfig()
        self.model_config = cfg.to_dict()
        self.sam = sam
        self.reliable_memory_encoder = reliable_memory_encoder
        self.thermal_prompt_generator = ThermalAwarePromptGenerator(
            dim=sam.hidden_dim,
            num_heads=cfg.tpg_num_heads,
            num_prompt_tokens=cfg.tpg_num_tokens,
            token_grid_size=cfg.tpg_token_grid_size,
            dropout=cfg.tpg_dropout,
            variant=cfg.tpg_variant,
            depth=cfg.tpg_depth,
        )
        self.st_adapter_stages = tuple(cfg.st_adapter_stages)
        if len(set(self.st_adapter_stages)) != len(self.st_adapter_stages):
            raise ValueError("st_adapter_stages must not contain duplicates")
        stage_ends = sam.image_encoder.trunk.stage_ends
        invalid_stages = [
            stage
            for stage in self.st_adapter_stages
            if stage < 0 or stage >= len(stage_ends)
        ]
        if invalid_stages:
            raise ValueError(
                f"invalid STAdapter stages: {invalid_stages}; valid stages are 0..{len(stage_ends) - 1}"
            )
        patch_sizes = list(cfg.st_adapter_patch_sizes)
        if len(patch_sizes) == 1:
            patch_sizes *= len(self.st_adapter_stages)
        if len(patch_sizes) != len(self.st_adapter_stages):
            raise ValueError(
                "st_adapter_patch_sizes must contain one value or one value per enabled stage"
            )
        stage_channels = [
            sam.image_encoder.trunk.blocks[stage_ends[stage]].dim_out
            for stage in self.st_adapter_stages
        ]
        self.st_adapters = nn.ModuleList(
            (
                STAdapter(
                    visual_dim=channels,
                    token_dim=sam.hidden_dim,
                    adapter_dim=cfg.st_adapter_dim,
                    patch_size=patch_size,
                    use_hsa=True,
                    num_heads=cfg.st_adapter_num_heads,
                    dropout=cfg.st_adapter_dropout,
                    layer_scale_init=cfg.st_adapter_layer_scale_init,
                    hsa_scale_init=cfg.st_adapter_hsa_scale_init,
                )
                for (channels, patch_size) in zip(stage_channels, patch_sizes)
            )
        )
        self._st_adapter_index = {
            stage: index for (index, stage) in enumerate(self.st_adapter_stages)
        }
        self.memory_bank = {}
        self.image_size = image_size
        self.rme_decision_window = cfg.rme_decision_window
        self.supervise_object_scores = (
            getattr(args, "object_score_loss_weight", 0.0) > 0
        )
        self.multiscale_temporal_selection_unit = MultiscaleTemporalSelectionUnit(
            dim=sam.hidden_dim
        )
        self.dynamic_temporal_aggregator = DynamicTemporalAggregator(
            dim=sam.hidden_dim,
            num_heads=cfg.dta_num_heads,
            dropout=cfg.dta_dropout,
            global_pool_size=cfg.dta_global_pool_size,
            max_history=cfg.dta_max_history,
        )

    def forward(self, samples, targets):
        """\xa0The forward expects a NestedTensor, which consists of:
           - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
           - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
           - targets:  list[dict]; during training contains masks, during inference frame Id info
        It returns a dict with the following elements:
           - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]
        """
        backbone_output: BackboneOutput = self.compute_backbone_output(samples, targets)
        (B, T) = (backbone_output.B, backbone_output.T)
        outputs = {"masks": []}
        if self.supervise_object_scores:
            outputs["object_score_logits"] = []
        for video_record in range(B):
            if self.training or T == 1:
                (self.memory_bank, self.last_frame_rme_applied) = ({}, 0)
            elif targets[video_record]["frame_ids"][0] == 0:
                (self.memory_bank, self.last_frame_rme_applied) = ({}, 0)
            dta_global_context = None
            dta_global_context = backbone_output.dta_global_contexts[
                video_record : video_record + 1
            ]
            prompt_tokens = backbone_output.thermal_prompt_tokens[
                video_record : video_record + 1
            ]
            thermal_prior_mask = backbone_output.thermal_prior_masks[
                video_record : video_record + 1
            ]
            temporal_positions = backbone_output.temporal_positions[video_record]
            for frame_idx in range(T):
                idx = video_record * T + frame_idx
                if self.training or T == 1:
                    memory_idx = frame_idx
                else:
                    memory_idx = targets[video_record]["frame_ids"][frame_idx]
                temporal_position = temporal_positions[frame_idx].item()
                current_vision_feats = backbone_output.get_current_feats(idx)
                decoder_out_w_mem: DecoderOutput = self.compute_decoder_out_w_mem(
                    backbone_output,
                    idx,
                    memory_idx,
                    self.memory_bank,
                    dta_global_context,
                    prompt_tokens,
                    thermal_prior_mask,
                    temporal_position,
                )
                if (
                    memory_idx - self.last_frame_rme_applied
                    >= self.rme_decision_window - 1
                    and memory_idx > self.rme_decision_window
                ):
                    decoder_out_no_mem_rme: DecoderOutput = (
                        self.compute_decoder_out_no_mem(
                            backbone_output,
                            idx,
                            dta_global_context,
                            prompt_tokens,
                            thermal_prior_mask,
                        )
                    )
                    pred_rme_logits = self.reliable_memory_encoder(
                        decoder_out_w_mem.obj_ptr.detach(),
                        decoder_out_no_mem_rme.early_obj_ptr.detach(),
                    )
                    if pred_rme_logits.argmax().item() == 1 and (not self.training):
                        decoder_out_w_mem = self.apply_memory_decision(
                            decoder_out_w_mem, decoder_out_no_mem_rme
                        )
                        self.last_frame_rme_applied = memory_idx
                    if self.training:
                        rme_label = get_same_object_labels(
                            decoder_out_w_mem.masks.detach().cpu(),
                            decoder_out_no_mem_rme.masks.detach().cpu(),
                            decoder_out_no_mem_rme.object_score_logits.detach(),
                        ).item()
                        if "pred_rme_logits" not in outputs:
                            outputs["pred_rme_logits"] = []
                            outputs["rme_label"] = []
                        outputs["pred_rme_logits"].append(pred_rme_logits)
                        outputs["rme_label"].append(rme_label)
                mem_dict_w_mem = self.compute_memory_bank_dict(
                    decoder_out_w_mem,
                    current_vision_feats,
                    backbone_output.feat_sizes,
                    temporal_position,
                )
                self.memory_bank[memory_idx] = mem_dict_w_mem
                outputs["masks"].append(decoder_out_w_mem.masks)
                if self.supervise_object_scores:
                    outputs["object_score_logits"].append(
                        decoder_out_w_mem.object_score_logits
                    )
        masks = torch.cat(outputs["masks"])
        if self.training:
            return outputs
        else:
            result = {"pred_masks": masks.squeeze(1)}
            if self.supervise_object_scores:
                result["object_score_logits"] = torch.cat(
                    outputs["object_score_logits"]
                )
            return result

    @staticmethod
    def preprocess_visual_features(samples, image_size):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)
        (video_samples, masks) = samples.decompose()
        (B, T, C, H, W) = video_samples.shape
        center_index = T // 2
        center_frames = video_samples[:, center_index]
        center_valid_masks = ~masks[:, center_index]
        samples = video_samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        BT = (B, T)
        return (samples, BT, orig_size, center_frames, center_valid_masks)

    @staticmethod
    def _extract_temporal_positions(targets, batch_size, clip_length, device):
        default_positions = torch.arange(
            clip_length, device=device, dtype=torch.float32
        )
        if targets is None:
            return default_positions.unsqueeze(0).expand(batch_size, -1)
        if len(targets) != batch_size:
            raise ValueError("targets must contain one record per video")
        positions = []
        for target in targets:
            values = target.get("frames_idx", target.get("frame_ids"))
            if values is None:
                positions.append(default_positions)
                continue
            values = torch.as_tensor(
                values, device=device, dtype=torch.float32
            ).flatten()
            if values.numel() != clip_length:
                raise ValueError(
                    "each temporal index sequence must match the clip length"
                )
            positions.append(values)
        return torch.stack(positions)

    def compute_backbone_output(self, samples, targets=None):
        (samples, BT, orig_size, center_frames, center_valid_masks) = (
            self.preprocess_visual_features(samples, self.image_size)
        )
        (B, T) = BT
        temporal_positions = self._extract_temporal_positions(
            targets, B, T, samples.device
        )
        (thermal_prompt_tokens, thermal_prior_masks) = self.thermal_prompt_generator(
            center_frames, center_valid_masks
        )
        (vis_outs, thermal_prompt_tokens) = self._forward_image_encoder(
            samples, T, thermal_prompt_tokens, temporal_positions
        )
        thermal_prompt_tokens = self.thermal_prompt_generator.refine_for_prompt_encoder(
            thermal_prompt_tokens
        )
        if self.thermal_prompt_generator.variant == "phy_deep":
            thermal_prior_masks = align_thermal_prior_to_sam(
                thermal_prior_masks,
                center_frames.shape[-2:],
                self.image_size,
                self.sam.sam_prompt_encoder.mask_input_size,
            )
        (backbone_out, multiscale_features) = self._forward_fpn(vis_outs)
        (_, vision_feats, vision_pos_embeds, feat_sizes) = (
            self.sam._prepare_backbone_features(backbone_out)
        )
        dta_global_contexts = None
        dta_global_contexts = self._build_dta_global_context(
            multiscale_features, B, T, temporal_positions
        )
        out = BackboneOutput(
            B,
            T,
            orig_size,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
            dta_global_contexts,
            thermal_prompt_tokens,
            thermal_prior_masks,
            temporal_positions,
        )
        return out

    def _forward_image_encoder(
        self, samples, clip_length, prompt_tokens, temporal_positions=None
    ):
        trunk = self.sam.image_encoder.trunk
        visual = trunk.patch_embed(samples)
        visual = visual + trunk._get_pos_embed(visual.shape[1:3])
        outputs = []
        stage_index = 0
        for block_index, block in enumerate(trunk.blocks):
            visual = block(visual)
            if block_index not in trunk.stage_ends:
                continue
            if stage_index in self._st_adapter_index:
                adapter = self.st_adapters[self._st_adapter_index[stage_index]]
                (visual_residual, token_residual) = adapter(
                    visual.permute(0, 3, 1, 2),
                    clip_length,
                    prompt_tokens,
                    temporal_positions,
                )
                visual = visual + visual_residual.permute(0, 2, 3, 1)
                prompt_tokens = prompt_tokens + token_residual
            if trunk.return_interm_layers or block_index == trunk.stage_ends[-1]:
                outputs.append(visual.permute(0, 3, 1, 2))
            stage_index += 1
        return (outputs, prompt_tokens)

    def _build_dta_global_context(
        self, multiscale_features, batch_size, num_frames, temporal_positions=None
    ):
        sequence_features = [
            feature.reshape(batch_size, num_frames, *feature.shape[1:])
            for feature in multiscale_features
        ]
        clip_features = self.multiscale_temporal_selection_unit(sequence_features)
        contexts = []
        for video_record in range(batch_size):
            contexts.append(
                self.dynamic_temporal_aggregator.build_global_context(
                    clip_features[video_record],
                    None
                    if temporal_positions is None
                    else temporal_positions[video_record],
                )
            )
        return torch.cat(contexts, dim=0)

    def _build_dta_history_context(
        self, memory_bank, frame_idx, pixel_features, current_temporal_position=None
    ):
        return self.dynamic_temporal_aggregator.build_history_context(
            memory_bank,
            frame_idx,
            pixel_features.shape[0],
            pixel_features.dtype,
            pixel_features.device,
            current_temporal_position,
        )

    def compute_decoder_out_w_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        memory_idx: int,
        memory_bank: dict,
        dta_global_context=None,
        prompt_tokens=None,
        thermal_prior_mask=None,
        temporal_position=None,
    ):
        current_vision_feats = backbone_out.get_current_feats(idx)
        current_vision_pos_embeds = backbone_out.get_current_pos_embeds(idx)
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)
        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            frame_idx=memory_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],
            num_frames=memory_idx + 1,
            memory_bank=memory_bank,
        )
        dta_history_context = self._build_dta_history_context(
            memory_bank, memory_idx, pix_feat_with_mem, temporal_position
        )
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            prompt_inputs=prompt_tokens,
            mask_inputs=thermal_prior_mask,
            high_res_features=high_res_features,
            dynamic_temporal_aggregator=self.dynamic_temporal_aggregator,
            dta_global_context=dta_global_context,
            dta_history_context=dta_history_context,
        )
        decoder_out.compute_mask(self.image_size, backbone_out.orig_size[idx])
        return decoder_out

    def compute_decoder_out_no_mem(
        self,
        backbone_out: BackboneOutput,
        idx: int,
        dta_global_context=None,
        prompt_tokens=None,
        thermal_prior_mask=None,
    ):
        current_vision_feats = backbone_out.get_current_feats(idx)
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)
        pix_feat_no_mem = current_vision_feats[-1:][-1] + self.sam.no_mem_embed
        (height, width) = backbone_out.feat_sizes[-1]
        pix_feat_no_mem = pix_feat_no_mem.permute(1, 2, 0).view(
            1, self.sam.hidden_dim, height, width
        )
        decoder_out: DecoderOutput = self.sam._forward_sam_heads(
            backbone_features=pix_feat_no_mem,
            prompt_inputs=prompt_tokens,
            mask_inputs=thermal_prior_mask,
            high_res_features=high_res_features,
            dynamic_temporal_aggregator=self.dynamic_temporal_aggregator,
            dta_global_context=dta_global_context,
            dta_history_context=None,
        )
        decoder_out.compute_mask(self.image_size, backbone_out.orig_size[idx])
        return decoder_out

    def compute_memory_bank_dict(
        self,
        decoder_out: DecoderOutput,
        current_vision_feats,
        feat_sizes,
        temporal_position=None,
    ):
        (maskmem_features, maskmem_pos_enc) = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        memory_dict = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": decoder_out.low_res_masks,
            "obj_ptr": decoder_out.obj_ptr,
            "temporal_position": temporal_position,
        }
        return memory_dict

    def apply_memory_decision(self, with_memory, without_memory):
        """Correct positive regions using the current frame's memory-free prediction."""
        masks = with_memory.high_res_masks
        reference = without_memory.high_res_masks
        masks[reference > 0] = reference[reference > 0] * 10
        with_memory.high_res_masks = masks
        return with_memory

    def _forward_fpn(self, vis_outs):
        (features, pos) = self.sam.image_encoder.neck(vis_outs)
        (features, pos) = (features[:-1], pos[:-1])
        multiscale_features = list(features)
        image_embedding = features[-1]
        backbone_out = {
            "vision_features": image_embedding,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        backbone_out["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        return (backbone_out, multiscale_features)

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        num_frames,
        memory_bank,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)
        C = self.sam.hidden_dim
        (H, W) = feat_sizes[-1]
        if self.sam.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat
        num_obj_ptr_tokens = 0
        if frame_idx != 0:
            (to_cat_memory, to_cat_memory_pos_embed) = ([], [])
            t_pos_and_prevs = []
            chosen_frames = []
            for t_pos in range(1, self.sam.num_maskmem):
                t_rel = self.sam.num_maskmem - t_pos
                prev_frame_idx = frame_idx - t_rel
                chosen_frames.append(prev_frame_idx)
                t_pos_and_prevs.append((t_pos, memory_bank.get(prev_frame_idx, None)))
            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"]
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                tpos_enc_id = self.sam.num_maskmem - t_pos - 1
                tpos_enc_id = min(tpos_enc_id, self.sam.maskmem_tpos_enc.shape[0] - 1)
                maskmem_enc = maskmem_enc + self.sam.maskmem_tpos_enc[tpos_enc_id]
                to_cat_memory_pos_embed.append(maskmem_enc)
            if self.sam.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(
                    num_frames, self.sam.max_obj_ptrs_in_encoder
                )
                pos_and_ptrs = []
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx - t_diff
                    if t < 0 or t >= num_frames:
                        break
                    out = memory_bank.get(t, None)
                    if out is not None:
                        pos_and_ptrs.append((t_diff, memory_bank[t]["obj_ptr"]))
                if len(pos_and_ptrs) > 0:
                    (pos_list, ptrs_list) = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.sam.mem_dim)
                    if self.sam.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.sam.mem_dim, self.sam.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(
                            C // self.sam.mem_dim, dim=0
                        )
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            pix_feat_with_mem = current_vision_feats[-1] + self.sam.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        pix_feat_with_mem = self.sam.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem


from models.path_utils import SAM2_PATHS_CONFIG


def build_dta_sam(args):
    (sam2_weights, sam2_config) = SAM2_PATHS_CONFIG[args.sam2_version]
    sam2_weights = getattr(args, "sam2_checkpoint", None) or sam2_weights
    if not os.path.isfile(sam2_weights):
        raise FileNotFoundError(
            f"SAM2-{args.sam2_version} checkpoint not found: {sam2_weights}"
        )
    with initialize(version_base=None, config_path="sam2", job_name="test_app"):
        cfg = compose(config_name=sam2_config)
        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = True
        cfg.model.pred_obj_scores_mlp = True
        cfg.model.fixed_no_obj_ptr = True
        sam = instantiate(cfg.model, _recursive_=True)
    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)[
        "model"
    ]
    sam.load_state_dict(state_dict, strict=False)
    reliable_memory_encoder = ReliableMemoryEncoder(sam.hidden_dim)
    model = DTASAM(
        image_size=sam.image_size,
        sam=sam,
        reliable_memory_encoder=reliable_memory_encoder,
        args=args,
    )
    configure_trainable_parameters(model, args)
    return model


def configure_trainable_parameters(model, args):
    """Train the complete adaptation stack; freeze SAM except the NUDT presence head."""
    trainable_prefixes = (
        "thermal_prompt_generator.",
        "st_adapters.",
        "reliable_memory_encoder.",
        "multiscale_temporal_selection_unit.",
        "dynamic_temporal_aggregator.",
    )
    if getattr(args, "object_score_loss_weight", 0.0) > 0:
        trainable_prefixes += (
            "sam.sam_mask_decoder.pred_obj_score_head.",
            "sam.sam_mask_decoder.obj_score_token.",
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefixes)
