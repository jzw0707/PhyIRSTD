import torch
import torch.nn.functional as F
from torch import nn


class TemporalUnit(nn.Module):
    """Apply group-wise backward, stationary, and forward frame shifts."""

    def __init__(self, num_groups=3):
        super().__init__()
        if num_groups != 3:
            raise ValueError("TemporalUnit currently uses three shift groups")
        self.directions = (-1, 0, 1)

    @staticmethod
    def shift(feature, offset):
        if offset == 0 or feature.shape[1] <= 1:
            return feature

        shift = min(abs(offset), feature.shape[1] - 1)
        padding = feature.new_zeros(
            feature.shape[0], shift, *feature.shape[2:]
        )
        # Positive offsets move earlier features forward; negative offsets
        # move later features backward. Zero padding prevents clip wraparound.
        if offset > 0:
            return torch.cat([padding, feature[:, :-shift]], dim=1)
        return torch.cat([feature[:, shift:], padding], dim=1)

    def forward(self, feature, group_index):
        return self.shift(feature, self.directions[group_index])


class MultiscaleTemporalSelectionUnit(nn.Module):
    """Select and aggregate multiscale sequence features before DTA.

    This follows Figure 7 of PhyIRSTD: three backbone features are aligned to
    the middle resolution and concatenated, then jointly modulated by channel,
    spatial, and direction-aware temporal selectors. The selected feature is
    split back into three scales, added to the aligned residual features, and
    summed for the downstream Dynamic Temporal Aggregator.
    """

    def __init__(self, dim=256, num_scales=3, temporal_groups=3):
        super().__init__()
        if num_scales != 3:
            raise ValueError("MTSU expects exactly three feature scales")
        if temporal_groups != num_scales:
            raise ValueError(
                "temporal_groups must match the three aligned feature scales"
            )

        self.dim = dim
        self.num_scales = num_scales
        self.temporal_groups = temporal_groups
        fused_dim = dim * num_scales

        self.channel_unit = nn.Linear(fused_dim, fused_dim)
        self.spatial_unit = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.temporal_unit = TemporalUnit(temporal_groups)

    @staticmethod
    def _resize_feature(feature, size):
        batch_size, num_frames, channels, height, width = feature.shape
        if (height, width) == size:
            return feature
        feature = F.interpolate(
            feature.reshape(batch_size * num_frames, channels, height, width),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        return feature.reshape(batch_size, num_frames, channels, *size)

    def forward(self, features, return_multiscale=False):
        """Process three tensors shaped ``[B, T, C, H_l, W_l]``."""
        if len(features) != self.num_scales:
            raise ValueError(f"expected {self.num_scales} feature scales")

        reference_shape = features[0].shape[:3]
        if len(reference_shape) != 3:
            raise ValueError("MTSU features must have shape [B, T, C, H, W]")
        for feature in features:
            if feature.ndim != 5:
                raise ValueError("MTSU features must have shape [B, T, C, H, W]")
            if feature.shape[:3] != reference_shape:
                raise ValueError("all MTSU features must share B, T, and C")
            if feature.shape[2] != self.dim:
                raise ValueError(
                    f"expected {self.dim} channels, got {feature.shape[2]}"
                )

        target_size = features[1].shape[-2:]
        channel_descriptors = []
        spatial_response = None
        for feature in features:
            aligned_feature = self._resize_feature(feature, target_size)
            channel_descriptors.append(
                aligned_feature.mean(dim=(1, 3, 4))
            )
            scale_spatial_response = aligned_feature.amax(dim=2, keepdim=True)
            if spatial_response is None:
                spatial_response = scale_spatial_response
            else:
                spatial_response = torch.maximum(
                    spatial_response, scale_spatial_response
                )

        channel_weights = torch.sigmoid(
            self.channel_unit(torch.cat(channel_descriptors, dim=1))
        ).split(self.dim, dim=1)

        batch_size, num_frames = features[0].shape[:2]
        spatial_weights = self.spatial_unit(
            spatial_response.reshape(batch_size * num_frames, 1, *target_size)
        )
        spatial_weights = torch.sigmoid(spatial_weights).reshape(
            batch_size, num_frames, 1, *target_size
        )

        outputs = []
        aggregated = None
        for scale_index, (feature, scale_channel_weights) in enumerate(
            zip(features, channel_weights)
        ):
            aligned_feature = self._resize_feature(feature, target_size)
            temporal_weights = self.temporal_unit(
                aligned_feature, scale_index
            )
            selected_feature = (
                aligned_feature
                * scale_channel_weights[:, None, :, None, None]
                * spatial_weights
                * temporal_weights
            )
            output = selected_feature + aligned_feature
            aggregated = output if aggregated is None else aggregated + output
            if return_multiscale:
                outputs.append(output)

        if return_multiscale:
            return aggregated, outputs
        return aggregated
