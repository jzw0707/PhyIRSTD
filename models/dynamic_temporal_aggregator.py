import torch
import torch.nn.functional as F
from torch import nn

from models.sam2.modeling.sam2_utils import get_1d_sine_pe


class CrossAttentionFusion(nn.Module):
    """Residual cross-attention followed by a feed-forward block."""

    def __init__(self, dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, context):
        attended = self.attention(
            query=query,
            key=context,
            value=context,
            need_weights=False,
        )[0]
        query = self.norm1(query + self.dropout1(attended))
        return self.norm2(query + self.dropout2(self.ffn(query)))


class DynamicTemporalAggregator(nn.Module):
    """Enhance memory-conditioned pixels and mask queries with clip context.

    Queries retrieve pooled visual context and historical object pointers.
    This implementation uses residual cross-attention and feed-forward blocks.
    """

    def __init__(
        self,
        dim=256,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
        global_pool_size=4,
        max_history=7,
    ):
        super().__init__()
        if global_pool_size < 1:
            raise ValueError("global_pool_size must be positive")
        if max_history < 0:
            raise ValueError("max_history must be non-negative")

        self.dim = dim
        self.global_pool_size = global_pool_size
        self.max_history = max_history
        self.global_norm = nn.LayerNorm(dim)
        self.pixel_global_fusion = CrossAttentionFusion(
            dim, num_heads, dim_feedforward, dropout
        )
        self.object_global_fusion = CrossAttentionFusion(
            dim, num_heads, dim_feedforward, dropout
        )
        self.object_history_fusion = CrossAttentionFusion(
            dim, num_heads, dim_feedforward, dropout
        )

    def build_global_context(self, clip_features, temporal_positions=None):
        """Create compact global tokens from a full clip.

        Args:
            clip_features: Tensor shaped [T, C, H, W].
        Returns:
            Tensor shaped [1, T * pool_size^2, C].
        """
        if clip_features.ndim != 4:
            raise ValueError("clip_features must have shape [T, C, H, W]")

        num_frames, channels = clip_features.shape[:2]
        if channels != self.dim:
            raise ValueError(
                f"expected {self.dim} visual channels, got {channels}"
            )

        pooled = F.adaptive_avg_pool2d(
            clip_features, (self.global_pool_size, self.global_pool_size)
        )
        visual_tokens = pooled.flatten(2).transpose(1, 2)

        if temporal_positions is None:
            frame_positions = torch.arange(
                num_frames, device=clip_features.device, dtype=torch.float32
            )
        else:
            frame_positions = torch.as_tensor(
                temporal_positions,
                device=clip_features.device,
                dtype=torch.float32,
            ).flatten()
            if frame_positions.numel() != num_frames:
                raise ValueError(
                    "temporal_positions must contain one value per frame"
                )
        frame_positions = frame_positions - frame_positions[0]
        frame_positions = frame_positions / max(num_frames - 1, 1)
        temporal_pos = get_1d_sine_pe(frame_positions, dim=self.dim)
        temporal_pos = temporal_pos.to(dtype=visual_tokens.dtype)
        visual_tokens = visual_tokens + temporal_pos[:, None, :]
        visual_tokens = visual_tokens.flatten(0, 1).unsqueeze(0)
        return self.global_norm(visual_tokens)

    def build_history_context(
        self,
        memory_bank,
        frame_idx,
        batch_size,
        dtype,
        device,
        current_temporal_position=None,
    ):
        if self.max_history == 0 or not memory_bank:
            return None

        previous_indices = [idx for idx in memory_bank if idx < frame_idx]
        previous_indices.sort(key=lambda idx: frame_idx - idx)
        previous_indices = previous_indices[: self.max_history]
        if not previous_indices:
            return None

        pointers = [memory_bank[idx]["obj_ptr"] for idx in previous_indices]
        history = torch.stack(pointers, dim=1).to(device=device, dtype=dtype)
        if history.shape[0] != batch_size:
            raise ValueError(
                "historical object-pointer batch size does not match current frame"
            )

        use_actual_positions = current_temporal_position is not None
        if current_temporal_position is None:
            current_temporal_position = frame_idx
        memory_positions = []
        for idx in previous_indices:
            position = memory_bank[idx].get("temporal_position")
            memory_positions.append(
                position
                if use_actual_positions and position is not None
                else idx
            )
        distances = torch.tensor(
            [
                current_temporal_position - position
                for position in memory_positions
            ],
            device=device,
            dtype=torch.float32,
        )
        distances = distances / max(self.max_history, 1)
        temporal_pos = get_1d_sine_pe(distances, dim=self.dim)
        return history + temporal_pos.to(dtype=dtype).unsqueeze(0)

    def fuse_pixel_features(self, pixel_features, global_context):
        if pixel_features.ndim != 4:
            raise ValueError("pixel_features must have shape [B, C, H, W]")
        batch_size, channels, height, width = pixel_features.shape
        if channels != self.dim:
            raise ValueError(f"expected {self.dim} pixel channels, got {channels}")

        context = global_context.to(
            device=pixel_features.device, dtype=pixel_features.dtype
        )
        pixel_tokens = pixel_features.flatten(2).transpose(1, 2)
        pixel_tokens = self.pixel_global_fusion(pixel_tokens, context)
        pixel_features = pixel_tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )

        return pixel_features

    def fuse_object_tokens(self, object_tokens, global_context, history_context=None):
        context = global_context.to(
            device=object_tokens.device, dtype=object_tokens.dtype
        )
        object_tokens = self.object_global_fusion(object_tokens, context)
        if history_context is not None:
            history_context = history_context.to(
                device=object_tokens.device, dtype=object_tokens.dtype
            )
            object_tokens = self.object_history_fusion(
                object_tokens, history_context
            )
        return object_tokens

    def forward(
        self,
        pixel_features,
        mask_tokens,
        global_context,
        memory_bank,
        frame_idx,
        current_temporal_position=None,
    ):
        pixel_features = self.fuse_pixel_features(pixel_features, global_context)
        history_context = self.build_history_context(
            memory_bank,
            frame_idx,
            pixel_features.shape[0],
            mask_tokens.dtype,
            mask_tokens.device,
            current_temporal_position,
        )
        object_tokens = self.fuse_object_tokens(
            mask_tokens, global_context, history_context
        )
        return pixel_features, object_tokens
