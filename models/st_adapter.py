import math

import torch
from einops import rearrange
from torch import nn

from models.sam2.modeling.sam2_utils import get_1d_sine_pe


class CrossAttentionGate(nn.Module):
    """Multiplicative cross-attention used by the original CMT adapter."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout
        )

    def forward(self, query, context):
        attended = self.multihead_attn(
            query=query,
            key=context,
            value=context,
            need_weights=False,
        )[0]
        return query * attended


class HierarchicalSelectiveAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout
        )
        self.linear = nn.Linear(dim, dim)

    def forward(self, tokens):
        attended = self.self_attn(
            tokens, tokens, tokens, need_weights=False
        )[0]
        return self.linear(attended)


class STAdapter(nn.Module):
    """Spatial-temporal adapter conditioned by key-frame thermal prompts.

    Visual features and prompt tokens exchange information in both directions.
    Local spatiotemporal attention supplies the temporal residual.
    """

    def __init__(
        self,
        visual_dim,
        token_dim=256,
        adapter_dim=256,
        patch_size=2,
        use_hsa=True,
        num_heads=8,
        dropout=0.0,
        layer_scale_init=0.1,
        hsa_scale_init=0.01,
    ):
        super().__init__()
        if adapter_dim % num_heads != 0:
            raise ValueError("adapter_dim must be divisible by num_heads")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")

        self.adapter_dim = adapter_dim
        self.patch_size = patch_size
        self.use_hsa = use_hsa

        if layer_scale_init < 0.0 or hsa_scale_init < 0.0:
            raise ValueError("STAdapter scale initializers must be non-negative")

        self.proj_vis_down = nn.Sequential(
            nn.Conv2d(visual_dim, adapter_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_vis_up = nn.Sequential(
            nn.ConvTranspose2d(
                adapter_dim, visual_dim, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_token_down = nn.Linear(token_dim, adapter_dim, bias=False)
        self.proj_token_up = nn.Linear(adapter_dim, token_dim, bias=False)

        # Small learnable residual scales keep newly attached adapters from
        # perturbing the frozen SAM2 representation at initialization.
        self.visual_scale = nn.Parameter(
            torch.full((visual_dim,), float(layer_scale_init))
        )
        self.token_scale = nn.Parameter(
            torch.full((token_dim,), float(layer_scale_init))
        )

        self.visual_from_token = CrossAttentionGate(
            adapter_dim, num_heads=num_heads, dropout=dropout
        )
        self.token_from_visual = CrossAttentionGate(
            adapter_dim, num_heads=num_heads, dropout=dropout
        )

        if self.use_hsa:
            self.spatiotemporal_attention = HierarchicalSelectiveAttention(
                adapter_dim, num_heads=num_heads
            )
            self.spatial_pos_embed = nn.Parameter(
                torch.zeros(
                    1, 1, 1, patch_size * patch_size, adapter_dim
                )
            )
            nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
            self.pre_temporal_norm = nn.LayerNorm(adapter_dim)
            self.hsa_scale = nn.Parameter(
                torch.full((adapter_dim,), float(hsa_scale_init))
            )

    def _hierarchical_temporal_residual(
        self,
        visual_tokens,
        batch_size,
        clip_length,
        height,
        width,
        temporal_positions=None,
    ):
        patch_size = self.patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "STAdapter patch_size must divide the visual feature size; "
                f"got patch_size={patch_size}, feature_size=({height}, {width})"
            )

        features = rearrange(
            visual_tokens,
            "(h w) (b t) c -> b t h w c",
            b=batch_size,
            t=clip_length,
            h=height,
            w=width,
        )
        features = rearrange(
            features,
            "b t (gh ph) (gw pw) c -> b t (gh gw) (ph pw) c",
            ph=patch_size,
            pw=patch_size,
        )
        if temporal_positions is None:
            temporal_positions = torch.arange(
                clip_length, device=features.device, dtype=torch.float32
            ).unsqueeze(0).expand(batch_size, -1)
        if temporal_positions.shape != (batch_size, clip_length):
            raise ValueError(
                "temporal_positions must have shape [B, T]; "
                f"got {tuple(temporal_positions.shape)}"
            )
        relative_positions = temporal_positions.to(
            device=features.device, dtype=torch.float32
        )
        relative_positions = relative_positions - relative_positions[:, :1]
        # Consecutive frames retain the previous 2*pi temporal range, while
        # wider strides produce proportionally wider temporal distances.
        relative_positions = (
            (relative_positions + 1.0) / max(clip_length, 1) * 2.0 * math.pi
        )
        temporal_pos = get_1d_sine_pe(
            relative_positions, dim=self.adapter_dim
        ).to(dtype=features.dtype)
        features = (
            features
            + temporal_pos[:, :, None, None, :]
            + self.spatial_pos_embed
        )
        features = rearrange(
            features, "b t n p c -> (t p) (b n) c"
        )
        features = self.spatiotemporal_attention(
            self.pre_temporal_norm(features)
        )
        features = rearrange(
            features,
            "(t p) (b n) c -> b t n p c",
            t=clip_length,
            p=patch_size * patch_size,
            b=batch_size,
        )
        return rearrange(
            features,
            "b t (gh gw) (ph pw) c -> (gh ph gw pw) (b t) c",
            gh=height // patch_size,
            gw=width // patch_size,
            ph=patch_size,
            pw=patch_size,
        )

    def forward(
        self,
        visual_features,
        clip_length,
        prompt_tokens,
        temporal_positions=None,
    ):
        """
        Args:
            visual_features: Tensor shaped [B*T, C, H, W].
            clip_length: Number of frames T.
            prompt_tokens: Tensor shaped [B, N, token_dim].
            temporal_positions: Optional real frame indices shaped [B, T].
        Returns:
            Visual and prompt-token residuals with their respective input shapes.
        """
        if visual_features.ndim != 4:
            raise ValueError("visual_features must have shape [B*T, C, H, W]")
        if prompt_tokens.ndim != 3:
            raise ValueError("prompt_tokens must have shape [B, N, C]")
        if clip_length < 1 or visual_features.shape[0] % clip_length != 0:
            raise ValueError("clip_length must divide the visual batch dimension")

        batch_size = visual_features.shape[0] // clip_length
        if prompt_tokens.shape[0] != batch_size:
            raise ValueError("visual and prompt-token batch sizes do not match")

        visual = self.proj_vis_down(visual_features)
        height, width = visual.shape[-2:]
        visual = rearrange(visual, "bt c h w -> (h w) bt c")
        tokens = self.proj_token_down(prompt_tokens).transpose(0, 1)

        adapted_visual = visual
        if self.use_hsa:
            adapted_visual = (
                adapted_visual
                + self._hierarchical_temporal_residual(
                    visual,
                    batch_size,
                    clip_length,
                    height,
                    width,
                    temporal_positions,
                ) * self.hsa_scale.view(1, 1, -1)
            )

        repeated_tokens = tokens.repeat_interleave(clip_length, dim=1)
        adapted_visual = self.visual_from_token(
            adapted_visual, repeated_tokens
        )
        clip_visual = rearrange(
            visual,
            "hw (b t) c -> hw b t c",
            b=batch_size,
            t=clip_length,
        ).mean(dim=2)
        adapted_tokens = self.token_from_visual(tokens, clip_visual)

        adapted_visual = rearrange(
            adapted_visual,
            "(h w) (b t) c -> (b t) c h w",
            b=batch_size,
            t=clip_length,
            h=height,
            w=width,
        )
        visual_residual = self.proj_vis_up(adapted_visual)
        visual_residual = visual_residual * self.visual_scale.view(1, -1, 1, 1)
        token_residual = self.proj_token_up(adapted_tokens).transpose(0, 1)
        token_residual = token_residual * self.token_scale.view(1, 1, -1)
        return visual_residual, token_residual
