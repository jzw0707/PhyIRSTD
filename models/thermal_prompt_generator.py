import math

import torch
import torch.nn.functional as F
from torch import nn


class ThermalDiffusionConv(nn.Module):
    """Extract local infrared features and apply stable thermal diffusion."""

    def __init__(self, out_channels=256):
        super().__init__()
        mid_channels = out_channels // 2
        self.local_extractor = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        initial_logit = math.log(0.1 / (0.24 - 0.1))
        self.horizontal_diffusivity = nn.Parameter(torch.tensor(initial_logit))
        self.vertical_diffusivity = nn.Parameter(torch.tensor(initial_logit))
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def _diffusion_kernel(self, channels, dtype, device):
        alpha_x = 0.24 * torch.sigmoid(self.horizontal_diffusivity)
        alpha_y = 0.24 * torch.sigmoid(self.vertical_diffusivity)
        kernel = torch.zeros(3, 3, dtype=dtype, device=device)
        kernel[1, 0] = alpha_x
        kernel[1, 2] = alpha_x
        kernel[0, 1] = alpha_y
        kernel[2, 1] = alpha_y
        kernel[1, 1] = 1.0 - 2.0 * alpha_x - 2.0 * alpha_y
        return kernel.reshape(1, 1, 3, 3).expand(channels, 1, 3, 3)

    def forward(self, thermal_image):
        local_features = self.local_extractor(thermal_image)
        kernel = self._diffusion_kernel(
            local_features.shape[1], local_features.dtype, local_features.device
        )
        diffused_features = F.conv2d(
            local_features,
            kernel,
            padding=1,
            groups=local_features.shape[1],
        )
        return local_features + self.refine(
            torch.cat([local_features, diffused_features], dim=1)
        )


class MultiplicativePromptFusion(nn.Module):
    """Bounded elementwise query/context interaction, added as a small residual."""

    def __init__(self, dim):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.context_gate = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(self, queries, context):
        gate = 2 * torch.sigmoid(self.context_gate(self.context_norm(context)))
        product = self.query_norm(queries) * gate
        return queries + self.scale * self.output_projection(product)


class PromptRefinementBlock(nn.Module):
    """Self-attention -> FFN -> cross-attention -> multiply -> MLP.

    The ordering follows the local PhyIRSTD Figure 5. Residual scaling and
    bounded multiplication are implementation choices, not specified formulas
    from the paper.
    """

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.cross_norm = nn.LayerNorm(dim)
        self.visual_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.multiply = MultiplicativePromptFusion(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.residual_scales = nn.Parameter(torch.full((4, dim), 1e-3))

    def forward(self, queries, visual_tokens, key_padding_mask=None):
        normalized = self.self_norm(queries)
        attended = self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        queries = queries + self.residual_scales[0] * attended
        queries = queries + self.residual_scales[1] * self.ffn(self.ffn_norm(queries))
        visual = self.visual_norm(visual_tokens)
        context = self.cross_attention(self.cross_norm(queries), visual, visual,
                                       key_padding_mask=key_padding_mask, need_weights=False)[0]
        queries = queries + self.residual_scales[2] * context
        queries = self.multiply(queries, context)
        return queries + self.residual_scales[3] * self.mlp(self.mlp_norm(queries))


def align_thermal_prior_to_sam(prior, input_size, image_size, mask_input_size):
    """Map the rectangular prior through the image's bottom/right square padding."""
    height, width = input_size
    if min(height, width) <= 0 or max(height, width) > image_size:
        raise ValueError("Thermal prior input must fit inside the SAM square image")
    image_logits = F.interpolate(prior.float(), size=(height, width),
                                 mode="bilinear", align_corners=False)
    square_logits = F.pad(image_logits, (0, image_size - width, 0, image_size - height), value=-10.)
    return F.interpolate(square_logits, size=mask_input_size, mode="bilinear",
                         align_corners=False, antialias=True)


class MaskPriorGenerator(nn.Module):
    """Generate key-frame prompt tokens and a dense mask prior.

    The mask branch follows the multiplicative mask-token/visual-feature design
    used by MPG-SAM, while the token path follows the TPG blocks in Figure 5 of
    PhyIRSTD.
    """

    def __init__(
        self,
        dim=256,
        num_heads=8,
        num_prompt_tokens=4,
        token_grid_size=16,
        dropout=0.1,
        variant="legacy",
        depth=2,
    ):
        super().__init__()
        self.dim = dim
        self.num_prompt_tokens = num_prompt_tokens
        self.token_grid_size = token_grid_size
        if variant not in ("legacy", "phy_deep"):
            raise ValueError("Unknown TPG variant")
        if depth < 1 or token_grid_size < 1 or num_prompt_tokens < 1:
            raise ValueError("TPG depth, grid size, and prompt count must be positive")
        self.variant = variant

        self.token_positional_conv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim
        )
        self.visual_self_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_norm1 = nn.LayerNorm(dim)
        self.visual_norm2 = nn.LayerNorm(dim)
        self.visual_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        self.prompt_queries = nn.Parameter(
            torch.empty(1, num_prompt_tokens, dim)
        )
        nn.init.trunc_normal_(self.prompt_queries, std=0.02)
        self.query_self_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query_cross_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query_norm1 = nn.LayerNorm(dim)
        self.query_norm2 = nn.LayerNorm(dim)
        self.query_norm3 = nn.LayerNorm(dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.prompt_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.mask_feature_projection = nn.Conv2d(dim, dim, kernel_size=1)
        self.mask_token_projection = nn.Linear(dim, dim)
        self.mask_product = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
        )
        # Keep legacy key names for loading existing trained prompt weights.
        if variant == "phy_deep":
            self.first_multiply = MultiplicativePromptFusion(dim)
            self.refinement_blocks = nn.ModuleList([
                PromptRefinementBlock(dim, num_heads, dropout) for _ in range(depth - 1)
            ])

    def forward(self, thermal_features, valid_mask=None):
        pooled = F.adaptive_avg_pool2d(
            thermal_features, (self.token_grid_size, self.token_grid_size)
        )
        padding_mask = None
        empty_samples = None
        if self.variant == "phy_deep" and valid_mask is not None:
            valid = F.interpolate(valid_mask[:, None].float(), size=thermal_features.shape[-2:], mode="nearest")
            coverage = F.adaptive_avg_pool2d(valid, (self.token_grid_size, self.token_grid_size))
            pooled = F.adaptive_avg_pool2d(thermal_features * valid, coverage.shape[-2:]) / coverage.clamp_min(1e-6)
            padding_mask = coverage.flatten(1) == 0
            empty_samples = padding_mask.all(dim=1)
            # Attention cannot have every key masked. Expose one zero dummy key.
            padding_mask = padding_mask.clone()
            padding_mask[empty_samples, 0] = False
        pooled = pooled + self.token_positional_conv(pooled)
        visual_tokens = pooled.flatten(2).transpose(1, 2)
        if padding_mask is not None:
            visual_tokens = visual_tokens.masked_fill(padding_mask[..., None], 0)
            visual_tokens = visual_tokens.masked_fill(empty_samples[:, None, None], 0)
        attended = self.visual_self_attention(
            visual_tokens, visual_tokens, visual_tokens, key_padding_mask=padding_mask, need_weights=False
        )[0]
        visual_tokens = self.visual_norm1(visual_tokens + attended)
        visual_tokens = self.visual_norm2(
            visual_tokens + self.visual_ffn(visual_tokens)
        )

        prompt_tokens = self.prompt_queries.expand(
            thermal_features.shape[0], -1, -1
        )
        attended = self.query_self_attention(
            prompt_tokens, prompt_tokens, prompt_tokens, need_weights=False
        )[0]
        prompt_tokens = self.query_norm1(prompt_tokens + attended)
        prompt_tokens = self.query_norm2(
            prompt_tokens + self.query_ffn(prompt_tokens)
        )
        attended = self.query_cross_attention(
            prompt_tokens, visual_tokens, visual_tokens, key_padding_mask=padding_mask, need_weights=False
        )[0]
        prompt_tokens = self.query_norm3(prompt_tokens + attended)
        if self.variant == "phy_deep":
            prompt_tokens = self.first_multiply(prompt_tokens, attended)
        prompt_tokens = self.prompt_mlp(prompt_tokens)
        if self.variant == "phy_deep":
            for block in self.refinement_blocks:
                prompt_tokens = block(prompt_tokens, visual_tokens, padding_mask)
            if empty_samples is not None:
                prompt_tokens = prompt_tokens.masked_fill(empty_samples[:, None, None], 0)

        mask_features = self.mask_feature_projection(thermal_features)
        mask_token = self.mask_token_projection(prompt_tokens).mean(dim=1)
        mask_product = mask_features * mask_token[:, :, None, None]
        reference_mask = self.mask_product(mask_product)
        return prompt_tokens, reference_mask


class ThermalAwarePromptGenerator(nn.Module):
    """Build visual prompts from the normalized center frame of each clip."""

    def __init__(
        self,
        dim=256,
        num_heads=8,
        num_prompt_tokens=4,
        token_grid_size=16,
        dropout=0.1,
        variant="legacy",
        depth=2,
    ):
        super().__init__()
        self.variant = variant
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "luminance_weights",
            torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.thermal_diffusion_conv = ThermalDiffusionConv(dim)
        self.mask_prior_generator = MaskPriorGenerator(
            dim=dim,
            num_heads=num_heads,
            num_prompt_tokens=num_prompt_tokens,
            token_grid_size=token_grid_size,
            dropout=dropout,
            variant=variant,
            depth=depth,
        )
        if variant == "phy_deep":
            self.prompt_output_norm = nn.LayerNorm(dim)
            self.prompt_output_mlp = nn.Sequential(
                nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
            )
            self.prompt_output_scale = nn.Parameter(torch.full((dim,), 1e-3))

    def refine_for_prompt_encoder(self, tokens):
        """Refine the ST-Adapter-updated tokens immediately before SAM prompts."""
        if self.variant == "legacy":
            return tokens
        return tokens + self.prompt_output_scale * self.prompt_output_mlp(self.prompt_output_norm(tokens))

    def forward(self, center_frame, valid_mask=None):
        image = center_frame * self.image_std + self.image_mean
        thermal_image = (image * self.luminance_weights).sum(dim=1, keepdim=True)
        thermal_image = thermal_image.clamp(0.0, 1.0)
        if valid_mask is not None:
            thermal_image = thermal_image * valid_mask[:, None].to(
                dtype=thermal_image.dtype
            )

        thermal_features = self.thermal_diffusion_conv(thermal_image)
        prompt_tokens, reference_mask = self.mask_prior_generator(
            thermal_features, valid_mask=valid_mask,
        )
        if valid_mask is not None:
            prior_valid_mask = F.interpolate(
                valid_mask[:, None].float(),
                size=reference_mask.shape[-2:],
                mode="nearest",
            ).bool()
            reference_mask = reference_mask.masked_fill(~prior_valid_mask, -10.0)
        return prompt_tokens, reference_mask
