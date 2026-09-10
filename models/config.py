"""Fixed complete-model configuration; component ablations are not exposed."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelConfig:
    tpg_variant: str = "phy_deep"
    tpg_depth: int = 2
    tpg_num_heads: int = 8
    tpg_num_tokens: int = 4
    tpg_token_grid_size: int = 16
    tpg_dropout: float = 0.1
    st_adapter_stages: tuple = (1, 2, 3)
    st_adapter_patch_sizes: tuple = (8, 4, 2)
    st_adapter_dim: int = 256
    st_adapter_num_heads: int = 8
    st_adapter_dropout: float = 0.0
    st_adapter_layer_scale_init: float = 0.1
    st_adapter_hsa_scale_init: float = 0.01
    dta_num_heads: int = 8
    dta_dropout: float = 0.1
    dta_global_pool_size: int = 4
    dta_max_history: int = 7
    rme_decision_window: int = 4

    def to_dict(self):
        return asdict(self)
