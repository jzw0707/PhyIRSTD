import torch
from torch import nn
from .layers import TransformerBlock, SelfAttentionBlock
import einops
from .layers import PositionEmbeddingSine1D


class ReliableMemoryEncoder(nn.Module):
    """Token-based memory reliability classifier used by DTA-SAM.

    The current implementation predicts a binary correction decision; it does
    not implement continuous interpolation with the previous frame's mask.
    """

    def __init__(self, sam_hidden_dim=256):
        super().__init__()
        self.transformerBlock = SelfAttentionBlock(d_model=sam_hidden_dim, nhead=8)
        self.decision_token = nn.Embedding(1, sam_hidden_dim)
        self.decision_class_head = nn.Linear(sam_hidden_dim, 2)

    def forward(self, obj_ptr_zero, obj_ptr_two):
        B = obj_ptr_zero.shape[0]
        obj_ptr_zero = obj_ptr_zero.unsqueeze(0)
        obj_ptr_two = obj_ptr_two.unsqueeze(0)

        input_sequence = torch.cat(
            [
                self.decision_token.weight.repeat(B, 1).unsqueeze(0),
                obj_ptr_zero,
                obj_ptr_two,
            ],
            dim=0,
        )
        out = self.transformerBlock(input_sequence)
        out = self.decision_class_head(out[0])
        # return CLS token
        return out
