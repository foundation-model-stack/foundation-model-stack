import torch
import torch.nn as nn


class ClassHead(nn.Module):
    """A general purpose Class Head"""

    def __init__(self, emb_dim: int, activation_fn: nn.Module, norm_eps: float):
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim)
        self.act = activation_fn
        self.ln = nn.LayerNorm(emb_dim, norm_eps)

    def forward(self, x: torch.FloatTensor):
        x = self.dense(x)
        x = self.act(x)
        x = self.ln(x)
        return x
