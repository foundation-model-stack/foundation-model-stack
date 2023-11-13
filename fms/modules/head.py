import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """A general purpose LM-Head with Multilayer Perceptron"""

    def __init__(
        self,
        src_vocab_size: int,
        emb_dim: int,
        activation_fn: nn.Module,
        norm_eps: float,
    ):
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim)
        self.act = activation_fn
        self.ln = nn.LayerNorm(emb_dim, norm_eps)
        self.head = nn.Linear(emb_dim, src_vocab_size)

    def forward(self, x: torch.FloatTensor):
        x = self.dense(x)
        x = self.act(x)
        x = self.ln(x)
        x = self.head(x)
        return x
