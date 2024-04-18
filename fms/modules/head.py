from typing import Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    A general purpose Classification Head. When applied on the output of a
    Headless model, will project from the embedding space to a space equal to
    the number of classes provided.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        activation_fn: nn.Module,
        layer_norm: Optional[nn.Module] = None,
        dense_bias: bool = True,
        head_bias: bool = True,
        dropout: float = 0.0,
        apply_pooling_fn: bool = False,
    ):
        """
        Initialize a ClassificationHead

        Parameters
        ----------
        emb_dim: int
            the embedding dimension
        num_classes: int
            the output number of classes
        activation_fn: nn.Module
            the activation function to use prior to apply the dense layer
        layer_norm: nn.Module, optional
            the layer norm to apply prior to running the model head, (default is no layer_norm)
        dense_bias: bool
            the bias param in the dense layer (default is True)
        head_bias: bool
            the bias param in the head layer (default is True)
        dropout: float
            the dropout to use directly after activation (default is 0.0)
        apply_pooling_fn: bool
            if True, will take the first token for each sequence in the batch as input to the dense layer. Otherwise,
            use the entire sequence as input to the dense layer
        """
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim, bias=dense_bias)
        self.act = activation_fn
        self.dropout = nn.Dropout(dropout)
        self.ln = layer_norm
        self.head = nn.Linear(emb_dim, num_classes, bias=head_bias)
        self.apply_pooling_fn = apply_pooling_fn

    def forward(self, x: torch.Tensor):
        """Run the forward method of a classification head

        Parameters
        ----------
        x: torch.Tensor
            typically the output from a headless model

        Returns
        -------
        torch.Tensor
            a tensor projected to a space given by num_classes
        """
        if self.apply_pooling_fn:
            x = x[:, 0]
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.ln is not None:
            x = self.ln(x)
        x = self.head(x)
        return x
