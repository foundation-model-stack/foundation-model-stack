import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    A two-layer, symmetric, fully-connected MLP structure.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : int
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent space.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor=4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        gain=1,
    ):
        super(FeedForwardBlock, self).__init__()
        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(emb_dim, hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.reset_params(gain=gain)

    def reset_params(self, gain=1):
        # Fulfills following constraints in expectation:
        #  - Norm of w1 and w2 are equal (for step-normalizing optimizers like AdamW / Sophia)
        #  - Norm of output equals norm of input times gamma
        # when activation is relu-like
        for layer in ["w1", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=(2**0.5 * gain / self.w1.weight.numel() ** 0.5) ** 0.5,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        out = self.a(self.w1(x))
        if self.p_dropout:
            out = self.d(out)
        return self.w2(out)


class GatedLinearUnit(nn.Module):
    """
    A two-point-five-layer, fully-connected gated linear MLP structure (GLU).
    Contains 50% extra params compared to FeedForwardBlock, adjust accordingly.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : float
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent gates.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor=4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        gain=1,
    ):
        super(GatedLinearUnit, self).__init__()
        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(emb_dim, hidden_dim, bias=use_bias)
        self.wg = nn.Linear(emb_dim, hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor
        self.reset_params(gain=gain)

    def reset_params(self, gain=1):
        # Fulfills following constraints in expectation:
        #  - Norm of w1, wg and w2 are equal (for step-normalizing optimizers like AdamW / Sophia)
        #  - Norm of output equals norm of input times gamma
        # when activation is relu-like and input is standard normal
        for layer in ["w1", "w2", "wg"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=(2 * gain**2 / self.grow_factor) ** (1 / 6) / self.width**0.5,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        out = self.a(self.wg(x)) * self.w1(x)
        if self.p_dropout:
            out = self.d(out)
        return self.w2(out)
