import torch


class RMSNorm(torch.nn.RMSNorm):
    """
    A RmsNorm implementation.
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).
        Recenter inputs around zero before normalizing, or just rescale?
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        use_high_precision_pow=True,
    ):
        super(RMSNorm, self).__init__(normalized_shape, eps)
        self.use_high_precision_pow = use_high_precision_pow

    def forward(self, x):
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        xf = super().forward(xf)
        x = xf.type_as(x)
        return x
