import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    A generalized LayerNorm implementation. With all optional arguments set to True, equivalent to nn.LayerNorm up to epsilon stabilization term
    (this class divides inputs by min(norm, eps), while nn.LayerNorm divides by norm + eps).
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale : bool
        Include a learned scaling term after normalization?
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale=True,
        use_high_precision_pow=False,
    ):
        super(LayerNormParameterized, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        self.use_high_precision_pow = use_high_precision_pow

        if self.elementwise_scale:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))

    def reset_parameters(self):
        if self.elementwise_scale:
            self.weight.data.fill_(1)

    def forward(self, x):
        # x = F.normalize(x, dim=-1)*math.sqrt(x.size(-1))
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale:
            x = self.weight * x
        return x
