import torch
import torch.nn as nn


class LayerNormParameterized(nn.Module):
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
    elementwise_shift : bool
        Include a learned bias term after normalization?
    use_mean : bool
        Recenter inputs around zero before normalizing, or just rescale?
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale=True,
        elementwise_shift=False,
        use_mean=False,
        use_high_precision_pow=False,
    ):
        super(LayerNormParameterized, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        self.elementwise_shift = elementwise_shift
        self.use_mean = use_mean
        self.use_high_precision_pow = use_high_precision_pow

        if self.elementwise_scale:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("weight", None)
        if self.elementwise_shift:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("bias", None)

    def reset_parameters(self):
        if self.elementwise_scale:
            self.weight.data.fill_(1)
        if self.elementwise_shift:
            self.bias.data.zero_()

    def forward(self, x):
        print(f"[LN-DBG] Input: {x.shape}, type = {'DTensor' if hasattr(x, '_spec') else 'Tensor'}")
        if hasattr(x, '_spec'):
            print(f"[LN-DBG] --> x._spec: placements = {x._spec.placements}, mesh = {x._spec.mesh}")
            print(f"[LN-DBG] --> x.to_local().shape = {x.to_local().shape}")

        if self.use_mean:
            print(f"[LN-DBG] Subtracting mean along dim -1")
            x = x - x.mean(-1, keepdim=True)
            print(f"[LN-DBG] After mean subtraction: {x.shape}, type = {'DTensor' if hasattr(x, '_spec') else 'Tensor'}")

        xf = x
        if self.use_high_precision_pow:
            print(f"[LN-DBG] Casting to float32 for high precision computation")
            xf = x.float()

        print(f"[LN-DBG] Before normalization: {xf.shape}, type = {'DTensor' if hasattr(xf, '_spec') else 'Tensor'}")
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        print(f"[LN-DBG] After normalization: {xf.shape}, type = {'DTensor' if hasattr(xf, '_spec') else 'Tensor'}")

        x = xf.type_as(x)
        print(f"[LN-DBG] After type cast: {x.shape}, type = {'DTensor' if hasattr(x, '_spec') else 'Tensor'}")

        if self.elementwise_scale:
            print(f"[LN-DBG] Applying elementwise scaling")
            x = self.weight * x

        if self.elementwise_shift:
            print(f"[LN-DBG] Applying elementwise shift")
            x = x + self.bias

        print(f"[LN-DBG] Output: {x.shape}, type = {'DTensor' if hasattr(x, '_spec') else 'Tensor'}")
        if hasattr(x, '_spec'):
            print(f"[LN-DBG] --> x._spec: placements = {x._spec.placements}, mesh = {x._spec.mesh}")
            print(f"[LN-DBG] --> x.to_local().shape = {x.to_local().shape}")
        return x
