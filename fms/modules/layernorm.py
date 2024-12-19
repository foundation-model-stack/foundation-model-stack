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
        if self.use_mean:
            x = x - x.mean(-1, keepdim=True)
        # x = F.normalize(x, dim=-1)*math.sqrt(x.size(-1))
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        # xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)

        # hpml_ibm
        if not xf.is_nested: 
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            xf_list = []
            for ele in xf.unbind(0):
                temp = ele.pow(2).mean(-1, keepdim=True)
                temp = temp + self.eps
                temp = torch.rsqrt(temp)
                temp = ele * temp 
                xf_list.append(temp)

            xf = torch.nested.nested_tensor(xf_list)
            
        # x = xf.type_as(x)

        # hpml_ibm
        if not xf.is_nested:
            x = xf.type_as(x)
        else:
            first_vector = xf.unbind(0)[0]
            x = torch.nested.nested_tensor([ele.type_as(first_vector) for ele in xf.unbind(0)])

        # if self.elementwise_scale:
        #     x = self.weight * x
        # if self.elementwise_shift:
        #     x = x + self.bias

        # hpml_ibm
        if not x.is_nested:
            if self.elementwise_scale:
                x = self.weight * x
            if self.elementwise_shift:
                x = x + self.bias
        else:
            if self.elementwise_scale:
                x = torch.nested.nested_tensor([self.weight * ele for ele in x.unbind(0)])
            if self.elementwise_shift:
                x = torch.nested.nested_tensor([self.bias + ele for ele in x.unbind(0)])

        return x
