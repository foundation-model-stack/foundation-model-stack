import torch.nn as nn


class UninitializedModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("I haven't been initialized yet!")

    def initialize(self, name) -> nn.Module:
        raise RuntimeError("I have to be replaced by a child class!")
