import torch
from torch import nn

## 1. noise layer 

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True, use_relative = False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.use_relative = use_relative
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            if self.use_relative:
                scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
                sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            else:
                sampled_noise = self.noise.expand(*x.size()).float().normal_() * self.sigma
            x = x + sampled_noise
        return x