import torch
from torch import nn, Tensor
import torch.nn.functional as F

class FFN_layer(nn.Module):
    def __init__(self,
                in_dim,
                out_dim,
                activation_fn = nn.ReLU(),
                dp_rate=None):
        super(FFN_layer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act_fn = activation_fn
        if dp_rate is not None:
            self.dp = nn.Dropout(dp_rate)
        else:
            self.dp = None
    def forward(self, x):
        x = self.act_fn(self.bn(self.fc(x)))
        if self.dp is not None:
            x = self.dp(x)
        return x
        