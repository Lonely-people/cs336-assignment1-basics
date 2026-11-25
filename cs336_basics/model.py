import torch
import torch.nn as nn
import einx
from math import sqrt

class Linear(nn.Module):
    def __init__(self,
                in_features: int,
                out_features: int,
                device: torch.device | None=None,
                dtype: torch.dtype | None=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.W=nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std=sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("... [d_in], d_out [d_in] -> ... d_out", x, self.W)