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

class Embedding(nn.Module):
    def __init__(self,
                num_embeddings: int,
                embedding_dim: int,
                device: torch.device | None=None,
                dtype: torch.dtype | None=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.weight=nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return einx.get_at("[vocab] d_model, ... -> ... d_model", self.weight, token_ids)

class RMSNorm(nn.Module):
    def __init__(self,
                d_model: int,
                eps: float = 1e-5,
                device: torch.device | None=None,
                dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=torch.sqrt(einx.mean("... [d_model] -> ...", x**2)+self.eps)
        x_norm=einx.divide("... d_model, ... -> ... d_model", x, rms)
        result=einx.multiply("... d_model, d_model -> ... d_model", x_norm, self.weight)
        return result.to(in_dtype)