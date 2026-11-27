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

class SwiGLU(nn.Module):
    def __init__(self,
                d_model: int,
                d_ff: int | None=None,
                device: torch.device | None=None,
                dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model=d_model
        if d_ff is None:
            self.d_ff=int(8/3*d_model)
            self.d_ff=(self.d_ff+63)//64*64
        else:
            self.d_ff=d_ff
        self.W1=Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.W2=Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.W3=Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1=self.W1(x)
        w3=self.W3(x)
        silu=einx.multiply("... d_ff, ... d_ff -> ... d_ff", w1, torch.sigmoid(w1))
        glu=einx.multiply("... d_ff, ... d_ff -> ... d_ff", silu, w3)
        return self.W2(glu)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                theta: float,
                d_k: int,
                max_seq_len: int,
                device: torch.device | None=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        positions=torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs=1.0/(theta**(torch.arange(0, d_k, 2, device=device, dtype=torch.float32)/d_k))
        angles=einx.multiply("s, d -> s d", positions, freqs)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos=einx.get_at("[s] d, ... -> ... d", self.cos_cache, token_positions).to(x.dtype)
        sin=einx.get_at("[s] d, ... -> ... d", self.sin_cache, token_positions).to(x.dtype)
        x_even, x_odd=einx.id("... (d p) -> p ... d", x, p=2)
        res_even=x_even*cos-x_odd*sin
        res_odd=x_even*sin+x_odd*cos
        return einx.id("... d, ... d -> ... (d (1 + 1))", res_even, res_odd)
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_exp=torch.exp(x-x.max(dim=dim, keepdim=True).values)
    return x_exp/x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None=None) -> torch.Tensor:
    d_k=Q.shape[-1]
    score=einx.dot("... n d_k, ... m d_k -> ... n m", Q, K)/sqrt(d_k)
    if mask is not None:
        # score=einx.where("... n m, ... n m, -> ... n m", mask, score, float("-inf"))
        score=torch.where(mask, score, float("-inf"))
    sftmx=softmax(score, dim=-1)
    return einx.dot("... n m, ... m d_v -> ... n d_v", sftmx, V)

class MultiHead_self_Attention(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                device: torch.device | None=None,
                dtype: torch.dtype | None=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.d_v=d_model//num_heads
        self.Q_proj=Linear(d_model, num_heads*self.d_k, device=device, dtype=dtype)
        self.K_proj=Linear(d_model, num_heads*self.d_k, device=device, dtype=dtype)
        self.V_proj=Linear(d_model, num_heads*self.d_v, device=device, dtype=dtype)
        self.Output_proj=Linear(num_heads*self.d_v, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len=x.shape[-2]
        q=einx.id("... seq (h d_k) -> ... h seq d_k", self.Q_proj(x), h=self.num_heads)
        k=einx.id("... seq (h d_k) -> ... h seq d_k", self.K_proj(x), h=self.num_heads)
        v=einx.id("... seq (h d_v) -> ... h seq d_v", self.V_proj(x), h=self.num_heads)
        causal_mask=torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn=scaled_dot_product_attention(q, k, v, mask=causal_mask)
        result=einx.id("... h seq d_v -> ... seq (h d_v)", attn)
        return self.Output_proj(result)

class MultiHead_self_Attention_with_RoPE(MultiHead_self_Attention):
    def __init__(self, d_model, num_heads, theta: float, max_seq_len: int, device = None, dtype = None):
        super().__init__(d_model, num_heads, device, dtype)
        self.rope=RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len=x.shape[-2]
        q=einx.id("... seq (h d_k) -> ... h seq d_k", self.Q_proj(x), h=self.num_heads)
        k=einx.id("... seq (h d_k) -> ... h seq d_k", self.K_proj(x), h=self.num_heads)
        v=einx.id("... seq (h d_v) -> ... h seq d_v", self.V_proj(x), h=self.num_heads)
        q=self.rope(q, token_positions)
        k=self.rope(k, token_positions)
        causal_mask=torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn=scaled_dot_product_attention(q, k, v, mask=causal_mask)
        result=einx.id("... h seq d_v -> ... seq (h d_v)", attn)
        return self.Output_proj(result)