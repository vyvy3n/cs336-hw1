from torch import nn
import torch 
from einops import einsum, rearrange, reduce



class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gs = nn.Parameter(torch.empty(size=(d_model,), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.gs)

        
    def forward(self, x:torch.Tensor): # Shape (batch, seq, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x * x, 'b s d -> b s 1', 'mean') + self.eps)        
        result = x / rms * self.gs
        
        return result.to(in_dtype)
