import torch 
from torch import nn 
from einops import reduce, rearrange, einsum


class PoinwiseFFN(nn.Module):
    
    
    def __init__(self, d_model, d_ff=None, device=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8/3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64 
        self.d_ff = d_ff 
        self.W1 = nn.Parameter(torch.empty(size=(self.d_ff, d_model), device=device))
        self.W3 = nn.Parameter(torch.empty(size=(self.d_ff, d_model), device=device))
        self.W2 = nn.Parameter(torch.empty(size=(d_model, self.d_ff), device=device))
        nn.init.trunc_normal_(self.W1)
        nn.init.trunc_normal_(self.W2)
        nn.init.trunc_normal_(self.W3)

    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


    def forward(self, x: torch.Tensor):
        a1 = self.SiLU(einsum(x, self.W1, "... d, f d -> ... f"))
        a2 = einsum(x, self.W3, "... d, f d -> ... f")
        a = a1 * a2
        return einsum(a, self.W2, "... f, d f -> ... d")
