import torch
from torch import nn
from einops import einsum, rearrange


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size=(in_features, out_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... i, i j -> ... j")

