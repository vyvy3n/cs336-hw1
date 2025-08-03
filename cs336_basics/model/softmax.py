from einops import rearrange, reduce, einsum
import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    result = exp_x / sum_exp
    
    return result