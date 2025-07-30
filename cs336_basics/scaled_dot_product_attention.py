import torch
from torch import nn, Tensor
from jaxtyping import Float
from cs336_basics.softmax import softmax
import math

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    dot_product = Q @ K.mT/math.sqrt(K.shape[-1])

    if mask is not None:
        mask_tensor = torch.zeros_like(mask).type_as(Q)
        mask_tensor = mask_tensor.masked_fill(~mask, float('-inf'))
        dot_product = dot_product + mask_tensor

    return softmax(dot_product, -1) @ V
