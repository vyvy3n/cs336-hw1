import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module, Parameter
from numpy import sqrt
from einops import einsum


@torch.no_grad()
def trunc_normal(
    tensor: Tensor,
    mean: float = 0.0,
    std: float | None = None,
    a: float = -3.0,
    b: float = 3.0,
) -> Tensor:
    """Fill the input Tensor with values drawn from a truncated normal distribution.

    Args:
        tensor: `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value"""

    if std is None:
        std = 1 / sqrt(tensor.size(0))
    tensor.normal_(mean=mean).mul_(std).clamp_(min=a, max=b)
    return tensor


class Linear(Module):
    """a Linear layer, compute the transformation of a batched input.

    Args:
        d_in (int): The size of the input dimension
        d_out (int): The size of the output dimension"""

    def __init__(
        self,
        d_in: int,
        d_out: int,
    ):
        super().__init__()

        self.weights = Parameter(trunc_normal(torch.empty(d_out, d_in), std=1 / sqrt(d_in), a=-3, b=3))

    def forward(self, in_features: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        """
        Args:
            in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

        Returns:
            Float[Tensor, "... d_out"]: The transformed output of your linear module.
        """
        return einsum(in_features, self.weights, "... d_in, d_out d_in -> ... d_out")
