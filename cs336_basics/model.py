import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module, Parameter
from numpy import sqrt
from einops import einsum


@torch.no_grad()
def trunc_normal(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
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


torch.nn.Linear


class Linear(Module):
    """
    a Linear layer, compute the transformation of a batched input.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
    ):
        """
        Args:
            d_in (int): The size of the input dimension
            d_out (int): The size of the output dimension
        """
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        self.weights = Parameter(trunc_normal(torch.empty(d_out, d_in), std=1 / sqrt(d_in), a=-3, b=3))

    def forward(self, in_features: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        """
        Args:
            in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

        Returns:
            Float[Tensor, "... d_out"]: The transformed output of your linear module.
        """
        return einsum(in_features, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(Module):
    """an Embedding layer, get the embeddings for a batch of token ids."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ) -> Float[Tensor, " ... d_model"]:
        """
        Args:
            vocab_size (int): The number of embeddings in the vocabulary
            d_model (int): The size of the embedding dimension
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = Parameter(trunc_normal(torch.empty(vocab_size, d_model), a=-3, b=3))

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        """
        Args:
            token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

        Returns:
            Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
        """
        return self.weights[token_ids]


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        x(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `x` with the output of applying
        SiLU to each element.
    """
    return x / (1 + torch.exp(-x))


class SwiGLU(Module):
    """SwiGLU network"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(
        self,
        x: Float[Tensor, " ... d_model"],
    ) -> Float[Tensor, " ... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RMSNorm(Module):
    """RMSNorm"""

    def __init__(
        self,
        d_model: int,
        eps: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = Parameter(trunc_normal(torch.empty(d_model), a=-3, b=3))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Args:
            x (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
                dimensions.
        Returns:
            Float[Tensor,"... d_model"]: Tensor of with the same shape as `x` with the output of running
            RMSNorm of the `x`.
        """
        return x * self.weights / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
