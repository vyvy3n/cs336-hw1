import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module, Parameter
from numpy import sqrt
from einops import einsum, rearrange
from cs336_basics.nn_utils import softmax


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
    ):
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
        """
        Args:
            d_model (int): The dimensionality of the RMSNorm input.
            eps: (float): A value added to the denominator for numerical stability.
        """
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


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.size(-1)
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k)
    QK[~mask] = -torch.inf
    S = softmax(QK, dim=-1)
    return einsum(S, V, "... queries keys, ... keys d_k -> ... queries d_k")


class MultiheadAttention(Module):
    """MultiheadAttention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of batched multi-headed attention.
        """
        q = rearrange(self.q_proj(x), "... sequence_length (h d) -> ... h sequence_length d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... sequence_length (h d) -> ... h sequence_length d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... sequence_length (h d) -> ... h sequence_length d", h=self.num_heads)
        shape = q.shape[:-1]
        shape = [*shape, shape[-1]]  # (..., h, sequence_length, sequence_length)
        mask = torch.ones(shape, dtype=torch.bool, device=x.device)
        mask.tril_()
        a = scaled_dot_product_attention(q, k, v, mask)
        a = rearrange(a, "... h sequence_length d -> ... sequence_length (h d)", h=self.num_heads)
        o = self.o_proj(a)
        return o


class RotaryPositionalEmbedding(Module):
    """RotaryPositionalEmbedding"""

    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
    ):
        """
        Args:
            d_k (int): Embedding dimension size for the query or key tensor.
            theta (float): RoPE parameter.
            max_seq_len (int): Maximum sequence length to pre-cache.
        """
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        power = -2 * torch.arange(d_k // 2, dtype=torch.float32) / d_k
        inv_freq = theta ** power.unsqueeze(0)
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        angles = positions * inv_freq

        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sign", (-1) ** torch.arange(1, d_k + 1), persistent=False)
        self.register_buffer("index", torch.arange(d_k).reshape(-1, 2).flip(1).flatten())

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        """
        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        cos = self.cos[token_positions].repeat_interleave(2, -1)
        sin = self.sin[token_positions].repeat_interleave(2, -1)
        sin = sin * self.sign
        return cos * x + sin * x[..., self.index]


class RoPEMultiheadAttention(Module):
    """RotaryPositionalEmbedding MultiheadAttention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            theta (float): RoPE parameter.
            max_seq_len (int): Maximum sequence length to pre-cache.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(self.d_k, theta, max_seq_len)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        """
        Args:
            x (Float[Tensor, "... sequence_length d_model"]): Input tensor to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_model"]: Tensor with RoPEd input.
        """
        q = rearrange(self.q_proj(x), "... sequence_length (h d) -> ... h sequence_length d", h=self.num_heads)
        ro_q = self.rope(q, token_positions)
        k = rearrange(self.k_proj(x), "... sequence_length (h d) -> ... h sequence_length d", h=self.num_heads)
        ro_k = self.rope(k, token_positions)
        v = rearrange(self.v_proj(x), "... sequence_length (h d) -> ... h sequence_length d", h=self.num_heads)
        shape = ro_q.shape[:-1]
        shape = [*shape, shape[-1]]  # (..., h, sequence_length, sequence_length)
        mask = torch.ones(shape, dtype=torch.bool, device=x.device)
        mask.tril_()
        a = scaled_dot_product_attention(ro_q, ro_k, v, mask)
        a = rearrange(a, "... h sequence_length d -> ... sequence_length (h d)", h=self.num_heads)
        o = self.o_proj(a)
        return o


if __name__ == "__main__":
    mla = RoPEMultiheadAttention(16, 4, 10, 64).cuda()
    x = torch.randn(5, 7, 16).cuda()
    y = torch.arange(5 * 7).reshape(-1, 7).cuda()
    print(mla(x, y).shape)
