"""
Neural network utility modules.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int


class Linear(nn.Module):
    """
    Linear transformation module without bias: y = Wx.

    This implementation follows the interface of PyTorch's nn.Linear but without bias.
    Uses truncated normal initialization with σ² = 2/(d_in + d_out).
    """

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Initialize the Linear layer.

        Args:
            in_features: Size of each input sample (final dimension)
            out_features: Size of each output sample (final dimension)
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using truncated normal distribution"""
        # σ² = 2/(d_in + d_out), so σ = sqrt(2/(d_in + d_out))
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        # Truncated at [-3σ, 3σ]
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, input: Float[torch.Tensor, " ... in_features"]) -> Float[torch.Tensor, " ... out_features"]:
        """
        Apply linear transformation to input.

        Args:
            input: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape(..., out_features)
        """
        return torch.einsum("...i,oi->...o", input, self.weight)

    def extra_repr(self):
        """String representation for debugging."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"


class Embedding(nn.Module):
    """
    Embedding lookup module that maps integer token IDs to dense vectors.

    This implementation follows the interface of PyTorch's nn.Embedding but
    implements the embedding lookup manually. Uses truncated normal initialization
    with σ² = 1 truncated at [-3, 3].
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the Embedding layer.

        Args:
            num_embeddings: Size of the vocabulary (number of embeddings)
            embedding_dim: Dimension of the embedding vectors (d_model)
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using truncated normal distribution"""
        # σ² = 1, so σ = 1.0, truncated at [-3σ, 3σ]
        with torch.no_grad():
            torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        """
        Lookup embedding vectors for the given token IDs.

        Args:
            token_ids: Input tensor of token IDs with arbitrary batch dimensions
                        Values should be integeres in range [0, num_embeddings]

        Returns:
            Embedded vectors of shape (..., embedding_dim) where ... matches
            the input batch dimension
        """
        return self.weight[token_ids]

    def extra_repr(self):
        """String representation for debugging."""
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes the input by its root mean square and applies a learnable
    elementwise affine transformation. Unlike LayerNorm, RMSNorm does not center
    the input by subtracting the mean.

    Paper: "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
    Formula: RMSNorm(x) = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mena(x^2) + eps)
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.device | None = None
    ) -> None:
        """
        Initialize the RMSNorm layer.

        Args:
            d_model: Hidden dimension of the model (size of the feature dimension)
            eps: Small constant added to denominator for numerical stability
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        """
        Apply RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape (..., d_model) where ... can be any number
                of batch dimensions (e.g., batch_size, sequence_length, d_model)

        Returns:
            Output tensor of the same shape as input with RMSNorm applied
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_squared = torch.mean(x.pow(2), dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + self.eps)

        normalized = x / rms
        result = normalized * self.weight

        return result.to(in_dtype)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, eps={self.eps}"


class SwiGLU(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU activation.

    SwiGLU combines the SiLu (Swish) activation functions with a Gated Linear Unit (GLU).
    This is used in modern language models like LLaMA and provides better performance
    than traditional ReLU-based feed-forward networks.

    Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
    Formula: SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) ⊙ W3 x)
    where SiLU(x) = x · σ(x) = x / (1 + e^(-x))
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the SwiGLU feed-forward network.

        Args:
            d_model: Input/output dimension (hidden dimension of the model)
            d_ff: Inner dimension of the feed-forward network. If None, defaults to
                  approximately 8/3 * d_model rounded to the nearest multiple of 64
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model

        # Set d_ff to approximately 8/3 * d_model, rounded to multiple of 64
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # Round to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)  # Gate projection
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)  # Output projection
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)  # Value projection

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        """
        Apply SwiGLU transformation to input.

        Args:
            x: Input tensor of shape (..., d_model) where ... can be any number
               of batch dimensions (e.g., batch_size, sequence_length, d_model)

        Returns:
            Output tensor of the same shape as input with SwiGLU applied
        """
        gate = self.w1(x)
        value = self.w3(x)

        silu_gate = gate * torch.sigmoid(gate)
        gated_value = silu_gate * value

        output = self.w2(gated_value)
        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
