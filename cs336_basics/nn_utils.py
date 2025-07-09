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
