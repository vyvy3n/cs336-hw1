"""
Neural network utility modules.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
