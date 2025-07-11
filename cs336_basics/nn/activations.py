"""
Activation functions and modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.nn.layers import Linear


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


def softmax(input: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    """
    Apply softmax function to the specified dimension of the input tensor.

    This implementation uses the numerically stable version that subtracts the maximum
    value from the input before applying exponential to prevent overflow.

    Formula: softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Args:
        input: Input tensor with arbitrary shape
        dim: Dimension along which to apply softmax. The resulting tensor
             will have the same shape as input, but the specified dimension
             will be normalized to sum to 1.

    Returns:
        Output tensor with same shape as input, with softmax applied along
        the specified dimension
    """
    max_vals = torch.max(input, dim=dim, keepdim=True)[0]
    input_stable = input - max_vals

    exp_vals = torch.exp(input_stable)
    sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)

    softmax_output = exp_vals / sum_exp
    return softmax_output


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """
    Compute cross-entropy loss with numerical stability.

    This implementation avoids explicitly computing softmax probabilities by using
    the log-sum-exp trick to prevent numerical overflow and underflow.

    The cross-entropy loss is computed as:
    CE = -log(softmax(logits)[target])
       = -log(exp(logits[target] - max(logits)) / sum(exp(logits - max(logits))))
       = -(logits[target] - max(logits)) + log(sum(exp(logits - max(logits))))
       = -logits[target] + log_sum_exp(logits)

    Args:
        logits: Unnormalized log probabilities with shape (..., vocab_size)
                where ... represents arbitrary batch dimensions
        targets: Target class indices with shape (...) mathcing the batch dimensions
                 of logits. Each value must be in [0, vocab_size-1]

    Returns:
        Scalar tensor containing the mean cross-entropy loss across all examples
    """
    vocab_size = logits.shape[-1]

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    batch_size = logits_flat.shape[0]

    max_logits = torch.max(logits_flat, dim=1, keepdim=True)[0]
    logits_stable = logits_flat - max_logits

    exp_logits = torch.exp(logits_stable)
    sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)

    batch_indices = torch.arange(batch_size, device=logits.device)
    target_logits = logits_stable[batch_indices, targets_flat]

    cross_entropy_losses = -target_logits + log_sum_exp.squeeze(1)

    mean_loss = torch.mean(cross_entropy_losses)
    return mean_loss
