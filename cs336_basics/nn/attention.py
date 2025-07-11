"""
Attention mechanisms and positional embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.nn.activations import softmax
from cs336_basics.nn.layers import Linear


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    RoPE applies rotation to pairs of embedding dimensions based on their position
    in the sequence. This provide relative positional information without requiring
    absolute positional embeddings.

    Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al, 2021)

    The rotation angle for position i and dimension pair k is:
    θ_{i,k} = i / Θ^{2k/d_k}

    The rotation matrix for each pair is
    R^k_i = [cos(θ_{i,k})  -sin(θ_{i,k})]
            [sin(θ_{i,k})   cos(θ_{i,k})]
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        """
        Initialize the RoPE module.

        Args:
            theta: θ value for RoPE (typically 10000)
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length for precomputing values
            device: Device to store buffers on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"

        # Precompute cos and sin values for all positions and dimension pairs
        # θ_{i,k} = i / Θ^{2k/d_k} for k ∈ {1, ..., d_k/2}

        # Create dimension indices: k ∈ {0, 1, 2, ..., d_k/2 - 1}
        # This corresponds to dimension pairs: (0,1), (2,3), (4,5), ...
        dim_indices = torch.arange(0, d_k // 2, dtype=torch.float32)

        # Compute frequency for each dimension pair: 1 / Θ^{2k/d_k}
        # Note: 2k/d_k = 2*k/d_k where k goes from 0 to d_k/2-1
        frequencies = 1.0 / (theta ** (2.0 * dim_indices / d_k))

        # Create position indices: i ∈ {0, 1, 2, ..., max_seq_len-1}
        positions = torch.arange(max_seq_len, dtype=torch.float32)

        # Compute all angles: θ_{i,k} = i * frequency_k
        # Shape: (max_seq_len, d_k//2)
        angles = torch.outer(positions, frequencies)

        # Precompute cos and sin values
        # Shape: (max_seq_len, d_k//2)
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)

        self.register_buffer("cos_values", cos_values.to(device), persistent=False)
        self.register_buffer("sin_values", sin_values.to(device), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor with arbitrary batch dimensions (..., seq_len, d_k)
            token_positions: Position indices for each token (..., seq_len)
                           Values should be in range [0, max_seq_len)

        Returns:
            Output tensor with same shape as input but with RoPE applied
        """
        cos_vals = self.cos_values[token_positions]
        sin_vals = self.sin_values[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation transformation
        # For each pair (x_even[k], x_odd[k]), apply rotation matrix:
        # [x_new_even[k]] = [cos(θ)  -sin(θ)] [x_even[k]]
        # [x_new_odd[k] ]   [sin(θ)   cos(θ)] [x_odd[k] ]
        #
        # This gives:
        # x_new_even[k] = cos(θ) * x_even[k] - sin(θ) * x_odd[k]
        # x_new_odd[k]  = sin(θ) * x_even[k] + cos(θ) * x_odd[k]
        x_new_even = cos_vals * x_even - sin_vals * x_odd
        x_new_odd = sin_vals * x_even + cos_vals * x_odd

        output = torch.zeros_like(x)
        output[..., 0::2] = x_new_even
        output[..., 1::2] = x_new_odd

        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}"


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... n_queries d_k"],
    K: Float[torch.Tensor, "... n_keys d_k"],
    V: Float[torch.Tensor, "... n_values d_v"],
    mask: Float[torch.Tensor, "... n_queries n_key"] | None = None,
) -> Float[torch.Tensor, "... n_queries d_v"]:
    """
    Scaled Dot-Product Attention mechanism.

    Computes attention weights between queries and keys, then uses these weights
    to compute a weighted average of the values. This is the core operation used
    in multi-headed attention layers.

    Paper: "Attention Is All You Need (Vaswani et al. 2017)"
    Formula: Attention(Q, K, V) = softmax(Q^T K / sqrt(d_k)) V

    Note: For column vectors, the formula becomes softmax(Q K^T / sqrt(d_k)) V
    due to PyTorch's row-major memory layout.

    Args:
        Q: Query tensor of shape (..., n_queries, d_k) where ... represents
           arbitrary batch dimensions (e.g., batch_size, num_heads)
        K: Key tensor of shape (..., n_keys, d_k) with same batch dimensions as Q
        V: Value tensor of shape (..., n_values, d_v) where n_values == n_keys
           and batch dimension match Q and K
        mask: Optional boolean mask of shape (..., n_queries, n_keys).
              True means attention is allowed, False means attention is blocked.
              When False, the attention score will be set to -inf before softmax.

    Returns:
        Output tensor with shape (..., n_queries, d_v) with same batch dimensions
        as input tensors
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q K^T / sqrt(d_k)
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)
    output = torch.einsum("...qk,...kv->...qv", attention_weights, V)
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention mechanism.

    This implementation follows the multi-head attention design from "Attention Is All You Need"
    (Vaswani et al., 2017) with causal masking to prevent attending to future tokens.

    The attention mechanism allows each position to attend to all positions up to and
    including itself, which is essential for autoregressive language modeling.

    Formula: MultiHeadSelfAttention(x) = W_O * MultiHead(W_Q x, W_K x, W_V x)
    where MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
    and head_i = Attention(Q_i, K_i, V_i)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the Multi-Head Self-Attention layer.

        Args:
            d_model: Dimensionality of the model (input/output dimension)
            num_heads: Number of attention heads
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head (d_k = d_v)
        self.d_v = d_model // num_heads

        factory_kwargs = {"device": device, "dtype": dtype}

        # Linear projections for all heads combined
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """
        Apply causal multi-head self-attention to input.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            rope: Optional RoPE module for positional encoding
            token_positions: Token positions for RoPE (required if rope is provided)

        Returns:
            Output tensor of same shape as input
        """
        # Get batch dimensions and sequence length
        *batch_dims, seq_len, d_model = x.shape
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

        # Project to Q, K, V for all heads at once
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to separate heads
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_v)

        # Transpose to put heads in batch dimension
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        if rope is not None and token_positions is not None:
            original_q_shape = Q.shape
            original_k_shape = K.shape

            # Flatten batch and head dimensions for RoPE application
            Q_flat = Q.reshape(-1, seq_len, self.d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)

            # Expand token_positions to match the flattened batch*head dimension
            pos_expanded = token_positions.unsqueeze(-2)
            pos_expanded = pos_expanded.expand(*batch_dims, self.num_heads, seq_len)
            pos_flat = pos_expanded.reshape(-1, seq_len)

            Q_flat = rope(Q_flat, pos_flat)
            K_flat = rope(K_flat, pos_flat)

            Q = Q_flat.reshape(original_q_shape)
            K = K_flat.reshape(original_k_shape)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = ~causal_mask

        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        attn_output = attn_output.transpose(-3, -2)
        attn_output = attn_output.reshape(*batch_dims, seq_len, self.d_model)

        output = self.output_proj(attn_output)
        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}"
