"""
Complete Transformer model implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.nn.activations import SwiGLU
from cs336_basics.nn.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.layers import Embedding, Linear, RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with multi-head attention and SwiGLU feed-forward.

    This implementation follows the pre-norm design where layer normalization is applied
    before each sublayer (attention and feed-forward) rather than after. This design
    has been shown to improve training stability and is used in mordern language models.

    Architecture
    1. z = x + MultiHeadSelfAttention(RMSNorm(x))
    2. y = z + SwiGLU(RMSNorm(z))

    Paper: "On Layer Normalization in the Transformer Architecture" (Xiong et.al., 2020)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the Transformer block.

        Args:
            d_model: Dimensionality of the model (input/output dimension)
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            eps: Epsilon value for RMSNorm numerical stability
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.attn = MultiHeadSelfAttention(d_model, num_heads, **factory_kwargs)
        self.ln1 = RMSNorm(d_model, eps, **factory_kwargs)

        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, eps, **factory_kwargs)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        """
        Apply the Transformer block to input.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            rope: Optional RoPE module for positional encoding
            token_positions: Token positions for RoPE (required if rope is provided)

        Returns:
            Output tensor of same shape as input
        """
        normalized_x = self.ln1(x)
        attn_output = self.attn(normalized_x, rope=rope, token_positions=token_positions)
        z = x + attn_output

        normalized_z = self.ln2(z)
        ffn_output = self.ffn(normalized_z)
        y = z + ffn_output

        return y

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"d_model={self.d_model}, num_heads={self.num_heads}, d_ff={self.d_ff}"


class TransformerLM(nn.Module):
    """
    Transformer Language Model for autoregressive text generation.

    This implementation follows the decoder-only Transformer architecture used in
    modern language models like GPT. It combines token embeddimgs, multiple
    Transformer blocks, and a final linear projection to produce next-token predictions.

    Architecture:
    1. Token Embedding: Maps integer from token IDs to dense vectors.
    2. Multiple Transformer Blocks: Each with self-attention and feed-forward layers
    3. Final Layer Normalization: Applied after the last transformer block (pre-norm design)
    4. Language Model Head: Linear projection to vocabulary size for next-token prediction

    The model uses RoPE (Rotary Position Embedding) for positional information
    and casual masking to ensure tokens can only attend to previous positions.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ) -> None:
        """
        Initialize the Transformer Language Model.

        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            context_length: Maximum sequence length the model can process
            d_model: Dimensionality of the model (embedding and hidden dimensions)
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads (d_model must be divisible by num_heads)
            d_ff: Dimensionality of the feed-forward inner layer
            rope_theta: RoPE theta parameter (typically 10000.0)
            eps: Epsilon value for RMSNorm numerical stability
            device: Device to store parameters on
            dtype: Data type for parameters
        """
        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
        assert context_length > 0, f"context_length must be positive, got {context_length}"
        assert num_layers > 0, f"num_layers must be positive, got {num_layers}"

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_k, max_seq_len=context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, eps=eps, **factory_kwargs)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, eps, **factory_kwargs)

        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self, input_ids: Int[torch.Tensor, "... batch_size seq_len"]
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        """
        Forward pass of the Transformer Language Model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
                       Values should be integers in range [0, vocab_size)
                       seq_len should be <= context_length

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
            representing next-token predictions for each position
        """
        batch_size, seq_len = input_ids.shape

        assert seq_len <= self.context_length, (
            f"Input sequence length ({seq_len}) exceeds context length ({self.context_length})"
        )

        token_positions = torch.arange(seq_len, device=input_ids.device)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"vocab_size={self.vocab_size}, context_length={self.context_length}, "
            f"d_model={self.d_model}, num_layers={self.num_layers}, "
            f"num_heads={self.num_heads}, d_ff={self.d_ff}, "
            f"rope_theta={self.rope_theta}"
        )
