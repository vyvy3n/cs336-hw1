import math

import torch
import torch.nn as nn

from .layers import Softmax, RotaryPositionalEmbedding, Linear, SwiGLU, RMSNorm
from einops import einsum, rearrange


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        """
        Scaled Dot-Product Attention.

        Computes softmax(Q K^T / sqrt(d_k)) V with optional masking.
        Mask is boolean where True allows attention and False masks it out.
        """
        super().__init__()
        self.softmax = Softmax(i=-1)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            Q: querys of shape (batch_size, ..., seq_len, d_k)
            K: keys of shape (batch_size, ..., seq_len, d_k)
            V: values of shape (batch_size, ..., seq_len, d_v)
            mask: Optional boolean tensor of shape (seq_len, seq_len).
                True means keep; False means mask out.
        Returns:
            Tensor of shape (batch_size, ..., seq_len, d_v)
        """
        d_k = Q.shape[-1]
        # Compute scaled dot-product scores using einops.einsum: (..., seq_q, seq_k)
        scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
        # scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)

        if mask is not None:
            # False entries get very negative to zero them after softmax; in-place for memory efficiency
            scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)

        # Attention weights along last dimension (keys)
        attn = self.softmax(scores)
        # Weighted sum of values: (..., seq_q, d_v)
        out = einsum(attn, V, "... n m, ... m d_v -> ... n d_v")  # i.e., attn @ V
        return out


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int,  # Dimensionality of the Transformer block inputs.
        num_heads: int,  # Number of heads to use in multi-headed attention.
        use_rope: bool,  # Whether to use RoPE to rotate queries and keys.
        max_seq_len: int | None = None,  # Maximum sequence length for RoPE.
        theta: float | None = None,   # RoPE parameter.
        device=None,  
        dtype=None):
        """
        Casual Multi-head self-attention module.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbedding(
                        theta=theta, 
                        d_k=d_model // num_heads, 
                        max_seq_len=max_seq_len, 
                        device=device) if use_rope else None
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional positions for RoPE, shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
    
        # Reshape for multi-head using einops
        Q = rearrange(Q, 'batch_size seq_len (h d_head) -> batch_size h seq_len d_head', h=self.num_heads)
        K = rearrange(K, 'batch_size seq_len (h d_head) -> batch_size h seq_len d_head', h=self.num_heads)
        V = rearrange(V, 'batch_size seq_len (h d_head) -> batch_size h seq_len d_head', h=self.num_heads)

        # Apply RoPE to Q and K
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Create causal mask: token i can attend to positions j <= i
        causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        # Apply scaled dot-product attention
        attn = ScaledDotProductAttention()
        out = attn(Q, K, V, mask=causal_mask)
        
        # Concatenate heads back using einops
        out = rearrange(out, 'b h s d -> b s (h d)')
        return self.o_proj(out)
    

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,  # Dimensionality of the Transformer block inputs.
        num_heads: int,  # Number of heads to use in multi-headed attention.
        d_ff: int,  # Dimensionality of the position-wise feedforward inner layer.
        use_rope: bool,  # Whether to use RoPE to rotate queries and keys.
        max_seq_len: int | None = None,  # Maximum sequence length for RoPE.
        theta: float | None = None, 
        token_positions: torch.Tensor | None = None,
        device=None, 
        dtype=None):
        """
        Transformer block with pre-norm.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = MultiheadSelfAttention(
                        d_model, 
                        num_heads, 
                        use_rope=use_rope, 
                        max_seq_len=max_seq_len, 
                        theta=theta, 
                        device=device, 
                        dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            token_positions: Optional positions for RoPE, shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,  # Size of the vocabulary.
        context_length: int,  # Context length, also maximum sequence length for RoPE.
        num_layers: int,  # Number of transformer blocks
        d_model: int,  # Dimensionality of the model embeddings and sublayer outputs.
        num_heads: int,  # Number of heads to use in multi-headed attention.
        d_ff: int,  # Dimensionality of the position-wise feedforward inner layer.
        use_rope: bool,  # Whether to use RoPE to rotate queries and keys.
        theta: float | None = None, 
        device=None, 
        dtype=None):
        """
        Transformer language model.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        # Token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, 
                num_heads, 
                d_ff, 
                use_rope, 
                max_seq_len=self.context_length,
                theta=theta, 
                device=device, 
                dtype=dtype)
            for _ in range(num_layers)
        ])

        # Final RMSNorm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.token_embeddings(x)  # (batch_size, seq_len, d_model)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)  # (batch_size, seq_len, d_model)

        # Final RMSNorm
        x = self.ln_final(x)  # (batch_size, seq_len, d_model)

        # Output projection
        return self.lm_head(x)  # (batch_size, seq_len, vocab_size)
