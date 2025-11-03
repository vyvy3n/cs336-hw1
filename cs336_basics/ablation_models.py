"""
Ablation-specific model variants for architecture experiments.

This module provides modified Transformer components for ablation studies:
- NoRMSNorm: Transformer without layer normalization
- PostNorm: Transformer with post-norm instead of pre-norm
- SiLUOnly: FFN with SiLU activation instead of SwiGLU
"""

import torch
import torch.nn as nn
from cs336_basics.layers import Linear, RMSNorm, Softmax, RotaryPositionalEmbedding


class SiLUFFN(nn.Module):
    """
    Simple FFN with SiLU activation (no gating).
    
    Computes: SiLU(x @ W1^T) @ W2^T
    
    This is used for the SwiGLU ablation to compare gated vs non-gated FFN.
    We match the parameter count by using d_ff = 4 * d_model.
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear projections (vs three in SwiGLU)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SiLU(x @ W1^T) @ W2^T
        """
        a = self.w1(x)
        a = a * torch.sigmoid(a)  # SiLU activation
        return self.w2(a)


class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        use_rope: bool,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,  
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbedding(
            theta=theta, 
            d_k=d_model // num_heads, 
            max_seq_len=max_seq_len, 
            device=device
        ) if use_rope else None
        
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.softmax = Softmax(i=-1, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        d_k = d_model // self.num_heads
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn = self.softmax(scores)
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.o_proj(out)


class TransformerBlockNoRMSNorm(nn.Module):
    """Transformer block WITHOUT layer normalization (ablation 1)."""
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool,
        ffn_type: str = "swiglu",  # "swiglu" or "silu"
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, use_rope, max_seq_len, theta, device, dtype
        )
        
        if ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device, dtype)
        else:
            from cs336_basics.layers import SwiGLU
            self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # No layer norm - just residual connections
        x = x + self.attn(x, token_positions)
        x = x + self.ffn(x)
        return x


class TransformerBlockPostNorm(nn.Module):
    """Transformer block with POST-NORM (ablation 2)."""
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool,
        ffn_type: str = "swiglu",
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, use_rope, max_seq_len, theta, device, dtype
        )
        
        if ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device, dtype)
        else:
            from cs336_basics.layers import SwiGLU
            self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Post-norm: apply norm AFTER residual connection
        x = self.ln1(x + self.attn(x, token_positions))
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerBlockSiLUOnly(nn.Module):
    """Transformer block with SiLU-only FFN (ablation 4)."""
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, use_rope, max_seq_len, theta, device, dtype
        )
        self.ffn = SiLUFFN(d_model, d_ff, device, dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Standard pre-norm with SiLU-only FFN
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLMAblation(nn.Module):
    """Transformer LM with support for ablations."""
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool,
        ablation_type: str = "none",  # "no_rmsnorm", "post_norm", "silu_only", "none"
        theta: float | None = None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.ablation_type = ablation_type

        # Token embedding
        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Choose block type based on ablation
        if ablation_type == "no_rmsnorm":
            block_class = TransformerBlockNoRMSNorm
        elif ablation_type == "post_norm":
            block_class = TransformerBlockPostNorm
        elif ablation_type == "silu_only":
            block_class = TransformerBlockSiLUOnly
        else:
            # Use standard pre-norm block
            from cs336_basics.models import TransformerBlock
            block_class = TransformerBlock
        
        # Create transformer blocks
        self.transformer_blocks = nn.ModuleList([
            block_class(
                d_model, num_heads, d_ff, use_rope,
                max_seq_len=context_length, theta=theta,
                device=device, dtype=dtype
            )
            for _ in range(num_layers)
        ])

        # Final norm (skip for no_rmsnorm ablation)
        if ablation_type != "no_rmsnorm":
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()

        # Output projection
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed tokens
        x = self.token_embeddings(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final norm
        x = self.ln_final(x)

        # Output projection
        return self.lm_head(x)

