import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange
from torch import Tensor
from typing import Optional

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        std = np.sqrt(2 / (in_features + out_features))

        weights = nn.init.trunc_normal_(torch.zeros(out_features, in_features, device=self.device, dtype=self.dtype), mean=0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(weights, requires_grad = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):

        super().__init__()

        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        weights = torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(weights, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return self.weight[token_ids]

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        self.eps = eps
        self.d_model = d_model
        # "gain" parameter
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        x_norm = x / rms * self.weight

        return x_norm.to(in_dtype)

class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):

        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        
        self.W1 = Linear(self.d_ff, self.d_model)
        self.W3 = Linear(self.d_ff, self.d_model)
        self.W2 = Linear(self.d_model, self.d_ff)

    def forward(self, x):

        silu = self.W1(x) * torch.sigmoid(self.W1(x))  
        return self.W2(silu * self.W3(x))

class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):

        super().__init__()
        self.device = device or torch.device('cpu')

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, "d_k must be even for RoPE"

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        positions = torch.arange(max_seq_len, device=device)

        # Outer product: [max_seq_len, d_k//2]
        angles = einsum(positions, inv_freq, "i, j-> i j")

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        # shape: (seq_len, d_k//2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # rearrange input into pairs: (x[1], x[2]), (x[3], x[4])
        x_pairs = rearrange(x, "... (split1 split2) -> ... split1 split2", split1=self.d_k//2, split2=2)

        rotated = torch.empty_like(x_pairs)

        rotated[..., 0] = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
        rotated[..., 1] = x_pairs[..., 0] * sin + x_pairs[..., 1] * cos

        return rotated.reshape(x.shape)

def softmax(tensor: torch.Tensor, dim: int):

    max_vals = torch.max(tensor, dim=dim, keepdim=True)[0]
    exp_tensor = torch.exp(tensor - max_vals)

    return exp_tensor / torch.sum(exp_tensor, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Q: (batch_size, ..., seq_len_q, d_k)
    K: (batch_size, ..., seq_len_k, d_k)
    V: (batch_size, ..., seq_len_k, d_v)
    mask: (seq_len_q, seq_len_k) boolean mask, True=attend, False=block
    """

    d_k = Q.shape[-1]

    att_logits = einsum(Q, K, "... q d, ... k d -> ... q k") / np.sqrt(d_k)

    if mask is not None:
        att_logits = att_logits.masked_fill(~mask, float("-inf"))

    att_weights = softmax(att_logits, dim=-1)

    output = einsum(att_weights, V, "... q k, ... k dv -> ... q dv")
    return output

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, apply_rope: bool, max_seq_len: Optional[int] = None, theta: Optional[float] = None, device=None, dtype=None):

        super().__init__()
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = self.d_model // num_heads
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype=dtype)
        if apply_rope and theta is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None
        
    def forward(self, x: torch.Tensor):

        B, T, _ = x.shape

        qkv = self.qkv_proj(x) # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1) # each (B, T, d_model)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=self.d_k)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads, d=self.d_k)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads, d=self.d_k)

        if self.rope is not None:
            positions = torch.arange(T, device=self.device)
            q = self.rope(q, positions)                         # (B, H, T, d_k)
            k = self.rope(k, positions)                         # (B, H, T, d_k)

        mask = torch.tril(torch.ones(T, T, device=self.device, dtype=torch.bool))


        result = scaled_dot_product_attention(q, k, v, mask)
        result = rearrange(result, "b n_h s d_k -> b s (n_h d_k)")
        result = self.output_proj(result)
        return result

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: Optional[int] = None, theta: Optional[float] = None, device=None, dtype=None):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.ln1 = RMSNorm(d_model, eps=1e-5, device=self.device, dtype=self.dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, apply_rope=True, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=self.device, dtype=self.dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # sub layer 1
        h = self.ln1(x)
        h = self.mha(h)
        x = x + h

        # sub layer 2
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h

        return x

class TransformerLM(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, max_seq_len: Optional[int] = None, theta: Optional[float] = None, device=None, dtype=None):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32

        self.theta = theta
        self.max_seq_len = max_seq_len if max_seq_len is not None else context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, max_seq_len=self.max_seq_len, theta=theta))

        self.ln_final = RMSNorm(d_model, eps=1e-5, device=self.device, dtype=self.dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    # FOR THE ADAPTER
    def load_state_dict(self, state_dict):
            # token embeddings
            self.token_embeddings.weight.data = state_dict['token_embeddings.weight']
            
            # layers
            for i in range(self.num_layers):
                layer = self.layers[i]
                layer.ln1.weight.data = state_dict[f'layers.{i}.ln1.weight']
                layer.mha.qkv_proj.weight.data = torch.cat([state_dict[f"layers.{i}.attn.q_proj.weight"], state_dict[f"layers.{i}.attn.k_proj.weight"], state_dict[f"layers.{i}.attn.v_proj.weight"]], dim=0)
                layer.mha.output_proj.weight.data = state_dict[f'layers.{i}.attn.output_proj.weight']
                layer.ln2.weight.data = state_dict[f'layers.{i}.ln2.weight']
                layer.ffn.W1.weight.data = state_dict[f'layers.{i}.ffn.w1.weight']
                layer.ffn.W2.weight.data = state_dict[f'layers.{i}.ffn.w2.weight']
                layer.ffn.W3.weight.data = state_dict[f'layers.{i}.ffn.w3.weight']
            
            # final layer norm and lm head
            self.ln_final.weight.data = state_dict['ln_final.weight']
            self.lm_head.weight.data = state_dict['lm_head.weight']
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (batch, seq_len)
        x = self.token_embeddings(x) # (B, T, d_model)
        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits