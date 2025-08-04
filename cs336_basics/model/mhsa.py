from einops import rearrange, reduce, einsum
from cs336_basics.model.scaled_attention import scaled_dot_product_attention
from cs336_basics.model.RoPe import RotaryPositionalEmbedding
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, mask=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.mask = mask
        
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        
        self.Wq = nn.Parameter(torch.empty(size=(d_model, d_model), device=device))
        self.Wk = nn.Parameter(torch.empty(size=(d_model, d_model), device=device))
        self.Wv = nn.Parameter(torch.empty(size=(d_model, d_model), device=device))
        self.Wo = nn.Parameter(torch.empty(size=(d_model, d_model), device=device))
        
        # Инициализация весов
        torch.nn.init.trunc_normal_(self.Wq)
        torch.nn.init.trunc_normal_(self.Wk)
        torch.nn.init.trunc_normal_(self.Wv)
        torch.nn.init.trunc_normal_(self.Wo)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        batch_size, seq_len, d_model = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        
        Q = einsum(x, self.Wq, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        K = einsum(x, self.Wk, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        V = einsum(x, self.Wv, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        
        Q_heads = rearrange(Q, 'batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k', num_heads=self.num_heads)
        K_heads = rearrange(K, 'batch_size seq_len (num_heads d_k) -> batch_size num_heads seq_len d_k', num_heads=self.num_heads)
        V_heads = rearrange(V, 'batch_size seq_len (num_heads d_v) -> batch_size num_heads seq_len d_v', num_heads=self.num_heads)
        
        Q_heads_rope = torch.zeros_like(Q_heads)
        K_heads_rope = torch.zeros_like(K_heads)
        
        for h in range(self.num_heads):
            Q_heads_rope[:, h, :, :] = self.rope.forward(Q_heads[:, h, :, :], token_positions)
            K_heads_rope[:, h, :, :] = self.rope.forward(K_heads[:, h, :, :], token_positions)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        if self.mask is not None:
            final_mask = causal_mask & self.mask
        else:
            final_mask = causal_mask
        attention_output = scaled_dot_product_attention(Q_heads_rope, K_heads_rope, V_heads, final_mask)
        attention_output = rearrange(attention_output, 'batch_size num_heads seq_len d_v -> batch_size seq_len (num_heads d_v)')
        output = einsum(attention_output, self.Wo, 'batch_size seq_len d_model, d_model d_model_out -> batch_size seq_len d_model_out')
        
        return output    

