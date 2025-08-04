from einops import rearrange, reduce, einsum
import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        k_indices = torch.arange(0, d_k // 2, device=device, dtype=torch.float32)
        freqs = theta / (10000.0 ** (2 * k_indices / d_k)) 
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) 
        angles = einsum(positions, freqs, 'pos, freq -> pos freq') 
        cos_vals = torch.cos(angles)  
        sin_vals = torch.sin(angles)  
        
        self.register_buffer('cos_vals', cos_vals, persistent=False)
        self.register_buffer('sin_vals', sin_vals, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_pairs = rearrange(x, '... seq (pairs two) -> ... seq pairs two', two=2)
        x_reshaped = rearrange(x_pairs, '... seq pairs two -> two ... seq pairs')
        x_even, x_odd = x_reshaped[0], x_reshaped[1]
        cos = self.cos_vals[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_vals[token_positions]  # (..., seq_len, d_k//2)
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        x_stacked = torch.stack([x_rot_even, x_rot_odd], dim=0)  
        x_rotated = rearrange(x_stacked, 'two ... seq pairs -> ... seq (pairs two)')
        
        return x_rotated