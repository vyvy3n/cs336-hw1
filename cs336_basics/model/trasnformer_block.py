from cs336_basics.model.poinwise_FFN import PoinwiseFFN
from cs336_basics.model.mhsa import MultiHeadSelfAttention
from cs336_basics.model.RMSNorm import RMSNorm
from torch import nn
import torch


class TransformerBlock(nn.Module):
    
    def __init__(self, d_model, num_heads, theta, max_seq_len, d_ff=None, mask=None, 
                 use_causal_mask=True, device=None):
        super().__init__()
        self.msha = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, 
                                         mask, device)
        self.ffn = PoinwiseFFN(d_model, d_ff=d_ff, device=device)
        self.rmsnorm1 = RMSNorm(d_model, device=device)
        self.rmsnorm2 = RMSNorm(d_model, device=device)
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        y = self.msha.forward(self.rmsnorm1.forward(x), token_positions)
        x_mid = x + y
        y_mid = self.ffn.forward(self.rmsnorm2.forward(x_mid))
        return x_mid + y_mid 

        
        