import torch 
from torch import nn 
from einops import repeat

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.trasnformer_block import TransformerBlock
from cs336_basics.model.RMSNorm import RMSNorm


class GPTZero(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, 
                 d_model: int = 512, num_heads: int = 8, d_ff: int = None,
                 theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.position_embedding = Embedding(context_length, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads, 
                theta=theta,
                max_seq_len=context_length,
                d_ff=d_ff,
                device=device
            ) 
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bs, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        token_embeds = self.token_embedding(input_ids) 
        pos_embeds = self.position_embedding(pos_ids) 
        x = token_embeds + pos_embeds
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, pos_ids) 
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits