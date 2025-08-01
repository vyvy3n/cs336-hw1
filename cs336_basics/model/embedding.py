from torch import nn
import torch


class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        self.emb_matrix = nn.Parameter(torch.empty(size=(num_embeddings,embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.emb_matrix)
        
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb_matrix[token_ids]