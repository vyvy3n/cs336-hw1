import torch
from torch import nn
import math

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.W1 = nn.Parameter(torch.ones((d_ff, d_model), **factory_kwargs))
        self.W2 = nn.Parameter(torch.ones((d_model, d_ff), **factory_kwargs))
        self.W3 = nn.Parameter(torch.ones((d_ff, d_model), **factory_kwargs))

    
    def forward(self, x: torch.Tensor):
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        w1x = x @ self.W1.T # (batch_size, sequence_length, d_model) * (d_ff, d_model).T -> (batch_size, sequence_length, d_ff)
        silu = w1x * torch.sigmoid(w1x) # (batch_size, sequence_length, d_ff) * (batch_size, sequence_length, d_ff) -> (batch_size, sequence_length, d_ff)
        w3x = x @ self.W3.T # (batch_size, sequence_length, d_model) * (d_ff, d_model).T -> (batch_size, sequence_length, d_ff)
        return (silu * w3x) @ self.W2.T # (batch_size, sequence_length, d_ff) * (d_model, d_ff).T -> (batch_size, sequence_length, d_model)
        