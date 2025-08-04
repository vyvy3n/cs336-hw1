from einops import rearrange, reduce, einsum
import numpy as np
from cs336_basics.model.softmax import softmax 
import torch


def scaled_dot_product_attention(Q,K,V, mask=None):
    QtK = einsum(Q,K, 'batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k') / np.sqrt(Q.shape[-1])
    if mask is not None:
        QtK = QtK.masked_fill(~mask, float('-inf'))
    scores = softmax(QtK, dim=-1)  
    return einsum(scores, V, 'batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_v d_v -> batch_size ... seq_len_q d_v')
