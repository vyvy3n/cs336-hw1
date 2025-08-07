import torch
import numpy
import torch.nn as nn

from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float, Int

def softmax(in_features: Float[Tensor, "..."], dim: Int):
    """
    in_features: 
    dim: input dimensions
    """ 
    maxes = torch.max(in_features, dim=dim, keepdim=True)[0] # returns index and values, batch * dim --> batch * 1
    features_exp = torch.exp(in_features - maxes) # batch * dim --> batch * dum 
    return features_exp / torch.sum(features_exp, dim=dim, keepdim=True)