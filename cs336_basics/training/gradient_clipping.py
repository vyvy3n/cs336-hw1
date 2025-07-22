"""
Gradient clipping utilities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by global L2 norm.

    This function clips the gradients of the given parameters to have a maximum L2 norm.
    If the global L2 norm of all gradients exceeds max_l2_norm, all gradients are
    scaled down proportionally.

    Args:
        parameters: iterable of parameters whose gradients should be clipped
        max_l2_norm: maximum L2 norm of the gradients
    """
    eps = 1e-6

    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)

    if total_norm > max_l2_norm:
        clip_coeff = max_l2_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coeff)
