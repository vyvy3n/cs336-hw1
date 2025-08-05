import torch
from typing import List


def gradient_clipping(parameters: List[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> None:
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    if not gradients:
        return
    total_norm = torch.sqrt(sum(torch.sum(grad * grad) for grad in gradients))
    if total_norm > max_norm:
        clip_coeff = max_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coeff)