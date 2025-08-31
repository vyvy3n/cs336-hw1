from jaxtyping import Float, Int
from torch import Tensor
import torch
import math
from typing import Optional, Any

def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]):

    # flatten batch + sequence into a single dimension
    inputs = inputs.reshape(-1, inputs.size(-1))   # shape (batch_size*seq_len, vocab_size)
    targets = targets.reshape(-1)             # shape (batch_size*seq_len,)

    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    inputs = inputs - inputs_max

    exp_inputs = torch.exp(inputs)
    exp_inputs_sum = torch.sum(exp_inputs, dim=-1, keepdim=True)

    batch_indices = torch.arange(inputs.shape[0])
    input_targets = inputs[batch_indices, targets]  # shape (batch_size,)
    
    log_probs = input_targets - torch.log(exp_inputs_sum) 

    losses = -log_probs

    return losses.mean()

def get_cosine_lr(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:

    if t < Tw:
        alpha_t = (t/Tw) * alpha_max
    
    elif Tw <= t <= Tc:
        alpha_t = alpha_min + (1/2) * (1 + math.cos(((t - Tw)/(Tc - Tw)) * math.pi)) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min

    return alpha_t

def gradient_clipping(params: list[torch.Tensor], M: float, eps: float = 1e-6) -> None:
    # Flatten all gradients into a single vector norm
    total_norm = torch.sqrt(sum((p.grad.detach()**2).sum() for p in params if p.grad is not None) + eps)

    clip_coef = M / (total_norm + eps)
    if clip_coef < 1.0:
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)

def save_checkpoint(model, optimizer, iteration, out, run_info: dict[str, Any] | None = None):

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'run_info': run_info
    }

    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):

    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']