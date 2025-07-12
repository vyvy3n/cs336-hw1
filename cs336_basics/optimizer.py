"""
AdamW Optimizer implementation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import IO, Any, BinaryIO

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Implements AdamW algorithm (decoupled weight decay).

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    AdamW is proposed in `Decoupled Weight Decay Regularization`_.

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """
        Initialize the AdamW optimizer.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr: learning rate (default: 1e-3)
            betas: coefficients used for computing running averages of gradient
                and its square (default: (0.9, 0.999))
            eps: term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay: weight decay coefficient (default: 0.01)

        Raises:
            ValueError: If any of the parameters have invalid values:
                - lr is negative
                - eps is negative
                - beta1 or beta2 is not in [0,1)
                - weight_decay is negative
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

        return loss


def cosine_learning_rate_schedule(
    iteration: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """
    Compute the learnig rate for a given iteration using cosine annealing with warmup.

    This implements the cosine learning rate schedule:
    - Warm-up phase: Linear increase from 0 to max_learning_rate
    - Cosine annealing phase: Cosine decay from max_learning to min_learning_rate
    - Post-annealing phase: Constant at min_learning_rate

    Args:
        iteration: Current iteration number (0-indexed)
        max_learning_rate: Maximum learning_rate (α_max)
        min_learning_rate: Minimum learning_rate (α_max)
        warmup_iters: Number of warmup iterations (T_w)
        cosine_cycle_iters: Total number of cosine cycle iterations (T_c)

    Returns:
        Learning rate for the given iteration

    Raises:
        ValueError: If any parameter is invalid
    """
    if iteration < 0:
        raise ValueError(f"Iteration must be non-negative, got {iteration}")
    if max_learning_rate < 0:
        raise ValueError(f"Max learning rate must be non-negative, got {max_learning_rate}")
    if min_learning_rate < 0:
        raise ValueError(f"Min learning rate must be non-negative, got {min_learning_rate}")
    if warmup_iters < 0:
        raise ValueError(f"Warmup iterations must be non-negative, got {warmup_iters}")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError(
            f"Cosine cycle iterations ({cosine_cycle_iters}) must be >= warmup iterations ({warmup_iters})"
        )

    if iteration < warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        return (iteration / warmup_iters) * max_learning_rate

    elif iteration <= cosine_cycle_iters:
        if cosine_cycle_iters == warmup_iters:
            return min_learning_rate

        progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)

    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients to have L2 norm at most max_l2_norm.

    This function implements gradient clipping.
    Given the gradient for all parameters g, we compute its L2-norm ||g||_2.
    If this norm is less than max_l2_norm, we leave g as is; otherwise, we scale
    g down by a factor of max_l2_norm / (||g||_2 + ε) where ε = 1e-6.

    Args:
        parameters: Iterable of parameters whose gradients should be clipped
        max_l2_norm: Maximum L2 norm allowed for the combined gradients

    Note:
        This function modified the parameter gradients in-place.
    """
    if max_l2_norm <= 0:
        raise ValueError(f"max_l2_norm must be positive, got {max_l2_norm}")

    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad.data)

    if not gradients:
        return

    total_norm_squared = torch.sum(torch.stack([torch.sum(g * g) for g in gradients]))
    total_norm = torch.sqrt(total_norm_squared)

    eps = 1e-6

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model, optimizer state, and iteration number to a checkpoint file.

    This function serializes the current training state to enable resumption
    of training from a specific point. The checkpoint includes all information
    needed to restore the exact training state.

    Args:
        model: The PyTorch model to save
        optimizer: The optimizer to save (including its internal state)
        iteration: Current training iteration number
        out: Path or file-like object to save the checkpoint to

    Raises:
        RuntimeError: If there's an error during serialization
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    try:
        torch.save(checkpoint, out)
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint {e}") from e


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    """
    Load model, optimizer state, and iteration number from a checkpoint file.

    This function restores the training state from a checkpoint, allowing
    training to resume from where it left off. The model and optimizer
    states are modified in-place.

    Args:
        src: Path or file-like object to load the checkpoint from
        model: The PyTorch model to restore state to
        optimizer: The optimizer to restore state to

    Returns:
        The iteration number that was saved in the checkpoint

    Raises
        RuntimeError: If there's an error during deserialization or loading
        KeyError: If the checkpoint is missing required keys
    """
    try:
        checkpoint = torch.load(src, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    required_keys = {"model_state_dict", "optimizer_state_dict", "iteration"}
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise KeyError(f"Checkpoint missing required keys: {missing_keys}")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["iteration"]
    except Exception as e:
        raise RuntimeError(f"Failed to restore checkpoint state: {e}") from e
