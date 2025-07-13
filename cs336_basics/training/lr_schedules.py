"""
Learning rate scheduling utilities.
"""

from __future__ import annotations

import math


def cosine_learning_rate_schedule(
    iteration: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """
    Cosine annealing learning rate schedule with warmup.

    This implements the cosine learning rate schedule used in LLaMA and other modern
    language models. The schedule has three phases:
    1. Warmup: linearly increase from 0 to max_learning_rate
    2. Cosine annealing: cosine decay from max_learning_rate to min_learning_rate
    3. Post-annealing: constant at min_learning_rate

    Args:
        iteration: current iteration number
        max_learning_rate: maximum learning rate (at end of warmup)
        min_learning_rate: minimum learning rate (at end of cosine annealing)
        warmup_iters: number of warmup iterations
        cosine_cycle_iters: total number of iterations for cosine annealing

    Returns:
        learning rate for the current iteration
    """
    if iteration < warmup_iters:
        # Warmup phase: linear increase
        return max_learning_rate * iteration / warmup_iters
    elif iteration <= cosine_cycle_iters:
        # Cosine annealing phase
        progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + (max_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        # Post-annealing phase: constant at minimum
        return min_learning_rate
