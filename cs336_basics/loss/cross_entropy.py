"""
Cross-entropy loss function.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """
    Compute cross-entropy loss between logits and targets.

    This function computes the cross-entropy loss with numerical stability by
    subtracting the maximum value from logits before applying softmax.

    Args:
        logits: predicted logits of shape (..., vocab_size)
        targets: target indices of shape (...)

    Returns:
        scalar cross-entropy loss averaged over the batch
    """
    # Move targets to the same device as logits
    targets = targets.to(logits.device)

    # Subtract max for numerical stability
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits_stable = logits - logits_max

    # Compute log softmax manually for numerical stability
    log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
    log_softmax = logits_stable - log_sum_exp

    # Gather the log probabilities for the target indices
    target_log_probs = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Return negative log likelihood (cross entropy)
    return -target_log_probs.mean()
