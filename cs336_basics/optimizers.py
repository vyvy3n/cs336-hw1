from collections.abc import Callable, Iterable 
from typing import Optional 
import math
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, device=None):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, vocab_size).
                logits[i][j] is the unnormalized logit of jth class for the ith example.
            targets (torch.Tensor): Target class indices of shape (batch_size,).
                Each value must be between 0 and vocab_size - 1.

        Returns:
            torch.Tensor: Scalar tensor containing the average cross-entropy loss across the batch.

        The loss is computed as:
            -log(softmax(logits)[target])

        For numerical stability:
            - Subtract the maximum logit value before computing softmax
            - Cancel out log and exp whenever possible
        """
        # Get the maximum logit for each example for numerical stability
        # Shape: (batch_size, 1)
        max_logits = torch.amax(logits, dim=-1, keepdim=True)

        # Subtract max for numerical stability
        # Shape: (batch_size, vocab_size)
        shifted_logits = logits - max_logits

        # Compute log-sum-exp: log(sum(exp(shifted_logits)))
        # Shape: (batch_size,)
        log_sum_exp = torch.logsumexp(shifted_logits, dim=-1)

        # Get the logit for the target class
        # Shape: (batch_size,)
        target_logits = logits[torch.arange(logits.shape[0]), targets]

        # Compute cross-entropy loss for each example
        # loss_i = log_sum_exp - (target_logit - max_logit)
        # This is equivalent to -log(softmax(logits)[target])
        # Shape: (batch_size,)
        losses = log_sum_exp - (target_logits - max_logits.squeeze(-1))

        # Return the average loss across the batch
        return losses.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """
        AdamW optimizer with decoupled weight decay.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            lr: Learning rate (default: 1e-3)
            betas: Tuple of (beta1, beta2) coefficients for computing running averages of gradient
                   and its square (default: (0.9, 0.999)); LLMs like llama and GPT-3 often use (0.9, 0.95)
            eps: Term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)  # First moment estimate
                    state["v"] = torch.zeros_like(p.data)  # Second moment estimate

                # Get state variables
                m = state["m"]
                v = state["v"]

                # Increment iteration counter
                state["t"] += 1

                # Update biased first moment estimate: 
                # m = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate: 
                # v = beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate:
                # alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                alpha_t = lr * math.sqrt(1 - beta2 ** state["t"]) / (1 - beta1 ** state["t"])

                # Update parameters: 
                # theta = theta - alpha_t * m / (sqrt(v) + eps)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                # Apply weight decay (decoupled from gradient update)
                # theta = theta - lambda * lr * theta
                p.data.mul_(1 - weight_decay * lr)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    The schedule has three phases:
    1. Warm-up (0 <= t < T_w): Linear increase from 0 to max_learning_rate
    2. Cosine annealing (T_w <= t <= T_c): Cosine decay from max to min learning rate
    3. Post-annealing (t > T_c): Constant at min_learning_rate

    Args:
        it (int): Current iteration number (t).
        max_learning_rate (float): alpha_max, the maximum learning rate.
        min_learning_rate (float): alpha_min, the minimum/final learning rate.
        warmup_iters (int): T_w, the number of iterations to linearly warm-up.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        float: Learning rate at the given iteration.
    """
    # Warm-up phase: linear increase from 0 to max_learning_rate
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # Cosine annealing phase
    elif it <= cosine_cycle_iters:
        # Calculate the cosine annealing factor
        # Formula: alpha_t = alpha_min + 0.5 * (1 + cos((t - T_w) * pi / (T_c - T_w))) * (alpha_max - alpha_min)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)

    # Post-annealing phase: constant at min_learning_rate
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable, max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Clip gradients of parameters to have L2 norm at most max_l2_norm, modifies in-place.

    Args:
        parameters (Iterable): Iterable of parameters (typically model.parameters()).
        max_l2_norm (float): Maximum L2 norm for the gradients.
        eps (float): Small value added for numerical stability (default: 1e-6).

    The algorithm:
    1. Compute the total L2 norm: ||g||_2 = sqrt(sum(||g_i||_2^2)) for all parameter gradients
    2. If ||g||_2 > max_l2_norm, scale all gradients by: max_l2_norm / (||g||_2 + eps)
    """
    # Collect all gradients from parameters that have gradients
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)

    # If no gradients, nothing to clip
    if len(gradients) == 0:
        return

    # Compute the total L2 norm of all gradients
    # ||g||_2 = sqrt(sum(||g_i||_2^2))
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in gradients))

    # Compute the clipping factor
    # If total_norm <= max_l2_norm, clip_factor >= 1, so no clipping occurs
    # If total_norm > max_l2_norm, clip_factor < 1, so gradients are scaled down
    clip_factor = max_l2_norm / (total_norm + eps)

    # Only clip if the norm exceeds the maximum
    if clip_factor < 1.0:
        for grad in gradients:
            grad.mul_(clip_factor)
