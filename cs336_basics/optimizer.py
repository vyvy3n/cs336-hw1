from collections.abc import Iterable
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step
from torch.nn import Parameter
import numpy as np


class AdamW(Optimizer):
    "AdamW optimizer."

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                p: Parameter
                grad = p.grad.data
                state: dict = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)

                state["t"] += 1

                m = state["m"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                v = state["v"]
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = v.sqrt().add_(group["eps"])

                alpha_t = np.sqrt(1 - beta2 ** state["t"]) / (1 - beta1 ** state["t"])

                p.data.addcdiv_(m, denom, value=-group["lr"] * alpha_t)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])


class CosineScheduler(LRScheduler):
    """CosineScheduler"""

    def __init__(
        self,
        optimizer: Optimizer,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
        last_epoch=-1,
        verbose="deprecated",
    ):
        super().__init__(optimizer, last_epoch, verbose)
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def get_lr(self) -> list[float]:
        """Compute learning rate using chainable form of the scheduler."""
        _warn_get_lr_called_within_step(self)
