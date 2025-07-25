from collections.abc import Iterable
from torch.optim import Optimizer
from torch.nn import Parameter


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
        super().__init__(params, dict(l=lr, weight_decay=weight_decay, betas=betas, eps=eps))

    def step(self):
        for group in self.param_groups:
            pass
