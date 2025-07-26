from collections.abc import Iterable
import torch
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

                denom = v.div(1 - beta2 ** state["t"]).sqrt_().add_(group["eps"])

                p.data.addcdiv_(m.div(1 - beta1 ** state["t"]), denom, value=-group["lr"])

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
