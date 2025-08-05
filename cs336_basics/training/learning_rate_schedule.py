import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


def cosine_learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:

    if t < T_w:
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        cosine_factor = 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi))
        return alpha_min + cosine_factor * (alpha_max - alpha_min)
    else:
        return alpha_min


def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    return cosine_learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c)


class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, alpha_max: float, alpha_min: float, T_w: int, T_c: int, last_epoch: int = -1):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.T_w = T_w
        self.T_c = T_c
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        t = self.last_epoch
        lr = cosine_learning_rate_schedule(t, self.alpha_max, self.alpha_min, self.T_w, self.T_c)
        return [lr for _ in self.base_lrs]
