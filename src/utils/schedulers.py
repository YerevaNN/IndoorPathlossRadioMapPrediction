from typing import List

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float,
        warmup_end_lr: float,
        eta_min: float,
        last_epoch: int,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.eta_min = eta_min
        self.cosine_start_lr = warmup_end_lr
        self.cosine_end_lr = eta_min
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + (self.warmup_end_lr - self.warmup_start_lr)
                * (self.last_epoch / (self.warmup_epochs - 1))
                for _ in self.base_lrs
            ]
        else:
            return [
                self.eta_min + 0.5 * (self.warmup_end_lr - self.eta_min) * (1 + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                ))
                for _ in self.base_lrs
            ]
    
    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (self.warmup_end_lr - self.warmup_start_lr)
                / max(1, self.warmup_epochs - 1)
                for _ in self.base_lrs
            ]
        return [
            self.eta_min + 0.5 * (self.warmup_end_lr - self.eta_min) * (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            ))
            for _ in self.base_lrs
        ]
