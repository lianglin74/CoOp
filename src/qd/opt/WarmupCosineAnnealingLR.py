import torch
import math


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        min_lr=0,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
        cosine_restart_after_warmup=False,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.cosine_restart_after_warmup = cosine_restart_after_warmup
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

        from qd.qd_common import print_frame_info
        print_frame_info()

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [
                base_lr * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            if not self.cosine_restart_after_warmup:
                return [
                    self.min_lr + (base_lr - self.min_lr) *
                    (1 + math.cos(math.pi * self.last_epoch / self.max_iter)) / 2
                    for base_lr in self.base_lrs
                ]
            else:
                return [
                    self.min_lr + (base_lr - self.min_lr) *
                    (1 + math.cos(math.pi * (self.last_epoch -
                                             self.warmup_iters) /
                                  (self.max_iter - self.warmup_iters))) / 2
                    for base_lr in self.base_lrs
                ]

def create_warmup_cosine_annealing_lr(optimizer,
        T_max, warmup_factor, warmup_iters, warmup_method, last_epoch=-1):
    return WarmupCosineAnnealingLR(optimizer,
            T_max, 0., warmup_factor, warmup_iters, warmup_method, last_epoch=last_epoch)

