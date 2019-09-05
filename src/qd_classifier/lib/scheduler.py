from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right

import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

class WarmupScheduler(_LRScheduler):
    """
    NOTE: This scheduler should be updated by epoch
    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        warmup_epochs: target learning rate is reached at warmup_epochs, gradually
        after_scheduler: after warmup_epochs, use this scheduler(eg. StepLR)
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler,
                 warmup_lr=0, last_epoch=-1, warmup_mode="linear"):
        self.warmup_epochs = warmup_epochs
        self.warmup_mode = warmup_mode
        self.warmup_lr = warmup_lr
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]
                self.after_scheduler.last_epoch = 0
                self.finished = True
            else:
                self.after_scheduler.last_epoch = self.last_epoch - self.warmup_epochs -1
            return self.after_scheduler.get_lr()

        if self.warmup_mode == "linear":
            return [self.warmup_lr + (base_lr - self.warmup_lr) * (self.last_epoch) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            raise NotImplementedError("invalid warmup mode: {}".format(self.warmup_mode))

if __name__ == "__main__":
    import torch.optim.lr_scheduler as lr_scheduler
    import torch.nn as nn
    import torch.optim

    model = nn.Conv2d(1, 2, 1, bias=False)
    init_lr = 0.4
    lr_mode = 'cosine'
    num_epochs = 20
    num_gpus = 8
    batch_size = 128 * num_gpus
    classes = 1000
    num_training_samples = 1281167
    num_batches = num_training_samples // batch_size
    warmup_epochs = 5

    start_epoch = 9

    groups = [dict(params=list(model.parameters()), initial_lr=init_lr)]
    optimizer = torch.optim.SGD(groups, init_lr,
                                momentum=0.9, weight_decay=1e-4, nesterov=True)

    sch = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1,
                                        last_epoch=start_epoch-1)
    # sch = lr_scheduler.CosineAnnealingLR(optimizer, 3)
    scheduler = WarmupScheduler(optimizer, warmup_epochs, after_scheduler=sch, last_epoch=start_epoch-1)
    for epoch in range(num_epochs):
        scheduler.step()
        print('Epoch[%d]\tlr=%s' % (start_epoch + epoch, str([group["lr"] for group in optimizer.param_groups])))
