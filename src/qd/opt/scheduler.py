from torch.optim.lr_scheduler import LambdaLR


class WarmupLinearSWASchedule(LambdaLR):
    # this is kind of different from swa implemented in pytorch
    def __init__(self, optimizer, warmup_steps, swa_start_steps, t_total,
                 swa_const_ratio, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.swa_start_steps = swa_start_steps
        self.t_total = t_total
        self.swa_const_ratio = swa_const_ratio
        self.last_epoch = last_epoch
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        elif step < self.swa_start_steps:
            alpha = (step - self.warmup_steps) * 1. / (self.swa_start_steps - self.warmup_steps)
            return (1. - alpha) * 1. + alpha * self.swa_const_ratio
        else:
            return self.swa_const_ratio


