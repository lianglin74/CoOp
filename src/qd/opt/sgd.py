from torch.optim.sgd import SGD
import logging


class SGDVerbose(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        debug = ((self.iter % 100) == 0)

        if debug:
            loss_wd = 0
            num_param = 0
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    loss_wd += (p.data * p.data).sum() * weight_decay
                    num_param += 1
            loss_wd *= 0.5
            logging.info('weight decay loss = {:.4f}/{}'.format(
                loss_wd, num_param))
        self.iter += 1


