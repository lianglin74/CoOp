from torch import nn


class KLDivLogitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.iter = 0

    def forward(self, data, target):
        self.iter += 1
        target = target.view(*data.shape)
        debug = (self.iter % 100) == 0
        if debug:
            import logging
            logging.info('max conf: {}'.format(target.max(dim=1)[0].mean()))
        data = self.log_soft(data)
        loss = self.kl(data, target)
        return loss

