import torch
from torch import nn
from qd.qd_pytorch import all_gather_grad_curr


class KLEntropyLoss(nn.Module):
    # used for mask-rcnn trainer engine
    def __init__(self, entropy_weight):
        super().__init__()
        self.eps = 1e-8
        self.entropy_weight = entropy_weight
        self.iter = 0
        self.temperature = 1
        #self.uniform_type = 'kl'
        self.uniform_type = 'avg_entropy'

    def forward(self, logits, target):
        logits = nn.functional.normalize(logits)
        logits /= self.temperature
        prob = nn.functional.softmax(logits, dim=1)
        prob = all_gather_grad_curr(prob)

        debug = (self.iter % 10) == 0
        if debug:
            argmax_idx = prob.argmax(dim=1)
            unique, cnt = torch.unique(argmax_idx, return_counts=True)
            unique, idx = unique.sort()
            cnt = cnt[idx]
            import logging
            logging.info('unique: {}; cnt: {}'.format(unique, cnt))

        log_p = torch.log(prob + self.eps)
        p_log_p = prob * log_p
        entropy = -p_log_p.mean()
        if self.uniform_type == 'kl':
            kl_part = (prob.mean(dim=0) * log_p.mean(dim=0)).sum()
            uniform = entropy + kl_part
        elif self.uniform_type == 'avg_entropy':
            mean_prob = prob.mean(dim=0)
            uniform = (mean_prob * torch.log(mean_prob)).sum()
        else:
            raise NotImplementedError

        loss = {'kl': (1. - self.entropy_weight) * uniform,
                'entropy': self.entropy_weight * entropy}

        self.iter += 1
        return loss


