from qd.torch_common import sample_token
import logging
from torch import nn


class TokenSample(nn.Module):
    def __init__(self, num, method=None):
        super().__init__()
        self.num = num
        self.method = method
        self.iter = 0
        self.keep = 0
        self.total = 0

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "num=" + str(self.num)
        tmpstr += ")"
        return tmpstr

    def forward(self, x):
        verbose = (self.iter % 100) == 0
        self.iter += 1
        B, N = x.shape[:2]

        if N <= self.num:
            self.keep += N
            self.total += N
        elif self.method is None:
            self.total += N
            self.keep += self.num
            x = sample_token(x, self.num)
        else:
            raise NotImplementedError
        if verbose:
            logging.info('kept = {}, {}/{}'.format(
                self.keep * 1. / self.total,
                self.keep, self.total,
            ))
        return x


