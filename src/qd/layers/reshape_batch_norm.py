import torch


class ReshapeSyncBatchNorm(torch.nn.SyncBatchNorm):
    def __init__(self, *args, **kwargs):
        super(ReshapeSyncBatchNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        num_unsqueeze = 0
        while x.dim() < 3:
            x = x.unsqueeze(-1)
            num_unsqueeze += 1
        x = super(ReshapeSyncBatchNorm, self).forward(x)
        for i in range(num_unsqueeze):
            x = x.squeeze(-1)
        return x

class ReshapeBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(ReshapeBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        num_unsqueeze = 0
        while x.dim() < 4:
            x = x.unsqueeze(-1)
            num_unsqueeze += 1
        x = super(ReshapeBatchNorm2d, self).forward(x)
        for i in range(num_unsqueeze):
            x = x.squeeze(-1)
        return x

def ensure_shape_bn_layer(norm, s, is_pre_linear):
    if not is_pre_linear:
        return norm(s)
    elif norm is torch.nn.SyncBatchNorm:
        return ReshapeSyncBatchNorm(s)
    elif norm is torch.nn.BatchNorm2d:
        return ReshapeBatchNorm2d(s)
    else:
        raise NotImplementedError()

