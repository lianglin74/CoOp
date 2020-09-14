raise ValueError('no need of a specific layer. train the model with lr=0')
import torch
from qd.layers.reshape_batch_norm import ReshapeBatchNorm2d
from qd.layers.reshape_batch_norm import ReshapeSyncBatchNorm


class PreciseSyncBatchNorm(torch.nn.SyncBatchNorm):
    def __init__(self, bn):
        super().__init__(
                num_features=bn.num_features,
                eps=bn.eps,
                momentum=bn.momentum,
                affine=bn.affine,
                track_running_stats=bn.track_running_stats,
                process_group=bn.process_group,
                )

    def forward(self, x):
        # calculate the output with running mean/running var
        self.train(mode=False)
        y = super().forward(x)

        # update the parameters and ignore the output
        self.train(mode=True)
        super().forward(x)
        return y

def create_precise_bn2(bn):
    if type(bn) is torch.nn.BatchNorm2d:
        return PreciseBatchNorm2d(bn)
    elif type(bn) is torch.nn.SyncBatchNorm:
        return PreciseSyncBatchNorm(bn)
    elif type(bn) is ReshapeBatchNorm2d:
        return PreciseReshapeBatchNorm2d(bn)
    elif type(bn) is ReshapeSyncBatchNorm:
        return PreciseReshapeSyncBatchNorm(bn)
    else:
        raise NotImplementedError

class PreciseReshapeSyncBatchNorm(ReshapeSyncBatchNorm):
    def __init__(self, bn):
        super().__init__(num_features=bn.num_features,
                eps=bn.eps,
                momentum=bn.momentum,
                affine=bn.affine,
                track_running_stats=bn.track_running_stats,
                process_group=bn.process_group)

    def forward(self, x):
        # calculate the output with running mean/running var
        self.train(mode=False)
        y = super().forward(x)

        # update the parameters and ignore the output
        self.train(mode=True)
        super().forward(x)
        return y

class PreciseReshapeBatchNorm2d(ReshapeBatchNorm2d):
    def __init__(self, bn):
        super().__init__(num_features=bn.num_features,
                eps=bn.eps,
                momentum=bn.momentum,
                affine=bn.affine,
                track_running_stats=bn.track_running_stats)

    def forward(self, x):
        # calculate the output with running mean/running var
        self.train(mode=False)
        y = super().forward(x)

        # update the parameters and ignore the output
        self.train(mode=True)
        super().forward(x)
        return y

class PreciseBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, bn):
        super(PreciseBatchNorm2d, self).__init__(
                num_features=bn.num_features,
                eps=bn.eps,
                momentum=bn.momentum,
                affine=bn.affine,
                track_running_stats=bn.track_running_stats,
                )

    def forward(self, x):
        # calculate the output with running mean/running var
        self.train(mode=False)
        y = super().forward(x)

        # update the parameters and ignore the output
        self.train(mode=True)
        super().forward(x)
        return y


#class PreciseBatchNorm(torch.nn.Module):
    #def __init__(self, bn):
        #super().__init__()
        #self.bn = bn
        #torch.nn.Sequential
        #if self.bn.affine:
            #self.weight = self.bn.weight
            #self.bias = self.bn.bias
        #else:
            #self.register_parameter('weight', None)
            #self.register_parameter('bias', None)

        #if self.bn.track_running_stats:
            #self.register_buffer('running_mean', self.bn.running_mean)
            #self.register_buffer('running_var', self.bn.running_var)
            #self.register_buffer('num_batches_tracked',
                    #self.bn.num_batches_tracked)
        #else:
            #self.register_parameter('running_mean', None)
            #self.register_parameter('running_var', None)
            #self.register_parameter('num_batches_tracked', None)

    #def forward(self, x):
        ## calculate the output with running mean/running var
        #self.bn.train(mode=False)
        #y = self.bn.forward(x)

        ## update the parameters and ignore the output
        #self.bn.train(mode=True)
        #self.bn.forward(x)
        #return y

