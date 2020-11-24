import logging
import torch
from qd.qd_common import get_mpi_size
import math
from torch.autograd.function import Function
import torch.distributed as dist


class LayerBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features,
                             eps, momentum, affine,
                             track_running_stats)

    def forward(self, x):
        origin_shape = x.shape
        need_reshape = len(origin_shape) > 2
        if need_reshape:
            x = x.view(-1, self.bn.num_features)
        x = self.bn(x)
        if need_reshape:
            x = x.view(origin_shape)
        return x

class ConvergingBatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, policy, max_iter, gamma=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_size = get_mpi_size()

        self.register_buffer('iter', torch.zeros(1))

        self.policy = policy
        self.max_iter = max_iter
        self.gamma = gamma

    def _check_input_dim(self, input):
        if input.dim() <= 1:
            raise ValueError('expected at least 2D input (got {}D input)'
                             .format(input.dim()))
    def forward(self, input):
        if not self.training:
            return super().forward(input)

        alpha = self.get_alpha()

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        mean_dim = list(range(input.dim()))
        del mean_dim[1]
        mean = torch.mean(input, dim=mean_dim)
        meansqr = torch.mean(input * input, dim=mean_dim)

        var = meansqr - mean * mean

        forward_mean = alpha * mean + (1. - alpha) * self.running_mean
        forward_var = alpha * var + (1. - alpha) * self.running_var

        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(forward_var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - forward_mean * scale

        param_shape = [1] * input.dim()
        param_shape[1] = -1
        scale = scale.reshape(param_shape)
        bias = bias.reshape(param_shape)
        if (self.iter % 100) == 0 and self.training:
            def get_info(batch, running):
                x1 = batch.detach().abs().sum()
                x2 = running.detach().abs().sum()
                return '{}/{}/{}'.format(round(float(x1.cpu()), 2),
                                         round(float(x2.cpu()), 2),
                                         round(float(2. * (x1 - x2)/(x2 + x1 +self.eps)), 2))
            info = []
            if 'name_from_root' in self.__dict__:
                info.append(self.name_from_root)
            info.append('alpha={}'.format(alpha))
            info.append('mean: batch/running/diff={}'.format(
                get_info(forward_mean, self.running_mean)))
            info.append('var: batch/running/diff={}'.format(
                get_info(forward_var, self.running_var)))
            logging.info('; '.join(info))
        self.iter += 1
        return input * scale + bias

    def get_alpha(self):
        if self.policy == 'exp':
            # alpha = exp(-gamma * i/max_iter)
            return torch.exp(-self.gamma * self.iter / self.max_iter)
        if self.policy == 'cosine':
            return torch.cos(math.pi / 2. * self.iter / self.max_iter)
        if self.policy == 'const':
            return self.gamma
        raise NotImplementedError

class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(get_mpi_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output

class NaiveSyncBatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
        self.world_size = get_mpi_size()

    def _check_input_dim(self, input):
        if input.dim() <= 1:
            raise ValueError('expected at least 2D input (got {}D input)'
                             .format(input.dim()))
    def forward(self, input):
        if self.world_size == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean_dim = list(range(input.dim()))
        del mean_dim[1]
        mean = torch.mean(input, dim=mean_dim)
        meansqr = torch.mean(input * input, dim=mean_dim)

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale

        param_shape = [1] * input.dim()
        param_shape[1] = -1
        scale = scale.reshape(param_shape)
        bias = bias.reshape(param_shape)
        if (self.iter % 100) == 0 and self.training:
            def get_info(batch, running):
                x1 = batch.detach().abs().sum()
                x2 = running.abs().sum()
                return '{}/{}/{}'.format(round(float(x1.cpu()), 2),
                        round(float(x2.cpu()), 2),
                        round(float(2. * (x1 - x2)/(x2 + x1 +self.eps)), 2))
            info = 'mean: {}; var: {}'.format(get_info(mean, self.running_mean),
                    get_info(var, self.running_var))
            logging.info(info)
        self.iter += 1
        return input * scale + bias

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, num_features, eps=0):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def extra_repr(self):
        return '{}, eps={}'.format(len(self.weight), self.eps)
