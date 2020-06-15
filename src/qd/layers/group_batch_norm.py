import torch
import logging
from torch import nn


def get_normalize_groups(num_features, num_groups, group_size):
    if num_groups:
        assert group_size is None or group_size * num_groups == num_features
        return num_groups
    assert (num_features % group_size) == 0
    return num_features // group_size


class GroupBatchNorm(nn.Module):
    def __init__(self, num_groups, num_features):
        super().__init__()
        self.num_groups = num_groups
        self.bn = self.create_normalization()
        from torch.nn.parameter import Parameter
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        logging.info(num_groups)

        self.reset_parameters()

    def create_normalization(self):
        return nn.modules.batchnorm._BatchNorm(self.num_groups, affine=False)

    def reset_parameters(self):
        from torch.nn import init
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        original_shape = x.shape
        N, C = x.shape[:2]
        x = x.view(N, self.num_groups, -1)

        x = self.bn(x)
        x = x.view(N, C, -1)
        x *= self.weight.view(1, C, -1)
        x += self.bias.view(1, C, -1)
        x = x.view(original_shape)
        return x

class SyncGroupBatchNorm(GroupBatchNorm):
    def __init__(self, num_groups, num_features):
        super().__init__(num_groups, num_features)

    def create_normalization(self):
        return nn.SyncBatchNorm(self.num_groups, affine=False)

