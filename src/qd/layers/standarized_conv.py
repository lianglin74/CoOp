from torch import nn


class StandarizedConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return nn.functional.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def convert_conv2d_to_standarized_conv2d(model):
    from qd.torch_common import replace_module
    model = replace_module(model, lambda m: type(m) == nn.Conv2d,
                   lambda m: StandarizedConv2d(m.in_channels,
                                               m.out_channels,
                                               m.kernel_size,
                                               m.stride,
                                               m.padding,
                                               m.dilation,
                                               m.groups,
                                               m.bias is not None,
                                               ))
    return model
