raise ValueError('use mitorch_models.factory to create mobilenetv3')
import collections
import torch


class ModuleBase(torch.nn.Module):
    def apply_settings(self, args):
        pass

    def reset_parameters(self):
        pass

class Add(ModuleBase):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, *inputs):
        x = inputs[0]
        for i in inputs[1:]:
            x = x + i
        return x

class Conv2dAct(ModuleBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bn=True, activation=None):
        super(Conv2dAct, self).__init__()
        self.explicit_settings = {'use_bn': use_bn, 'activation': activation}

        use_bn = use_bn if use_bn is not None else True
        activation = activation if activation is not None else 'relu'

        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=(not use_bn))
        self.bn = torch.nn.BatchNorm2d(out_channels) if use_bn else None
        self._set_activation(activation)

    def forward(self, input):
        x = self.conv(input)
        if self.bn:
            x = self.bn(x)
        return self.activation(x)

    def _set_activation(self, act):
        assert act in ['relu', 'hswish', 'swish', 'relu6']

        if act == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif act == 'hswish':
            self.activation = HardSwish()
        elif act == 'swish':
            self.activation = Swish()
        elif act == 'relu6':
            self.activation = torch.nn.ReLU6(inplace=True)

class Conv2dBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dBN, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):
        return self.bn(self.conv(input))

class MBConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expansion_channels, kernel_size=3, stride=1, use_se=True, use_se_swish=False, use_se_hsigmoid=True, activation='hswish'):
        super(MBConv, self).__init__()
        self.conv0 = Conv2dAct(in_channels, expansion_channels, kernel_size=1, activation=activation) if in_channels != expansion_channels else None
        self.conv1 = Conv2dAct(expansion_channels, expansion_channels, kernel_size=kernel_size, padding=kernel_size//2,
                               stride=stride, groups=expansion_channels, activation=activation)
        self.conv2 = Conv2dBN(expansion_channels, out_channels, kernel_size=1)

        self.se = SEBlock(expansion_channels, reduction_ratio=4, use_hsigmoid=use_se_hsigmoid, use_swish=use_se_swish) if use_se else None
        self.residual = Add() if stride == 1 and in_channels == out_channels else None

    def forward(self, input):
        x = self.conv0(input) if self.conv0 else input
        x = self.conv1(x)

        if self.se:
            x = self.se(x)

        x = self.conv2(x)

        if self.residual:
            x = self.residual(x, input)

        return x

class Swish(ModuleBase):
    def forward(self, input):
        return input * torch.sigmoid(input)

class HardSigmoid(ModuleBase):
    def forward(self, input):
        return torch.nn.functional.relu6(input + 3) / 6


class HardSwish(ModuleBase):
    def forward(self, input):
        return input * torch.nn.functional.relu6(input + 3) / 6

class SEBlock(ModuleBase):
    def __init__(self, in_channels, reduction_ratio, use_hsigmoid=False, use_swish=False):
        super(SEBlock, self).__init__()

        self.in_channels = in_channels
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc0 = torch.nn.Linear(in_channels, in_channels // reduction_ratio)
        self.activation0 = Swish() if use_swish else torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = HardSigmoid() if use_hsigmoid else torch.nn.Sigmoid()

    def forward(self, input):
        x = self.pool(input)
        x = x.view(-1, self.in_channels)
        x = self.activation0(self.fc0(x))
        x = self.sigmoid(self.fc1(x))
        x = x.view(-1, self.in_channels, 1, 1)
        return input * x

class Model(torch.nn.Module):
    MAJOR_VERSION = 0 # Major updates where the results can be changed.
    MINOR_VERSION = 0 # Minor updates that doesn't have impact on model outputs. e.g. module name changes.

    def __init__(self, output_dim, **kwargs):
        super(Model, self).__init__()
        self.output_dim = output_dim
        self.modules_kwargs = kwargs

    def __setattr__(self, name, value):
        if isinstance(value, torch.nn.Module):
            for m in value.modules():
                apply_settings = getattr(m, 'apply_settings', None)
                if apply_settings and callable(apply_settings):
                    m.apply_settings(self.modules_kwargs)

        super(Model, self).__setattr__(name, value)

    def get_output_shapes(self, output_names):
        input = torch.randn(1, 3, 224, 224)
        outputs = self.forward(input, output_names)
        return [o.shape[1] for o in outputs]

    def forward(self, input, output_names = None):
        if not output_names:
            if hasattr(self, 'features'):
                return self.features(input)
            else:
                raise NotImplementedError
        else:
            # TODO: Is there more efficient way to extract values?
            forward_hooks = []
            for i, name in enumerate(output_names):
                m = self._find_module_by_name(name)
                forward_hooks.append(m.register_forward_hook(functools.partial(self._extract_outputs_hook, index=i)))
            self._outputs = [None] * len(output_names)

            self.forward(input)

            for hook in forward_hooks:
                hook.remove()
            assert all([o is not None for o in self._outputs])
            return self._outputs

    def _find_module_by_name(self, name):
        paths = name.split('.')
        current = self
        for p in paths:
            children = current.named_children()
            for name, c in children:
                if name == p:
                    current = c
                    break

        return current

    def _extract_outputs_hook(self, module, input, output, index):
        self._outputs[index] = output

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, ModuleBase):
                m.reset_parameters()

class MobileNetV3(Model):
    def __init__(self, width_multiplier = 1, activation = 'relu6', dropout_ratio=0.2):
        m = width_multiplier
        super(MobileNetV3, self).__init__(int(1280 * m), activation=activation)

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv0', Conv2dAct(3, int(16 * m), kernel_size=3, padding=1, stride=2, activation='hswish')),
            ('block0_0', MBConv(int(16 * m), int(16 * m), int(16 * m), use_se=False, activation='relu6')),
            ('block1_0', MBConv(int(16 * m), int(24 * m), int(64 * m), use_se=False, activation='relu6', stride=2)),
            ('block1_1', MBConv(int(24 * m), int(24 * m), int(72 * m), use_se=False, activation='relu6')),
            ('block2_0', MBConv(int(24 * m), int(40 * m), int(72 * m), activation='relu6', stride=2, kernel_size=5)),
            ('block2_1', MBConv(int(40 * m), int(40 * m), int(120 * m), activation='relu6', kernel_size=5)),
            ('block2_2', MBConv(int(40 * m), int(40 * m), int(120 * m), activation='relu6', kernel_size=5)),
            ('block3_0', MBConv(int(40 * m), int(80 * m), int(240 * m), use_se=False, stride=2)),
            ('block3_1', MBConv(int(80 * m), int(80 * m), int(200 * m), use_se=False)),
            ('block3_2', MBConv(int(80 * m), int(80 * m), int(184 * m), use_se=False)),
            ('block3_3', MBConv(int(80 * m), int(80 * m), int(184 * m), use_se=False)),
            ('block3_4', MBConv(int(80 * m), int(112 * m), int(480 * m))),
            ('block3_5', MBConv(int(112 * m), int(112 * m), int(672 * m))),
            ('block4_0', MBConv(int(112 * m), int(160 * m), int(672 * m), stride=2, kernel_size=5)),
            ('block4_1', MBConv(int(160 * m), int(160 * m), int(960 * m), kernel_size=5)),
            ('block4_2', MBConv(int(160 * m), int(160 * m), int(960 * m), kernel_size=5)),
            ('conv1', Conv2dAct(int(160 * m), int(960 * m), kernel_size=1, activation='hswish')),
            ('pool0', torch.nn.AdaptiveAvgPool2d(1)),
            ('conv2', Conv2dAct(int(960 * m), int(1280 * m), kernel_size=1, use_bn=False, activation='hswish')),
            ('dropout', torch.nn.Dropout(p=dropout_ratio)),
            ('flatten', torch.nn.Flatten())
        ]))

