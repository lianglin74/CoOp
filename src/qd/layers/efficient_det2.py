# code is based on https://github.com/toandaominh1997/EfficientDet.Pytorch
import torch
from torch import nn
from torch.nn import functional as F
import re
import math
import collections

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size',
    'non_local_net',
])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio',
    'non_local_net',
])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(
        filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params

def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError(
            'model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * \
            self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = SimpleConv(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = SimpleConv(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = SimpleConv(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = SimpleConv(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = SimpleConv(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate,
                                 training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self,
                 blocks_args=None,
                 global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = SimpleConv(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        non_local_net = global_params.non_local_net
        if non_local_net is not None:
            stride2_stages = [i for i, b in enumerate(self._blocks_args) if b.stride[0] == 2]
            idx_non_local = stride2_stages[non_local_net - 1]
            assert self._blocks_args[idx_non_local].num_repeat >= 2
        else:
            idx_non_local = None
        for idx_block, block_args in enumerate(self._blocks_args):

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for idx_repeat in range(block_args.num_repeat - 1):
                if idx_block == idx_non_local and \
                        idx_repeat == block_args.num_repeat - 2:
                    # as the paper of non-local-network suggests, we add it
                    # before  the last building block
                    block_args = block_args._replace(non_local_net=True)
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = SimpleConv(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        # before the name is _fc, rather than fc. This is a breaking change which
        # might impact the classifier testing accuracy. We do this because all
        # resnet-family network's name is fc rather than _fc. This consistency
        # will make the unsupervised pre-training much easier.
        self.fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self.fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

class SimpleMaxPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride,
                                 padding=(kernel_size-1)//2)

    def forward(self, x):
        return self.pool(x)

class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = SimpleConv(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            bias=False)
        self.pointwise_conv = SimpleConv(
            in_channels,
            out_channels,
            kernel_size=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1
        assert dilation == 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias,
                              groups=groups,
                              padding=(kernel_size - 1) // 2)
        self.stride = self.conv.stride
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        else:
            self.stride = list(self.stride)

    def forward(self, x):
        return self.conv(x)

class EffNetFPN(nn.Module):
    backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
    fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
    conv_channel_coef = {
        0: [40, 112, 320],
        1: [40, 112, 320],
        2: [48, 120, 352],
        3: [48, 136, 384],
        4: [56, 160, 448],
        5: [64, 176, 512],
        6: [72, 200, 576],
        7: [72, 200, 576],
    }
    conv_channel_coef2345 = {
        # the channels of P2/P3/P4/P5.
        0: [24, 40, 112, 320],
        1: [24, 40, 112, 320],
        2: [24, 48, 120, 352],
        3: [32, 48, 136, 384],
        4: [56, 160],
        5: [64, 176],
        6: [72, 200],
        7: [72, 200],
    }
    fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
    def __init__(self, compound_coef=0, start_from=3):
        super().__init__()

        self.backbone_net = EfficientNetD(self.backbone_compound_coef[compound_coef],
                                          load_weights=False)
        if start_from == 3:
            conv_channel_coef = self.conv_channel_coef[compound_coef]
        else:
            conv_channel_coef = self.conv_channel_coef2345[compound_coef]

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[compound_coef],
                    conv_channel_coef,
                    True if _ == 0 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.out_channels = self.fpn_num_filters[compound_coef]
        self.start_from = start_from
        assert self.start_from in [2, 3]

    def forward(self, inputs):
        if self.start_from == 3:
            _, p3, p4, p5 = self.backbone_net(inputs)

            features = (p3, p4, p5)
            features = self.bifpn(features)
            return features
        else:
            p2, p3, p4, p5 = self.backbone_net(inputs)
            features = (p2, p3, p4, p5)
            features = self.bifpn(features)
            return features

class BiFPN(nn.Module):
    def __init__(self, num_channels,
                 conv_channels,
                 first_time=False,
                 epsilon=1e-4,
                 onnx_export=False):
        super().__init__()
        self.epsilon = epsilon
        self.first_fusion_output_convs = nn.ModuleList([DepthwiseSeparableConvModule(num_channels, onnx_export=onnx_export)
                       for _ in range(4)])

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.down_channels = nn.ModuleList([nn.Sequential(
                SimpleConv(c, num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            ) for c in conv_channels])

            if len(conv_channels) == 3:
                self.down_channel_reduce_spatial = nn.Sequential(
                    SimpleConv(conv_channels[2], num_channels, 1),
                    nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                    SimpleMaxPool(3, 2)
                )
            self.reduce_spatial = SimpleMaxPool(3, 2)

            self.extra_down_channel = nn.ModuleList([nn.Sequential(
                SimpleConv(c, num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            ) for c in conv_channels[1:3]])

        self.first_fusion_weights = nn.ParameterList([nn.Parameter(torch.ones(2,
                                                        dtype=torch.float32),
                                             requires_grad=True) for _ in
                                range(4)])
        self.second_fusion_weights = nn.ParameterList([nn.Parameter(torch.ones(3 if i < 3 else 2,
                                                dtype=torch.float32),
                                     requires_grad=True) for i in range(4)])

        self.second_fusion_output_convs = nn.ModuleList([
            DepthwiseSeparableConvModule(num_channels, onnx_export=onnx_export)
            for _ in range(4)])


    def forward(self, inputs):
        #ipdb> pp [i.shape for i in inputs]
        #[torch.Size([1, 24, 96, 128]),
         #torch.Size([1, 40, 48, 64]),
         #torch.Size([1, 112, 24, 32]),
         #torch.Size([1, 320, 12, 16])]
        if len(inputs) == 3:
            five_inputs = [c(i) for i, c in zip(inputs, self.down_channels)]
            five_inputs.append(self.down_channel_reduce_spatial(inputs[-1]))
            five_inputs.append(self.reduce_spatial(five_inputs[-1]))
        elif len(inputs) == 4:
            five_inputs = [c(i) for i, c in zip(inputs, self.down_channels)]
            five_inputs.append(self.reduce_spatial(five_inputs[-1]))
        else:
            five_inputs = inputs

        #ipdb> pp [i.shape for i in five_inputs]
        #[torch.Size([1, 64, 96, 128]),
         #torch.Size([1, 64, 48, 64]),
         #torch.Size([1, 64, 24, 32]),
         #torch.Size([1, 64, 12, 16]),
         #torch.Size([1, 64, 6, 8])]

        first_fusion_tb_out = [None for i in range(4)] # from top to bottom, tb: top to bottom
        # feature map (idx in five_inputs) (idx in tb_out)
        # 7           (4)
        # 6           (3)                  (3)
        # 5           (2)                  (2)
        # 4           (1)                  (1)
        # 3           (0)                  (0)
        for i in range(3, -1, -1):
            weight = nn.functional.relu(self.first_fusion_weights[i])
            norm_weight = weight / (weight.sum() + self.epsilon)
            if i == 3:
                previous = five_inputs[-1]
            else:
                previous = first_fusion_tb_out[i + 1]
            curr = five_inputs[i]
            upsample = nn.Upsample(size=curr.shape[-2:])
            x = norm_weight[0] * curr + norm_weight[1] * upsample(previous)
            x = self.swish(x)
            x = self.first_fusion_output_convs[i](x)
            first_fusion_tb_out[i] = x

        if self.first_time:
            for i in range(1, 3):
                five_inputs[i] = self.extra_down_channel[i - 1](inputs[i])

        second_fusion_bt_out = [None for _ in range(5)] # bt: bottom to top
        second_fusion_bt_out[0] = first_fusion_tb_out[0]

        for i in range(1, 5):
            previous = second_fusion_bt_out[i - 1]
            weight = nn.functional.relu(self.second_fusion_weights[i - 1])
            norm_weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
            curr_in = five_inputs[i]
            downsample = SimpleMaxPool(3, 2)
            if i < 4:
                first_fusion = first_fusion_tb_out[i]
                x = norm_weight[0] * curr_in + norm_weight[1] * first_fusion + norm_weight[2] * downsample(previous)
            else:
                x = norm_weight[0] * curr_in + norm_weight[1] * downsample(previous)
            x = self.swish(x)
            out = self.second_fusion_output_convs[i - 1](x)
            second_fusion_bt_out[i] = out

        return second_fusion_bt_out


class EfficientNetD(nn.Module):
    def __init__(self, compound_coef, load_weights=False,
                 drop_connect_rate=None,
                 non_local_net=None,
                 ):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        override_params = {'num_classes': 1000}
        override_params['non_local_net'] = non_local_net
        model = EfficientNet.from_name(
            f'efficientnet-b{compound_coef}',
            override_params=override_params)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model.fc
        self.model = model
        if drop_connect_rate is None:
            drop_connect_rate = self.model._global_params.drop_connect_rate
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x

        return feature_maps[1:]

