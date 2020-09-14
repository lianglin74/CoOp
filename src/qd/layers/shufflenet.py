from torch import nn

class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, out_strides):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]

        from torchvision.models.shufflenetv2 import InvertedResidual
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.out_strides = set(out_strides)
        for x in self.out_strides:
            assert x in set([2, 4, 8, 16, 32, 64])

    def get_out_channels(self):
        result = []
        if 2 in self.out_strides:
            result.append(self._stage_out_channels[0])
        if 4 in self.out_strides:
            result.append(self._stage_out_channels[0])
        if 8 in self.out_strides:
            result.append(self._stage_out_channels[1])
        if 16 in self.out_strides:
            result.append(self._stage_out_channels[2])
        if 32 in self.out_strides:
            result.append(self._stage_out_channels[3])
        if 64 in self.out_strides:
            result.append(self._stage_out_channels[4])
        return result

    def forward(self, x):
        result = []
        x = self.conv1(x)
        if 2 in self.out_strides:
            result.append(x)
        x = self.maxpool(x)
        if 4 in self.out_strides:
            result.append(x)
        x = self.stage2(x)
        if 8 in self.out_strides:
            result.append(x)
        x = self.stage3(x)
        if 16 in self.out_strides:
            result.append(x)
        x = self.stage4(x)
        if 32 in self.out_strides:
            result.append(x)
        x = self.conv5(x)
        if 64 in self.out_strides:
            result.append(x)
        return result
