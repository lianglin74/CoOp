from torch import nn


class VisAdaptiveAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x2s = nn.functional.normalize(x, dim=1)
        x2s = x2s.sum(dim=1)
        for x2 in x2s:
            min_value = x2.min()
            max_value = x2.max()

            x2 = 255. * (x2 - min_value) / (max_value - min_value)
            x2 = x2.byte()
            x2 = x2.cpu().numpy()
            from qd.process_image import show_image
            show_image(x2)

        y = self.avg_pool(x)
        return y

