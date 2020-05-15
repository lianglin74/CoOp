from torch import nn
from collections import OrderedDict


class ModelLoss(nn.Module):
    # used for mask-rcnn trainer engine
    def __init__(self, model, criterion):
        super(ModelLoss, self).__init__()
        self.module = model
        self.criterion = criterion

    def forward(self, data, target):
        out = self.module(data)
        loss = self.criterion(out, target)
        if isinstance(loss, dict):
            return loss
        else:
            return {'criterion_loss': loss}

class UnsupervisedLoss(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.module = model
        self.criterion = criterion

    def forward(self, image, origin_target):
        # image could be 2 or 3 or multiple views
        feature_label = self.module(*image)
        loss = self.criterion(*feature_label)
        if isinstance(loss, dict):
            return loss
        else:
            return {'criterion_loss': loss}

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.weights = weights

    def extra_repr(self):
        return 'weights={}'.format(self.weights)

    def forward(self, *args):
        num = len(args) // 2
        loss = OrderedDict()
        for i in range(num):
            l = self.loss(args[2 * i], args[2 * i + 1])
            if self.weights:
                w = self.weights[i]
                loss['loss_{}'.format(i)] = l * w
            else:
                loss['loss_{}'.format(i)] = l
        return loss

