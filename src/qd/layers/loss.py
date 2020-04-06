from torch import nn

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
        output, label = self.module(image[0], image[1])
        loss = self.criterion(output, label)
        return {'criterion_loss': loss}

