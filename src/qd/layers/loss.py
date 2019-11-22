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

