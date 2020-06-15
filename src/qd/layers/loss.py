import torch
from torch import nn
import logging
from collections import OrderedDict


class FocalLossWithLogitsNegLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target):
        sigmoid_pred = pred.sigmoid()
        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        loss = (target == 1) * self.alpha * torch.pow(1. - sigmoid_pred, self.gamma) * log_sigmoid

        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)
        loss += (target == 0) * (1 - self.alpha) * torch.pow(sigmoid_pred, self.gamma) * log_sigmoid_inv

        return -loss

class FocalSmoothBCEWithLogitsNegLoss(nn.Module):
    def __init__(self, alpha, gamma, pos, neg):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos = pos
        self.neg = neg

    def forward(self, logits, target):
        target_prob = target.clone().float()
        target_prob[target == 1] = self.pos
        target_prob[target == 0] = self.neg

        sigmoid_pred = logits.sigmoid()

        log_sigmoid = torch.nn.functional.logsigmoid(logits)
        log_sigmoid_inv = torch.nn.functional.logsigmoid(-logits)

        coef = (target == 1) * self.alpha * torch.pow((self.pos - sigmoid_pred).abs(), self.gamma)
        loss = coef * (self.pos * log_sigmoid + (1 - self.pos) * log_sigmoid_inv)

        coef = (target == 0) * (1 - self.alpha) * torch.pow((sigmoid_pred - self.neg).abs(), self.gamma)
        loss += coef * (self.neg * log_sigmoid + (1 - self.neg) * log_sigmoid_inv)

        return -loss

class SmoothBCEWithLogitsNegLoss(nn.Module):
    def __init__(self, pos, neg, reduction=None):
        super().__init__()
        self.pos = pos
        self.neg = neg
        self.reduction = reduction

    def forward(self, logits, target):
        target_prob = target.clone().float()
        target_prob[target == 1] = self.pos
        target_prob[target == 0] = self.neg
        valid = ((target == 1) | (target == 0))
        log_sig = nn.functional.logsigmoid(logits)
        log_sig_neg = nn.functional.logsigmoid(-logits)
        loss = valid * target_prob * log_sig + valid * (1 - target_prob) * log_sig_neg
        if self.reduction is None:
            return -loss
        else:
            raise NotImplementedError

class BCEWithLogitsNegLoss(nn.Module):
    def __init__(self, reduction=None):
        super(BCEWithLogitsNegLoss, self).__init__()
        self.reduction = reduction

    def forward(self, feature, target):
        return bce_with_logits_neg_loss(feature, target,
                                        self.reduction)

def bce_with_logits_neg_loss(feature, target, reduction=None):
    target = target.float()
    weight = torch.ones_like(target)
    weight[target == -1] = 0
    weight_sum = torch.sum(weight)
    if weight_sum == 0:
        return 0
    else:
        criterion = nn.BCEWithLogitsLoss(weight, reduction='sum')
        loss = criterion(feature, target)
        if reduction == 'sum':
            return loss
        else:
            return loss / weight_sum

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

    def forward(self, image, origin_target=None):
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
            if self.weights is None:
                self.weights = [1. / num] * num
            w = self.weights[i]
            loss['loss_{}'.format(i)] = l * w
        return loss

class SmoothLabelCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        self.eps = eps
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.iter = 0

    def forward(self, feature, target):
        debug_print = (self.iter % 100) == 0
        self.iter += 1
        eps = self.eps
        n_class = feature.size(1)
        one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(feature)
        if debug_print:
            prob = torch.nn.functional.softmax(feature.detach(), dim=1)
            num = feature.size(0)
            avg_prob = prob[torch.arange(num), target].mean()
            logging.info('avg positive = {}'.format(avg_prob))
        loss = self.kl(log_prb, one_hot)
        return loss

