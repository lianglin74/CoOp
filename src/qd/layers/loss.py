import torch
from torch import nn
import logging
from collections import OrderedDict


class ExclusiveMultiHotCrossEntropyLoss(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, logits, target):
        logits_exp = logits.exp()
        numerator = (target * logits_exp).sum(dim=1)
        denormenator = ((1. - target) * logits_exp).sum(dim=1)
        loss = -numerator.log() + self.factor * denormenator.log()
        return loss.mean()

class ExclusiveCrossEntropyLoss(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        #self.iter = 0

    def forward(self, logits, target):
        #verbose = (self.iter % 10) == 0
        #self.iter += 1
        logits_exp = logits.exp()
        pos = logits[torch.arange(len(logits_exp)), target]
        #if verbose:
            #import logging
            #from qd.torch_common import describe_tensor
            #logging.info('pos: {}'.format(describe_tensor(pos)))
        exp_pos = logits_exp[torch.arange(len(logits_exp)), target]
        exp_neg = logits_exp.sum(dim=1) - exp_pos
        loss = -pos + self.factor * exp_neg.log()
        return loss.mean()

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

class FocalLossWithLogitsNegSoftLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target):
        # the loss can be generalized as -weight * (\sigma(x) - target)^r *
        # [target * log(sigma(x)) + (1 - target) * log(1 - sigma(x))]
        # target == 0: means neg: target > 0 means positive; target < 0 means
        # ignorable

        weight = torch.zeros_like(target)
        weight[target == 0] = 1. - self.alpha  # negative
        weight[target > 1e-5] = self.alpha # positive

        sigmoid_pred = pred.sigmoid()

        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)

        coef = weight * torch.pow((sigmoid_pred - target).abs(), self.gamma)
        loss = target * log_sigmoid + (1. - target) * log_sigmoid_inv
        loss = (coef * loss).sum()

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

    def forward(self, *args):
        if len(args) == 2:
            data, target = args
        else:
            data, target = args[0]['image'], args[0]['label']
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

class DistilCrossEntropyLoss(nn.Module):
    def __init__(self, gt_weight):
        super().__init__()
        self.gt_weight = gt_weight
    def forward(self, feature, target):
        with torch.no_grad():
            softmax_feature = feature.softmax(dim=1)
            gt_matrix = torch.zeros_like(feature)
            gt_matrix.scatter_(dim=1, index=target.view(-1, 1), src=torch.tensor(1))
            soft_target = (1. - self.gt_weight) * softmax_feature + self.gt_weight * gt_matrix
        log_soft = nn.functional.log_softmax(feature, dim=1)
        loss = nn.functional.kl_div(log_soft, soft_target, reduction='batchmean')
        return loss

class MultiHotCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiHotCrossEntropyLoss, self).__init__()

    def forward(self, feature, target):
        return multi_hot_cross_entropy(feature, target)

def multi_hot_cross_entropy(pred, soft_targets):
    assert ((soft_targets != 0) & (soft_targets != 1)).sum() == 0
    logsoftmax = nn.LogSoftmax(dim=1)
    target_sum = torch.sum(soft_targets)
    if target_sum == 0:
        return 0
    else:
        return torch.sum(-soft_targets * logsoftmax(pred)) / target_sum

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

