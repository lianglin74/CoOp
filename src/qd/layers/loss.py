from qd.torch_common import describe_tensor
import torch
from torch import nn
import logging
from collections import OrderedDict
from qd.torch_common import accuracy
from qd.torch_common import SparseTensor
from qd.torch_common import IgnoreLastDimSparseTensor


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

class DistillFocalLossWithLogitsNegLoss(nn.Module):
    def __init__(self, alpha, gamma, t=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.T = t

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target, guide):
        weight = torch.zeros_like(target)
        weight[target == 0] = 1. - self.alpha  # negative
        weight[target > 1e-5] = self.alpha # positive

        sigmoid_pred = pred.sigmoid()
        sigmoid_guide = (guide / self.T).sigmoid()

        log_sigmoid = torch.nn.functional.logsigmoid(pred)
        log_sigmoid_inv = torch.nn.functional.logsigmoid(-pred)

        coef = weight * torch.pow((sigmoid_pred - target).abs(), self.gamma)
        loss = sigmoid_guide * log_sigmoid + (1. - sigmoid_guide) * log_sigmoid_inv
        loss = (coef * loss).sum()

        return -loss

class FocalLossWithLogitsNegSoftLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def extra_repr(self):
        return 'alpha={}, gamma={}'.format(self.alpha, self.gamma)

    def forward(self, pred, target):

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
        super().__init__()
        self.reduction = reduction

    def forward(self, feature, target):
        return bce_with_logits_neg_loss(feature, target,
                                        self.reduction)

class MultiDimCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, x, y):
        x = x.reshape((-1, x.shape[-1]))
        y = y.reshape(-1)
        return super().forward(x, y)

def bce_with_logits_neg_loss(feature, target, reduction=None):
    target = target.float()
    weight = torch.ones_like(target)
    weight[target == -1] = 0
    weight_sum = torch.sum(weight)
    if weight_sum == 0:
        return torch.tensor(0, device=feature.device, dtype=feature.dtype, requires_grad=True)
    else:
        criterion = nn.BCEWithLogitsLoss(weight, reduction='sum')
        loss = criterion(feature, target)
        if reduction == 'sum':
            return loss
        elif reduction == 'pos':
            return criterion / ((target > 0.99).sum() + 1e-5)
            raise NotImplementedError
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

class ModelLossWithInput(nn.Module):
    # used for mask-rcnn trainer engine
    def __init__(self, model, criterion):
        super().__init__()
        self.module = model
        self.criterion = criterion

    def forward(self, *args):
        data_dict = args[0]
        out = self.module(data_dict['image'])
        loss = self.criterion(out, data_dict['label'],
                              data_dict)
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
        self.iter = 0

    def extra_repr(self):
        return 'weights={}'.format(self.weights)

    def forward(self, *args):
        verbose = (self.iter % 100) == 0
        self.iter += 1

        info = []
        num = len(args) // 2
        loss = OrderedDict()
        for i in range(num):
            logits = args[2 * i]
            label = args[2 * i + 1]
            if verbose:
                with torch.no_grad():
                    # in some cases, there is less than 5 outputs. so don't run
                    # top5. top1 is also good enough
                    top1, = accuracy(logits, label, (1,))
                    pos = logits.gather(1, label[:, None])
                    if logits.numel() == pos.numel():
                        neg = 0
                    else:
                        neg = (logits.sum() - pos.sum()) / (
                            logits.numel() - pos.numel())
                    info.append('top1 = {:.1f}, pos_avg = {:.1f}, '
                                'neg_avg = {:.1f}'.format(
                        float(top1),
                        float(pos.mean()),
                        float(neg)
                    ))

            l = self.loss(logits, label)
            if self.weights is None:
                self.weights = [1.] * num
            w = self.weights[i]
            loss['loss_{}'.format(i)] = l * w
        if verbose:
            logging.info('\n'.join(info))
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

class BCELogitsNormByEachPositive(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter = 0

    def forward_fast(self, feature, target, verbose):
        assert isinstance(target, IgnoreLastDimSparseTensor)
        # ignore the element if the corresponding target value is -1
        feature_shape = feature.shape
        batch_size = feature.shape[0]
        count = target.keep_first_sum()

        count = count.clamp(min=1)
        weight = torch.ones(feature_shape[:-1],
                            device=feature.device)
        if weight.dim() == 2 and count.dim() == 1:
            count = count[:, None]
        weight /= count

        target2d = target.reshape((-1, feature_shape[-1]))
        valid = torch.ones((target2d.shape[0],), dtype=torch.long, device=feature.device)
        valid[target2d.ignore_index] = 0
        valid = valid.bool()
        feature = feature.view((-1, feature_shape[-1]))
        feature = feature[valid]
        weight = weight.view(-1)
        weight = weight[valid]
        target = target2d.remove_ignore_2d_to_dense()
        weight = weight[:,None].expand_as(feature)

        loss = nn.functional.binary_cross_entropy_with_logits(
            feature, target, weight=weight, reduction='sum')
        if verbose:
            with torch.no_grad():
                pos_value = feature[target > 0.9].mean()
                neg_value = feature[target == 0].mean()
                logging.info('pos = {}; neg = {}'.format(pos_value, neg_value))
        return loss / batch_size

    def forward(self, feature, target):
        verbose = (self.iter % 100) == 0
        self.iter += 1
        if isinstance(target, IgnoreLastDimSparseTensor):
            return self.forward_fast(feature, target, verbose)
        elif isinstance(target, (torch.sparse.FloatTensor, SparseTensor)):
            target = target.to_dense()
        return self.forward_dense(feature, target, verbose)

        #if isinstance(target, (IgnoreLastDimSparseTensor, torch.sparse.FloatTensor, SparseTensor)):
            #target = target.to_dense()
        #return self.forward_dense(feature, target)

    def forward_dense(self, feature, target, verbose=False):
        # the following is a slow version
        batch_size = feature.shape[0]
        assert target.shape[0] == batch_size
        feature = feature.view((batch_size, -1))
        target = target.view((batch_size, -1))
        target = target.float()
        weight = torch.ones_like(target)
        weight[target < 0] = 0
        if verbose:
            with torch.no_grad():
                pos_value = feature[target > 0.9].mean()
                neg_value = feature[target == 0].mean()
                logging.info('pos = {}; neg = {}'.format(pos_value, neg_value))
        loss = nn.functional.binary_cross_entropy_with_logits(
            feature, target, weight=weight, reduction='none')

        weight = target.clone().detach()
        weight[target < 0] = 0
        weight = weight.view((batch_size, -1))
        loss = (loss.view((batch_size, -1)).sum(dim=1) / weight.sum(dim=1).clamp(min=1))
        return loss.mean()

class BCELogitsNormByPositive(nn.Module):
    def __init__(self, reduction=None):
        super().__init__()
        self.iter = 0
        self.reduction = reduction
        assert self.reduction in [None, 'each']
    def forward(self, feature, target):
        verbose = (self.iter % 100) == 0
        self.iter += 1
        if verbose:
            with torch.no_grad():
                pos_value = feature[target.bool()].mean()
                neg_value = feature[(1 - target).bool()].mean()
                logging.info('pos = {}; neg = {}'.format(pos_value, neg_value))
        if self.reduction is None:
            return nn.functional.binary_cross_entropy_with_logits(
                feature, target.float(), reduction='sum') / target.sum().float().clamp(min=1)
        else:
            each_loss = nn.functional.binary_cross_entropy_with_logits(
                feature, target.float(), reduction='none')
            loss = each_loss.sum(dim=1) / target.sum(dim=1).float().clamp(min=1)
            return loss.mean()

class MultiHotCrossEntropyWithNegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature, target):
        if isinstance(target, (
            torch.sparse.FloatTensor,
            SparseTensor,
            IgnoreLastDimSparseTensor
        )):
            target = target.to_dense()
        soft_dim = feature.shape[-1]
        assert target.shape[-1] == soft_dim
        feature = feature.reshape((-1, soft_dim))
        target = target.reshape((-1, soft_dim))
        pos = target.sum(dim=1) >= 0
        feature = feature[pos]
        target = target[pos]
        loss = -(target * nn.functional.log_softmax(feature, dim=1)).sum(dim=1)
        loss = loss / target.sum(dim=1).clamp(min=1)
        return loss.mean()

class MultiHotCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter = 0

    def forward(self, feature, target):
        verbose = (self.iter % 100) == 0
        self.iter += 1
        if verbose:
            x = nn.functional.softmax(feature, dim=1)[target == 1]
            _, idx = feature.max(dim=1)
            acc = target[torch.arange(len(target)), idx].mean()
            logging.info('prob at gt = {}; acc = {}'.format(
                describe_tensor(x), acc))
        return multi_hot_cross_entropy(feature, target)

def multi_hot_cross_entropy(pred, soft_targets):
    assert ((soft_targets != 0) & (soft_targets != 1)).sum() == 0
    logsoftmax = nn.LogSoftmax(dim=1)
    target_sum = torch.sum(soft_targets)
    if target_sum == 0:
        return torch.tensor(0, dtype=torch.float, device=pred.device,
                            requires_grad=True)
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

class MultiKLCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.loss = KLCrossEntropyLoss()
        self.weights = weights
        self.iter = 0

    def extra_repr(self):
        return 'weights={}'.format(self.weights)

    def forward(self, *args):
        verbose = (self.iter % 100) == 0
        self.iter += 1

        info = []
        num = len(args) // 2
        loss = OrderedDict()
        for i in range(num):
            logits = args[2 * i]
            label = args[2 * i + 1]
            if verbose:
                with torch.no_grad():
                    # in some cases, there is less than 5 outputs. so don't run
                    # top5. top1 is also good enough
                    int_label = label.argmax(dim=1)
                    top1, = accuracy(logits, int_label, (1,))
                    pos = logits.gather(1, int_label[:, None])
                    if logits.numel() == pos.numel():
                        neg = 0
                    else:
                        neg = (logits.sum() - pos.sum()) / (
                            logits.numel() - pos.numel())
                    info.append('top1 = {:.1f}, pos_avg = {:.1f}, '
                                'neg_avg = {:.1f}'.format(
                        float(top1),
                        float(pos.mean()),
                        float(neg)
                    ))

            l = self.loss(logits, label)
            if self.weights is None:
                self.weights = [1.] * num
            w = self.weights[i]
            loss['loss_{}'.format(i)] = l * w
        if verbose:
            logging.info('\n'.join(info))
        return loss

class KLCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter = 0

    def forward(self, logits, conf_targets):
        verbose = (self.iter % 100 == 0)
        self.iter += 1
        # logits.shape == conf_targets.shape
        loss = -(conf_targets * torch.nn.functional.log_softmax(
            logits, dim=1)).sum(dim=1)
        if verbose:
            with torch.no_grad():
                hard_target = conf_targets.argmax(dim=1)
                top1, = accuracy(logits, hard_target)
                logging.info('top1 = {:.1f}'.format(float(top1)))
        ret = loss.mean()
        return ret

class L2Loss(nn.Module):
    def forward(self, x, y):
        d = x - y
        return 0.5 * (d * d).sum() / len(x)

class DistillCrossEntropyLoss(nn.Module):
    def __init__(self, num_image, num_class, momentum,
                 dist_weight=1.):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_ce = KLCrossEntropyLoss()
        history = torch.randn(num_image, num_class)
        history = torch.nn.functional.softmax(history, dim=1)
        self.register_buffer('history', history)
        self.momentum = momentum
        self.dist_weight = dist_weight

    def forward(self, feature, target, data_dict):
        loss1 = self.ce_loss(feature, target)
        idx = data_dict['idx']
        teacher_signal = self.history[idx].detach()
        loss2 = self.kl_ce(feature, teacher_signal)
        with torch.no_grad():
            softmax_feature = torch.nn.functional.softmax(feature, dim=1)
            from qd.torch_common import concat_all_gather
            idx = concat_all_gather(idx)
            softmax_feature = concat_all_gather(softmax_feature)
            teacher_signal = concat_all_gather(teacher_signal)

            self.history[idx] = (self.momentum * teacher_signal + (1. - self.momentum) * softmax_feature)
        return {'ce': loss1, 'kl': loss2 * self.dist_weight}

class EfficientDetCrossEntropy(nn.Module):
    def __init__(self, no_reg, sep):
        super().__init__()
        self.no_reg = no_reg
        self.sep = sep

    def forward(self, features, target):
        regression, classification = features[1:3]
        loss = {}
        if self.sep:
            for i in range(5):
                ind = (features[-1]['stride_idx'] == i)
                cls  = classification[:, ind, :].mean(dim=1)
                loss['cls_{}'.format(i)] = nn.functional.cross_entropy(cls, target)
        else:
            classification = classification.mean(dim=1)
            cls_loss = nn.functional.cross_entropy(classification, target)
            loss['cls_loss'] = cls_loss
        if not self.no_reg:
            regression = regression.mean(dim=1)
            reg_loss = nn.functional.cross_entropy(regression, target)
            loss['reg_loss'] = reg_loss
        return loss

class MultiDomainCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs, target):
        loss = {}
        for i, x in enumerate(xs):
            loss['loss_{}'.format(i)] = torch.nn.functional.cross_entropy(
                x, target[:, i], ignore_index=-1)
        return loss

class SinkhornClusterLoss(torch.nn.Module):
    def __init__(self, cluster_size, dim, eps=0.05, T=0.1):
        super().__init__()
        cluster_center = torch.randn(cluster_size, dim)
        self.cluster_center = torch.nn.Parameter(cluster_center)
        from qd.layers.kl_div_logit_loss import KLDivLogitLoss
        self.criterion = KLDivLogitLoss()
        self.eps = eps
        self.sinkhorn_iter = 3
        self.T = T
        self.iter = 0

    def forward(self, q, k):
        verbose = (self.iter % 100) == 0
        self.iter += 1

        k = nn.functional.normalize(k, dim=1)
        q = nn.functional.normalize(q, dim=1)
        norm_cluster_center = torch.nn.functional.normalize(self.cluster_center, dim=1)

        loss1 = self.forward_once(norm_cluster_center, q, k, verbose)
        loss2 = self.forward_once(norm_cluster_center, k, q, verbose)
        return (loss1 + loss2) / 2.

    def forward_once(self, norm_cluster_center, norm_q, norm_k, verbose):
        sim_mat = torch.matmul(norm_k, norm_cluster_center.T)
        from qd.torch_common import distributed_sinkhorn
        # Q: B * K
        with torch.no_grad():
            curr_code = distributed_sinkhorn(
                sim_mat,
                eps=self.eps,
                niters=self.sinkhorn_iter)

        if verbose:
            with torch.no_grad():
                x = curr_code.sum(dim=0)
                from qd.torch_common import sum_single_reduce_
                sum_single_reduce_(x, 0)
                logging.info('uniform in clusters: {}'.format(
                    describe_tensor(x)))
                logging.info('code = {}'.format(
                    describe_tensor(curr_code)))

        pred = torch.matmul(norm_q, norm_cluster_center.T) / self.T
        return self.criterion(pred, curr_code)


