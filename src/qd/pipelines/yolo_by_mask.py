import os.path as op
from qd.tsv_io import TSVDataset
import logging
import torch
from pprint import pformat
from mtorch.caffesgd import CaffeSGD
from mtorch.multifixed_scheduler import MultiFixedScheduler
from mtorch.dataloaders import yolo_train_data_loader
from mtorch.yolo_v2 import yolo_2extraconv
from mtorch.darknet import darknet_layers
from qd.layers.loss import ModelLoss
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import torch.nn as nn
from qd.qd_common import json_dump
import torch.nn as nn
from mtorch.caffetorch import Slice, SoftmaxWithLoss, EuclideanLoss
from mtorch.reshape import Reshape
from mtorch.region_target import RegionTarget
from mtorch.softmaxtree_loss import SoftmaxTreeWithLoss
import math
from qd.qd_common import print_frame_info


def _list_collate(batch):
    """ Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader,
    if you want to have a list of items as an output, as opposed to tensors
    """
    items = list(zip(*batch))
    return items

def freeze_parameters_by_last_name(model, last_fixed_param):
    fixed_modules, names = get_all_module_need_fixed(model, last_fixed_param)
    logging.info('fix the parameters of the following modules: {}'.format(
        pformat(names)))
    freeze_parameters(fixed_modules)

def freeze_parameters_by_name(model, fixed_param):
    fixed_modules, names = get_all_module_with_name(model, fixed_param)
    logging.info('fix the parameters of the following modules: {}'.format(
        pformat(names)))
    freeze_parameters(fixed_modules)

def get_all_module_need_fixed(model, last_fixed_param):
    found = False
    result = []
    names = []
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if not found:
            result.append(m)
            names.append(n)
            if last_fixed_param in n:
                found = True
                break
        else:
            assert last_fixed_param not in n
    assert found
    return result, names

def get_all_module_with_name(model, name):
    result = []
    names = []
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if name in n:
            result.append(m)
            names.append(n)
    assert(len(result) > 0)
    return result, names

def freeze_parameters(modules):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False
        from torch.nn import BatchNorm2d
        if isinstance(m, BatchNorm2d):
            m.eval()

class StandardYoloPredictModel(nn.Module):
    def __init__(self, model, predictor):
        super(StandardYoloPredictModel, self).__init__()
        self.module = model
        self.predictor = predictor

    def forward(self, im_info):
        result = []
        for im, h, w in im_info:
            im = im.unsqueeze_(0)
            im = im.float()
            if next(self.parameters()).is_cuda:
                im = im.cuda()
            with torch.no_grad():
                features = self.module(im)
            prob, bbox = self.predictor(features, torch.Tensor((h, w)))
            result.append({'prob': prob,
                'bbox': bbox,
                'h': h,
                'w': w})
        return result

class RegionTargetPt(nn.Module):
    def __init__(self, biases=None, rescore=True, anchor_aligned_images=12800, coord_scale=1.0, positive_thresh=0.6,
                 gpus_size=1, seen_images=0, opt_anchor=False):
        super(RegionTargetPt, self).__init__()
        if biases is None:
            biases = []
        self.rescore = rescore  # type: bool
        self.coord_scale = coord_scale
        self.positive_thresh = positive_thresh
        self.anchor_aligned_images = anchor_aligned_images
        self.gpus_size = gpus_size
        self.opt_anchor = opt_anchor
        if not opt_anchor:
            import numpy as np
            self.register_buffer('biases', torch.from_numpy(np.array(biases, dtype=np.float32).reshape(-1, 2)))
        else:
            pt_bias = torch.tensor(biases, requires_grad=True).reshape(-1, 2)
            pt_bias = torch.nn.Parameter(pt_bias)
            self.register_parameter('biases', pt_bias)
        # noinspection PyCallingNonCallable,PyUnresolvedReferences
        self.register_buffer('seen_images', torch.tensor(seen_images, dtype=torch.long))
        self.iteration = 0

    def forward(self, xy, wh, obj, truth):
        '''
        xy: 2 * 6 * 13 * 13
        wh: 2 * 6 * 13 * 13
        obj: 2 * 3 * 13 * 13
        truth: 2 * 150
        '''
        xy, wh, obj = xy.detach(), wh.detach(), obj.detach()
        print_debug = False
        if self.opt_anchor and (self.iteration % 100) == 0:
            print_debug = True
        if print_debug:
            logging.info('bias.abs().sum() = {}'.format(self.biases.abs().sum()))
        self.iteration += 1
        self.seen_images += xy.size(0) * self.gpus_size
        warmup = self.seen_images.item() < self.anchor_aligned_images

        bbs = self.get_bbs(xy, wh)

        truth = truth.view(truth.shape[0], -1, 5)
        iou = self.calculate_iou(
                bbs.view(bbs.shape[0], -1, 4),
                truth)

        iou = iou.view(*obj.shape, -1)

        if warmup:
            t_xy = torch.full_like(xy, 0.5)
            t_wh = torch.zeros_like(wh)
            t_xywh_weight = torch.full_like(xy, 0.01)
        else:
            t_xy = xy.clone()
            t_wh = wh.clone()
            t_xywh_weight = torch.zeros_like(xy)

        t_o_noobj = torch.zeros_like(obj)
        t_o_obj = obj.clone()
        t_label = torch.full_like(obj, -1)

        ignorable = iou.max(dim=-1)[0] > self.positive_thresh
        t_o_noobj[ignorable] = obj[ignorable]

        num_image, num_anchor2, h, w = xy.shape

        target_i, target_j, target_anchor = self.calc_gt_target(truth, w, h)
        self.disable_invalid_target(truth, target_i, target_j, target_anchor)

        num_gt = truth.shape[1]
        num_anchor = obj.shape[1]

        if print_debug:
            x = target_anchor.view((-1, 1)) == torch.arange(num_anchor).to(target_anchor.device).view((1, -1))
            num_each_anchor = x.sum(dim=0)
            logging.info('#each anchor = {}'.format(num_each_anchor))
        for idx_image in range(num_image):
            for idx_gt in range(num_gt):
                i = target_i[idx_image, idx_gt]
                j = target_j[idx_image, idx_gt]
                n = target_anchor[idx_image, idx_gt]
                if i < 0 or j < 0:
                    continue
                if j >= h or i >= w:
                    continue
                tx, ty, tw, th, cls = truth[idx_image, idx_gt]
                if tw <= 0 or th <= 0:
                    continue
                t_xy[idx_image, n, j, i] = tx * w - i
                t_xy[idx_image, n + num_anchor, j, i] = ty * h - j
                t_wh[idx_image, n, j, i] = torch.log(tw*w/self.biases[n, 0])
                t_wh[idx_image, n + num_anchor, j, i] = torch.log(th*h/self.biases[n, 1])
                t_xywh_weight[idx_image, n, j, i] = self.coord_scale * (2 - tw * th)
                t_xywh_weight[idx_image, n + num_anchor, j, i] = self.coord_scale * (2 - tw * th)

                if not self.rescore:
                    t_o_obj[idx_image, n, j, i] = 1
                else:
                    t_o_obj[idx_image, n, j, i] = iou[idx_image, n, j, i, idx_gt]
                t_o_noobj[idx_image, n, j, i] = obj[idx_image, n, j, i]
                t_label[idx_image, n, j, i] = cls

        return t_xy, t_wh, t_xywh_weight, t_o_obj, t_o_noobj, t_label

    def disable_invalid_target(self, truth, target_i, target_j, target_anchor):
        invalid_pos = ((truth[:, :, 0] <= 0) |
                (truth[:, :, 0] >= 1) |
                (truth[:, :, 1] <= 0) |
                (truth[:, :, 1] >= 1))
        target_i[invalid_pos] = -1
        target_j[invalid_pos] = -1
        target_anchor[invalid_pos] = -1

    def calc_gt_target(self, truth, w, h):
        '''
        truth = B x T x 5
        '''
        target_i = (truth[:, :, 0] * w).long()
        target_j = (truth[:, :, 1] * h).long()
        target_i[target_i >= w] = w
        target_j[target_j >= h] = h

        inter_left = torch.max(-self.biases[None, :, 0, None] / 2 / w,
            - truth[:, None, :, 2] / 2)
        inter_right = torch.min(self.biases[None, :, 0, None] / 2 / w,
            truth[:, None, :, 2] / 2)
        inter_top = torch.max(-self.biases[None, :, 1, None] / 2 / h,
            - truth[:, None, :, 3] / 2)
        inter_bottom = torch.min(self.biases[None, :, 1, None] / 2 / h,
            truth[:, None, :, 3] / 2)
        overlap = (inter_right - inter_left).clamp(min=0.) * (inter_bottom -
                inter_top).clamp(min=0)
        a1 = self.biases[None, :, 0, None] * self.biases[None, :, 1, None] / w / h
        a2 = truth[:, None, :, 2] * truth[:, None, :, 3]
        iou = overlap / (a1 + a2 - overlap)
        _, target_anchor = torch.max(iou, dim=1)
        return target_i, target_j, target_anchor

    def calculate_iou(self, bbs, truth):
        # num_image x num_pred x 1 + num_image x 1 x num_t
        inter_left = torch.max(bbs[:, :, 0, None] - bbs[:, :, 2, None] / 2,
            truth[:, None, :, 0] - truth[:, None, :, 2] / 2)
        inter_right = torch.min(bbs[:, :, 0, None] + bbs[:, :, 2, None] / 2,
            truth[:, None, :, 0] + truth[:, None, :, 2] / 2)
        inter_top = torch.max(bbs[:, :, 1, None] - bbs[:, :, 3, None] / 2,
            truth[:, None, :, 1] - truth[:, None, :, 3] / 2)
        inter_bottom = torch.min(bbs[:, :, 1, None] + bbs[:, :, 3, None] / 2,
            truth[:, None, :, 1] + truth[:, None, :, 3] / 2)
        overlap = (inter_right - inter_left).clamp(min=0.) * (inter_bottom -
                inter_top).clamp(min=0)
        a1 = bbs[:, :, 2, None] * bbs[:, :, 3, None]
        a2 = truth[:, None, :, 2] * truth[:, None, :, 3]
        iou = overlap / (a1 + a2 - overlap)
        return iou

    def get_bbs(self, xy, wh):
        # from mtorch.yolobbs.forward
        n, anchors, height, width = xy.size()
        anchors //= 2
        bbs = xy.new_empty((n, anchors, height, width, 4))
        # tensor views into input
        x = xy[:, :anchors, :, :]
        y = xy[:, anchors:, :, :]
        w = wh[:, :anchors, :, :]
        h = wh[:, anchors:, :, :]

        # Note: avoid aliasing the output tensors, for AutoGrad (if we ever wanted to compute backward)

        # Use broadcasting to convert to Yolo bounding box in-place
        i = torch.arange(width, dtype=xy.dtype, device=xy.device)
        bbs[:, :, :, :, 0] = (x + i) / width
        del i
        j = torch.arange(height, dtype=xy.dtype, device=xy.device).view(-1, 1)
        bbs[:, :, :, :, 1] = (y + j) / height
        del j
        bbs[:, :, :, :, 2] = w.exp() * self.biases[:, 0].view(anchors, 1, 1) / width
        bbs[:, :, :, :, 3] = h.exp() * self.biases[:, 1].view(anchors, 1, 1) / height

        return bbs

    def extra_repr(self):
        """Extra information
        """
        return '{}biases={}{}'.format(
            "rescore, " if self.rescore else "", [round(b.item(), 3) for b in self.biases.view(-1)],
            ", seen_images={}".format(self.seen_images.item()) if self.seen_images.item() else ""
        )

class NegativeSelectiveLoss(nn.Module):
    def __init__(self, max_iter, loss_weight, num_select_exp,
            entropy_weight_exp):
        super().__init__()
        print_frame_info()
        self.register_buffer('curr_iter', torch.tensor(0, dtype=torch.float))
        self.max_iter = max_iter
        self.loss_weight = loss_weight
        self.num_select_exp = num_select_exp
        self.entropy_weight_exp = entropy_weight_exp

    def forward(self, pred, neg_target):
        self.curr_iter += 1
        num_image = pred.shape[0]
        pred = pred.view(-1)
        neg_target = neg_target.view(-1)

        neg_idx = torch.nonzero(neg_target == 0)
        progress_iter = self.curr_iter / self.max_iter
        display = (self.curr_iter % 20) == 0
        messages = []
        if display:
            messages.append('{}/{}={}'.format(self.curr_iter, self.max_iter,
                progress_iter))
        num_select_factor = progress_iter ** self.num_select_exp
        num_select = (neg_idx.numel() * torch.sin(num_select_factor * math.pi / 2.)).long()
        if display:
            messages.append('num_select = {}'.format(num_select))
        if num_select == 0:
            loss = torch.tensor(0, device=pred.device, dtype=pred.dtype, requires_grad=True)
            return loss
        select_idx_in_idx = torch.randperm(len(neg_idx))[:num_select]
        neg_idx = neg_idx[select_idx_in_idx]

        pos_avg = pred[neg_target > 0].mean().detach()

        if display:
            messages.append('pos_avg = {}'.format(pos_avg))

        pred = pred[neg_idx]

        entropy_weight_factor = progress_iter ** self.entropy_weight_exp
        entropy_weight = progress_iter ** entropy_weight_factor
        loss = (1 - entropy_weight) * (pred * pred).sum() / 2. / num_image

        entropy = -(pred * torch.log((pred + 10**-5) / (pos_avg + 10**-5)) + \
                (1 - pred) * torch.log((1 - pred + 10**-5) / (1 - pos_avg + 10**-5))).sum()
        loss += entropy_weight * entropy / num_image
        if display:
            logging.info('; '.join(messages))

        return loss * self.loss_weight

class RegionTargetLoss(nn.Module):
    """Abstract class for constructing different kinds of RegionTargetLosses
    Parameters:
        num_classes: int, number of classes for classification
        biases: list, default anchors
        obj_esc_thresh: int, objectness threshold
        rescore: boolean,
        xy_scale: float, weight of the xy loss
        wh_scale: float, weight of the wh loss
        object_scale: float, weight of the objectness loss
        noobject_scale: float, weight of the no-objectness loss
        coord_scale:float,
        anchor_aligned_images: int, threshold to pass to warm stage
        ngpu: int, number of gpus
        seen_images: int, number of images seen by the model
    """

    def __init__(self, num_classes,
                 biases,
                 obj_esc_thresh=0.6, rescore=True, xy_scale=1.0,
                 wh_scale=1.0,
                 object_scale=5.0, noobject_scale=1.0, coord_scale=1.0,
                 anchor_aligned_images=12800, ngpu=1, seen_images=0,
                 valid_norm_xywhpos=False, use_pt_rt=False,
                 opt_anchor=False,
                 neg_selective_loss=False,
                 max_iter=None,
                 num_select_exp=None,
                 entropy_weight_exp=None,
                 ):
        print_frame_info()
        super(RegionTargetLoss, self).__init__()
        self.num_classes = num_classes
        assert (len(biases) % 2) == 0
        self.num_anchors = len(biases) // 2
        slice_points = [self.num_anchors * 2, self.num_anchors * 4, self.num_anchors * 5]
        slice_points.append((num_classes + 4 + 1) * self.num_anchors)
        self.slice_region = Slice(1, slice_points)
        if opt_anchor:
            if not use_pt_rt:
                logging.info('use pt version of rt')
            use_pt_rt = True
        if not use_pt_rt:
            assert not opt_anchor
            self.region_target = RegionTarget(
                biases, rescore=rescore,
                anchor_aligned_images=anchor_aligned_images,
                coord_scale=coord_scale,
                positive_thresh=obj_esc_thresh,
                gpus_size=ngpu,
                seen_images=seen_images
            )
        else:
            self.region_target = RegionTargetPt(
                biases, rescore=rescore,
                anchor_aligned_images=anchor_aligned_images,
                coord_scale=coord_scale,
                positive_thresh=obj_esc_thresh,
                gpus_size=ngpu,
                seen_images=seen_images,
                opt_anchor=opt_anchor,
            )
        self.xy_loss = EuclideanLoss(loss_weight=xy_scale)
        self.wh_loss = EuclideanLoss(loss_weight=wh_scale)
        self.o_obj_loss = EuclideanLoss(loss_weight=object_scale)
        if neg_selective_loss:
            self.o_noobj_loss = NegativeSelectiveLoss(max_iter=max_iter,
                    loss_weight=noobject_scale,
                    num_select_exp=num_select_exp,
                    entropy_weight_exp=entropy_weight_exp)
        else:
            self.o_noobj_loss = EuclideanLoss(loss_weight=noobject_scale)
        reshape_axis = 1
        reshape_num_axes = 1
        self.reshape_conf = Reshape(self.shape, reshape_axis, reshape_num_axes)
        self.valid_norm_xywhpos = valid_norm_xywhpos

    def forward(self, x, label):
        """
        :param x: torch tensor, the input to the loss layer
        :param label: [N x (coords+1)], expected format: x,y,w,h,cls
        :return: float32 loss scalar 
        """
        xy, wh, obj, conf = self.slice_region(x)
        sig = nn.Sigmoid()
        xy = sig(xy)
        obj = sig(obj)
        conf = self.reshape_conf(conf)
        t_xy, t_wh, t_xywh_weight, t_o_obj, t_o_noobj, t_label = self.region_target(xy, wh, obj, label)
        o_obj_loss_func = self.o_obj_loss
        if self.valid_norm_xywhpos:
            total = (label.view(label.shape[0], -1, 5).sum(dim=2) > 0).sum().float()
            if total > 0:
                t_xywh_weight /= total
                o_obj_loss_func = EuclideanLoss(loss_weight= self.o_obj_loss.loss_weight / total)
            # all these 3 losses should be 0 if total = 0 and thus we don't
            # need to norm it here
        xy_loss = self.xy_loss(xy, t_xy, t_xywh_weight)
        wh_loss = self.wh_loss(wh, t_wh, t_xywh_weight)
        o_obj_loss = o_obj_loss_func(obj, t_o_obj)

        o_noobj_loss = self.o_noobj_loss(obj, t_o_noobj)
        cls_loss = self.classifier_loss(conf, t_label)
        return {'xy': xy_loss, 'wh': wh_loss, 'o_obj': o_obj_loss,
                'o_noobj': o_noobj_loss, 'cls': cls_loss}

    @property
    def seen_images(self):
        """getter to the number of images that were evaluated by model"""
        return self.region_target.seen_images

    def classifier_loss(self, x, label):
        """calculates classification loss (SoftMaxTreeLoss)
        after permuting the dimensions of input features (compatibility to Caffe)
        :param x: torch tensor, features
        :param label: ground truth label
        :return torch tensor with loss value
        """
        raise NotImplementedError(
            "Please create an instance of RegionTargetWithSoftMaxLoss or RegionTargetWithSoftTreeMaxLoss")

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        raise NotImplementedError(
            "Please create an instance of RegionTargetWithSoftMaxLoss or RegionTargetWithSoftTreeMaxLoss")


class RegionTargetWithSoftMaxLoss(RegionTargetLoss):
    """Extends RegionTargetLosses by calculating classification loss based on SoftMaxLoss"""

    def __init__(self, ignore_label=-1, class_scale=1.0, normalization='valid', **kwargs):
        super(RegionTargetWithSoftMaxLoss, self).__init__(**kwargs)
        self.normalization = normalization
        self._classifier_loss = SoftmaxWithLoss(loss_weight=class_scale, ignore_label=ignore_label,
                                                valid_normalization=(normalization
                                                    == 'valid'))

    def classifier_loss(self, x, label):
        """calculates classification loss (SoftMaxTreeLoss)
        after permuting the dimensions of input features (compatibility to Caffe)
        :param x: torch tensor, features
        :param label: ground truth label
        :return torch tensor with loss value
        """
        return self._classifier_loss(x.permute([0, 2, 1, 3, 4]), label)

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        return [self.num_anchors, self.num_classes]


class RegionTargetWithSoftMaxTreeLoss(RegionTargetLoss):
    """Extends RegionTargetLosses by calculating classification loss based on SoftMaxTreeLoss"""

    def __init__(self, tree, ignore_label=-1, class_scale=1.0,
            normalization='valid', **kwargs):
        super(RegionTargetWithSoftMaxTreeLoss, self).__init__(**kwargs)
        self.normalization = normalization
        self._classifier_loss = SoftmaxTreeWithLoss(
            tree, ignore_label=ignore_label, loss_weight=class_scale,
            valid_normalization=(normalization == 'valid')
        )

    def classifier_loss(self, x, label):
        """calculates classification loss (SoftMaxTreeLoss)
        :param x: torch tensor, features
        :param label: ground truth label
        :return torch tensor with loss value
        """
        return self._classifier_loss(x, label)

    @property
    def shape(self):
        """
        :return: list, dimensions for Reshape
        """
        return [self.num_classes, self.num_anchors]

class YoloByMask(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super(YoloByMask, self).__init__(**kwargs)
        self._default.update({'stagelr': [0.0001,0.001,0.0001,0.00001],
            'effective_batch_size': 64,
            'ovthresh': [0.5],
            'display': 100,
            'lr_policy': 'multifixed',
            'momentum': 0.9,
            'is_caffemodel_for_predict': False,
            'num_extra_convs': 2,
            'anchors': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
            'test_input_size': 416,
            'nms_threshold': 0.45,
            'obj_thresh': 0.01,
            'thresh': 0.01,
            'num_workers': 16,
            'opt_anchor_lr_mult': 1.,
            'max_detections_per_image': -1,
            'train_version': 0,
            'weight_decay': 5e-4,
            })
        self.num_train_images = None

        if self.yolo_train_session_param is not None:
            from qd.qd_common import dict_update_nested_dict
            dict_update_nested_dict(self.kwargs, self.yolo_train_session_param)

    def append_predict_param(self, cc):
        super(YoloByMask, self).append_predict_param(cc)
        if self.test_input_size != 416:
            cc.append('InputSize{}'.format(self.test_input_size))
        if self.single_class_nms:
            cc.append('class_agnostic_nms')
        if self.nms_threshold != 0.45:
            cc.append('NMS{}'.format(self.nms_threshold))

    def _get_checkpoint_file(self, epoch=None, iteration=None):
        assert epoch is None, 'not supported'
        if iteration is None:
            iteration = self.max_iter
        iteration = self.parse_iter(iteration)
        return op.join(self.output_folder, 'snapshot',
                "model_iter_{:07d}.pt".format(iteration))

    def monitor_train(self):
        # not implemented. but we just return rather than throw exception
        # need to use maskrcnn checkpointer to load and save the intermediate
        # models.
        return

    def get_optimizer(self, model):
        decay, no_decay, lr2 = [], [], []
        rt_anchor = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self.opt_anchor_lr_mult != 1 and name.endswith(
                    'region_target.biases'):
                rt_anchor.append(param)
            elif "last_conv" in name and name.endswith(".bias"):
                lr2.append(param)
            elif "scale" in name:
                decay.append(param)
            elif len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        lrs = self.stagelr
        param_groups = [{'params': no_decay, 'weight_decay': 0., 'initial_lr': lrs[0], 'lr_mult': 1.},
                        {'params': decay, 'initial_lr': lrs[0], 'lr_mult': 1.},
                        {'params': lr2, 'weight_decay': 0., 'initial_lr': lrs[0] * 2., 'lr_mult': 2.}]
        if len(rt_anchor) > 0:
            param_groups.append({'params': rt_anchor, 'weight_decay': 0., 'initial_lr': lrs[0],
                'lr_mult': self.opt_anchor_lr_mult})
        optimizer = CaffeSGD(param_groups, lr=lrs[0],
                             momentum=float(self.momentum),
                             weight_decay=float(self.weight_decay))
        return optimizer

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        steps = list(map(lambda x:int(x*self.max_iter/7000),
            [100,5000,6000,7000]))
        scheduler = MultiFixedScheduler(optimizer, steps, self.stagelr,
                                        last_iter=last_epoch)
        return scheduler

    def get_train_data_loader(self, start_iter=0):
        dataset = TSVDataset(self.data)
        train_data_path = '$'.join([self.data, 'train', str(self.train_version)])
        param = {'use_maskrcnn_sampler': True,
                 'solver_params': {'max_iter': self.max_iter}}
        if self.yolo_train_session_param is not None:
            param.update(self.yolo_train_session_param)
        data_loader = yolo_train_data_loader(train_data_path,
                cmapfile=dataset.get_labelmap_file(),
                batch_size=self.effective_batch_size // self.mpi_size,
                num_workers=self.num_workers,
                distributed=self.distributed,
                kwargs=param)
        return data_loader

    def get_test_data_loader(self):
        from mtorch.augmentation import TestAugmentation
        test_augmenter = TestAugmentation(test_input_size=self.test_input_size)

        from mtorch.imdbtsvdata import ImdbTSVData
        augmented_dataset = ImdbTSVData(
                path='$'.join([self.test_data, self.test_split]),
                cmapfile=self.get_labelmap(),
                transform=test_augmenter(),
                labeler=None,
                predict_phase=True)

        from maskrcnn_benchmark.data import samplers
        sampler = samplers.DistributedSampler(augmented_dataset,
                shuffle=False,
                length_divisible=self.batch_size)

        from torch.utils.data import DataLoader
        return DataLoader(augmented_dataset,
                batch_size=self.test_batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=_list_collate)

    def predict_one(self, cv_im):
        h, w = cv_im.shape[: 2]
        model = self._get_load_model_demo()
        from qd.layers import ForwardPassTimeChecker
        model = ForwardPassTimeChecker(model)
        from mtorch.augmentation import TestAugmentation
        transform = TestAugmentation(test_input_size=self.test_input_size)
        img = cv_im.transpose([2, 0, 1])
        im = transform()({'image': img})['image']

        im = im.to(self.device)
        output = model([(im, h, w)])[0]
        return self.create_rects_from_output(output)

    def get_train_model(self):
        assert self.num_extra_convs == 2
        extra_yolo2extraconv_param = {}
        if self.anchors is not None:
            assert (len(self.anchors) % 2) == 0
            extra_yolo2extraconv_param['num_anchors'] = len(self.anchors) // 2
        num_classes = len(self.get_labelmap())
        model = yolo_2extraconv(self.get_backbone_model(),
                weights_file=None,
                num_classes=num_classes,
                **extra_yolo2extraconv_param
                )
        model.train()

        if self.last_fixed_param is not None:
            freeze_parameters_by_last_name(model, last_fixed_param=self.last_fixed_param)
        if self.fixed_param is not None:
            freeze_parameters_by_name(model, self.fixed_param)
        target_param = {
                'num_classes': len(self.get_labelmap()),
                'neg_selective_loss': self.neg_selective_loss,
                'max_iter': self.max_iter,
                'num_select_exp': self.num_select_exp,
                'entropy_weight_exp': self.entropy_weight_exp,
                }
        target_param['biases'] = self.anchors
        if self.rt_param is not None:
            target_param.update(self.rt_param)
        if self.use_treestructure:
            dataset = TSVDataset(self.data)
            criterion = RegionTargetWithSoftMaxTreeLoss(dataset.get_tree_file(), **target_param)
            logging.info("Using tree structure with {} softmax normalization".format(criterion.normalization))
        else:
            criterion = RegionTargetWithSoftMaxLoss(**target_param)
            logging.info("Using plain structure with {} softmax normalization".format(criterion.normalization))

        model = ModelLoss(model, criterion)
        return model

    def get_backbone_model(self):
        if self.net == 'darknet19_448':
            return darknet_layers()
        elif self.net == 'darknet_dilate':
            from mtorch.darknet import DarkDilatedNet
            return DarkDilatedNet()
        else:
            raise NotImplementedError

    def get_test_model(self):
        assert self.num_extra_convs == 2
        extra_yolo2extraconv_param = {}
        if self.anchors is not None:
            assert (len(self.anchors) % 2) == 0
            extra_yolo2extraconv_param['num_anchors'] = len(self.anchors) // 2
        num_classes = len(self.get_labelmap())
        model = yolo_2extraconv(self.get_backbone_model(),
                weights_file=None,
                num_classes=num_classes,
                **extra_yolo2extraconv_param
                )
        kwargs = {}
        kwargs['biases'] = self.anchors
        kwargs['num_anchors'] = len(self.anchors) // 2
        kwargs['feat_stride'] = 32 if self.net == 'darknet19_448' else 16
        from mtorch.yolo_predict import PlainPredictorSingleClassNMS, PlainPredictorClassSpecificNMS, \
                                TreePredictorSingleClassNMS, TreePredictorClassSpecificNMS
        if self.use_treestructure:
            dataset = TSVDataset(self.data)
            if self.single_class_nms:
                predictor = TreePredictorSingleClassNMS(dataset.get_tree_file(),
                        num_classes=num_classes,
                        nms_threshold=self.nms_threshold,
                        **kwargs,
                        )
            else:
                predictor = TreePredictorClassSpecificNMS(dataset.get_tree_file(),
                        num_classes=num_classes,
                        nms_threshold=self.nms_threshold,
                        **kwargs,
                        )
        else:
            if self.single_class_nms:
                predictor = PlainPredictorSingleClassNMS(num_classes=num_classes,
                        nms_threshold=self.nms_threshold,
                        **kwargs,
                        )
            else:
                predictor = PlainPredictorClassSpecificNMS(num_classes=num_classes,
                    nms_threshold=self.nms_threshold,
                    **kwargs,
                    )
        model = StandardYoloPredictModel(model, predictor)
        return model

    def load_test_model(self, model, model_file):
        from maskrcnn_benchmark.utils.model_serialization import load_state_dict
        from qd.qd_pytorch import torch_load
        checkpoint = torch_load(model_file)
        load_state_dict(model, checkpoint['model'])
        # check if the anchor size is the same
        key = ('criterion.region_target.biases'
                if 'criterion.region_target.biases' in checkpoint['model']
                else 'module.criterion.region_target.biases')
        anchor = checkpoint['model'][key]
        anchor = anchor.data.view(model.predictor.yolo_bboxs.biases.data.shape)
        diff = (model.predictor.yolo_bboxs.biases.data - anchor).abs().sum()
        if diff != 0:
            logging.info('we will respect the model in the checkpoint')
            logging.info('change the anchor from {} to {}'.format(
                model.predictor.yolo_bboxs.biases,
                anchor.data))
            model.predictor.yolo_bboxs.biases.data = anchor.data

    def create_rects_from_output(self, info):
        bbox, prob = info['bbox'], info['prob']
        h, w = info['h'], info['w']
        bbox = bbox.cpu().numpy()
        prob = prob.cpu().numpy()

        assert bbox.shape[-1] == 4
        bbox = bbox.reshape(-1, 4)
        prob = prob.reshape(-1, prob.shape[-1])
        from mmod.detection import result2bblist
        result = result2bblist((h, w), prob, bbox, self.get_labelmap(),
                thresh=self.thresh,
                obj_thresh=self.obj_thresh)
        return result

    def predict_output_to_tsv_row(self, output, keys):
        for info, key in zip(output, keys):
            result = self.create_rects_from_output(info)
            yield key, json_dump(result)

    def _get_test_normalize_module(self):
        return

