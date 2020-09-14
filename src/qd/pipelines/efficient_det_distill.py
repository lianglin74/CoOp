from qd.pipelines.efficient_det_pipeline import EfficientDetPipeline
from collections import OrderedDict
import torch
from qd.torch_common import torch_load, torch_save

from torch import nn
from qd.layers.efficient_det import EfficientDetBackbone

def merge_student_teacher_basemodel(
    # specific for efficient
    teacher_basemodel, student_basemodel, basemodel):

    teacher = torch_load(teacher_basemodel)
    student = torch_load(student_basemodel)

    out = OrderedDict()
    for k, v in student['model'].items():
        # remove the prefix in
        while k.startswith('module.'):
            k = k[len('module.'):]
        # add the prefix, which is specific to the DistillModel
        k = 'module.backbone_net.model.' + k
        out[k] = v

    for k, v in teacher['model'].items():
        # remove the prefix in
        while k.startswith('module.'):
            k = k[len('module.'):]
        # add the prefix, which is specific to the DistillModel
        k = 'teacher.' + k
        out[k] = v
    torch_save(out, basemodel)

class DistillLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        from qd.layers.efficient_det import FocalLoss
        self.loss = FocalLoss(alpha=config.focal_alpha,
                         gamma=config.focal_gamma,
                         cls_loss_type=config.cls_loss_type,
                         reg_loss_type=config.reg_loss_type,
                         smooth_bce_pos=config.smooth_bce_pos,
                         smooth_bce_neg=config.smooth_bce_neg,
                         at_least_1_assgin=config.at_least_1_assgin,
                         neg_iou_th=config.neg_iou_th,
                         pos_iou_th=config.pos_iou_th,
                         cls_weight=config.cls_weight,
                         reg_weight=config.reg_weight,
                         cls_target_on_iou=config.cls_target_on_iou,
                         assigner_type=config.assigner_type,
                         atss_topk=config.atss_topk,
                         wh_transform_type=config.wh_transform_type,
                         )
        bridge_direct = 'st'
        bridges = nn.ModuleList()
        for i in range(5):
            teacher_channel = EfficientDetBackbone.fpn_num_filters[
                config.teacher_net]
            student_channel = EfficientDetBackbone.fpn_num_filters[
                config.net]
            if bridge_direct == 'st':
                bridge = nn.Conv2d(in_channels=student_channel,
                                   out_channels=teacher_channel,
                                   kernel_size=1)
                bridges.append(bridge)
            else:
                raise NotImplementedError


        self.bridges = bridges
        self.bridge_direct = 'st'

        self.distill_cls = config.distill_cls
        self.distill_cls_weight = config.distill_cls_weight
        self.distill_hint = config.distill_hint
        self.distill_hint_weight = config.distill_hint_weight

    def feature_alignment_loss(self, teacher_features, student_features):
        loss = 0
        for i, (teacher, student) in enumerate(zip(
                teacher_features, student_features)):
            if self.bridge_direct == 'st':
                predict_teacher = self.bridges[i](student)
                loss += (teacher - predict_teacher).abs().mean()
            else:
                raise NotImplementedError
        return loss

    def forward(self, annotations, guideline, predict):
        result = {}
        features, regression, classification, anchors = predict

        extra_param = {}
        if self.distill_cls:
            extra_param['guide_classifications'] = guideline[2]
        loss_array = self.loss(
            classification,
            regression,
            anchors,
            annotations,
            **extra_param,
        )
        result['cls_loss'] = loss_array[0]
        result['reg_loss'] = loss_array[1]
        if self.distill_cls:
            result['distill_cls_loss'] = loss_array[2]

        if self.distill_hint:
            # calculate the difference between the guideline and the student
            hint_loss = self.feature_alignment_loss(guideline[0], features)

            hint_loss *= self.distill_hint_weight
            hint_loss = hint_loss[None]

            result['hint_loss'] = hint_loss

        return result

class DistillModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.teacher = EfficientDetBackbone(
            num_classes=len(config.labelmap),
            compound_coef=config.teacher_net,
            ratios=config.anchors_ratios,
            scales=config.anchors_scales,
            prior_prob=config.prior_prob,
            adaptive_up=config.adaptive_up,
            anchor_scale=config.anchor_scale,
            drop_connect_rate=config.drop_connect_rate,
        )

        # we use the name of module because the model loading utility can
        # ignore this prefix, which is good for inference
        self.module = EfficientDetBackbone(
            num_classes=len(config.labelmap),
            compound_coef=config.net,
            ratios=config.anchors_ratios,
            scales=config.anchors_scales,
            prior_prob=config.prior_prob,
            adaptive_up=config.adaptive_up,
            anchor_scale=config.anchor_scale,
            drop_connect_rate=config.drop_connect_rate,
        )

        self.loss = DistillLoss(config)

    def forward(self, *args):
        assert len(args) == 1
        imgs, annotations = args[0][:2]
        self.teacher.eval()
        with torch.no_grad():
            guideline = self.teacher(imgs)
        predict = self.module(imgs)
        loss = self.loss(annotations, guideline, predict)
        return loss

class EfficientDetDistillPipeline(EfficientDetPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        curr_default = {
            'teacher_net': 3,
        }
        self._default.update(curr_default)

    def get_train_model(self):
        return DistillModel(self)

