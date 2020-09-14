from qd.pipelines.fcos import FCOSPipeline
from collections import OrderedDict
import torch
from qd.torch_common import torch_load, torch_save

from torch import nn
from qd.layers.efficient_det import EfficientDetBackbone
from fcos_core.structures.image_list import to_image_list

def merge_student_teacher_basemodel(
    teacher_basemodel, student_basemodel, basemodel):

    teacher = torch_load(teacher_basemodel)
    student = torch_load(student_basemodel)

    out = OrderedDict()
    for k, v in student['model'].items():
        # remove the prefix in
        while k.startswith('module.'):
            k = k[len('module.'):]
        # add the prefix, which is specific to the DistillModel
        k = 'module.backbone.backbone_net.model.' + k
        out[k] = v

    for k, v in teacher['model'].items():
        # remove the prefix in
        while k.startswith('module.'):
            k = k[len('module.'):]
        # add the prefix, which is specific to the DistillModel
        k = 'teacher.' + k
        out[k] = v
    torch_save(out, basemodel)

class DistillModel(nn.Module):
    def __init__(self, student, teacher, config):
        super().__init__()

        # easy for model loading
        self.module = student
        self.teacher = teacher

        self.T = config.T
        self.distill_roi_cls_weight = config.distill_roi_cls_weight

        self.distill_obj_weight = config.distill_obj_weight

    def distill_cls_loss(self, s_logits, t_logits):
        t_target = nn.functional.softmax(t_logits / self.T, dim=1)
        s_log_softmax = nn.functional.log_softmax(s_logits, dim=1)
        distill_cls_loss = -(t_target * s_log_softmax).sum() / t_target.shape[0]
        return self.distill_roi_cls_weight * distill_cls_loss

    def forward(self, *args):
        from qd.torch_common import query_modules_by_name
        student = self.module
        teacher = self.teacher
        xs = query_modules_by_name(student, 'backbone')
        assert len(xs) == 1
        s_backbone = xs[0]

        xs = query_modules_by_name(student, 'rpn')
        assert len(xs) == 1
        s_rpn = xs[0]

        xs = query_modules_by_name(student, 'roi_heads')
        assert len(xs) == 1
        s_roi_heads = xs[0]
        s_roi_heads.box.return_distill_info = True

        xs = query_modules_by_name(teacher, 'backbone')
        assert len(xs) == 1
        t_backbone = xs[0]


        xs = query_modules_by_name(teacher, 'roi_heads')
        assert len(xs) == 1
        t_roi_heads = xs[0]
        t_roi_heads.box.return_distill_info = True

        images, targets = args
        images = to_image_list(images)
        s_features = s_backbone(images.tensors)
        if self.distill_obj_weight:
            s_rpn.return_objectness = True
            s_rpn_res = s_rpn(images, s_features, targets)
            s_proposals, s_proposal_losses = s_rpn_res['others']
            s_objectness_logits = s_rpn_res['objectness']
            #ipdb> pp [x.shape for x in s_objectness_logits]
            #[torch.Size([2, 3, 64, 88]),
             #torch.Size([2, 3, 32, 44]),
             #torch.Size([2, 3, 16, 22]),
             #torch.Size([2, 3, 8, 11]),
             #torch.Size([2, 3, 4, 6])]
        else:
            s_proposals, s_proposal_losses = s_rpn(images, s_features, targets)
        # internally, proposals will be re-sampled
        x, detections, s_roi_loss = s_roi_heads(s_features, s_proposals, targets)
        s_detector_losses = s_roi_loss['loss']
        s_proposals = s_roi_loss['distill_info']['proposals']

        with torch.no_grad():
            t_backbone.eval()
            t_roi_heads.eval()
            t_features = t_backbone(images.tensors)
            x, result, t_detector_losses = t_roi_heads(t_features, s_proposals, targets)
            if self.distill_obj_weight:
                xs = query_modules_by_name(teacher, 'rpn.head')
                assert len(xs) == 1
                t_rpn_head = xs[0]
                t_rpn_head.eval()
                t_objectness, _ = t_rpn_head(t_features)
                #ipdb> pp [x.shape for x in t_objectness]
                #[torch.Size([2, 3, 64, 88]),
                 #torch.Size([2, 3, 32, 44]),
                 #torch.Size([2, 3, 16, 22]),
                 #torch.Size([2, 3, 8, 11]),
                 #torch.Size([2, 3, 4, 6])]
                t_objectness = [t.sigmoid() for t in t_objectness]


            #from qd.qd_pytorch import visualize_maskrcnn_input
            #t_rpn.eval()
            #t_proposals, _ = t_rpn(images, t_features, targets)
            #x, result, t_detector_losses = t_roi_heads(t_features, t_proposals, targets)
            #visualize_maskrcnn_input(images, result, show_box=True)

        s_logits = s_roi_loss['distill_info']['class_logits']
        t_logits = t_detector_losses['class_logits']
        distill_cls_loss = self.distill_cls_loss(
            s_logits, t_logits)

        losses = {}
        losses.update(s_detector_losses)
        losses.update(s_proposal_losses)

        if self.distill_obj_weight:
            for i, (s, t) in enumerate(zip(s_objectness_logits, t_objectness)):
                l =  nn.functional.binary_cross_entropy_with_logits(s, t)
                losses['distill_obj_loss_{}'.format(i)] = self.distill_obj_weight * l

        losses['distill_cls_loss'] = distill_cls_loss

        return losses

class FasterRCNNDistillPipeline(FCOSPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        curr_default = {
            'teacher_net': 3,
            'distill_roi_cls_weight': 1.,
            'T': 1.,
        }
        self._default.update(curr_default)

    def get_train_model(self):
        student = super().get_train_model()
        from fcos_core.config import cfg
        curr_compund = cfg.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND
        cfg.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND = self.teacher_net
        teacher = super().get_train_model()
        cfg.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND = curr_compund

        return DistillModel(
            student=student,
            teacher=teacher,
            config=self)

    def get_test_model(self):
        model  = super().get_train_model()
        if self.device == 'cpu':
            # sync-bn does not support cpu
            from qd.torch_common import replace_module
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.SyncBatchNorm),
                    lambda m: torch.nn.BatchNorm2d(m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats))
        return model

