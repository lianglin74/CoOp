from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from .layerfactory import *

class FasterRCNN(object):
    def add_input_data(self, n, num_classes):
        n.data, n.im_info, n.gt_boxes = L.Python(ntop=3, python_param=dict(module='roi_data_layer.layer', layer='RoIDataLayer', param_str="'num_classes': %d"%num_classes))

    def add_body(self, n, lr, num_classes, cnnmodel, deploy=False, cpp_version=False):
        bottom = last_layer(n)
        
        # rpn
        n['rpn_conv/3x3'], n['rpn_relu/3x3'] = conv_relu(bottom, nout=256, ks=3, stride=1, pad=1, lr=lr, deploy=deploy,
                  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

        n.rpn_cls_score = conv(n['rpn_relu/3x3'], nout=2*9, ks=1, stride=1, pad=0, lr=lr, deploy=deploy,
                  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
        n.rpn_bbox_pred = conv(n['rpn_relu/3x3'], nout=4*9, ks=1, stride=1, pad=0, lr=lr, deploy=deploy,
                  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

        n.rpn_cls_score_reshape = L.Reshape(n.rpn_cls_score, reshape_param=dict(shape=dict(dim=[0,2,-1,0])))

        if not deploy:
            n.rpn_labels, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights = \
                L.Python(n.rpn_cls_score, n.gt_boxes, n.im_info, n.data, ntop=4, 
                      python_param=dict(module='rpn.anchor_target_layer', layer='AnchorTargetLayer', param_str="'feat_stride': 16"))

            n.rpn_loss_cls = L.SoftmaxWithLoss(n.rpn_cls_score_reshape, n.rpn_labels, propagate_down=[1,0], loss_weight=1, loss_param=dict(ignore_label=-1, normalize=True))
            n.rpn_loss_bbox = L.SmoothL1Loss(n.rpn_bbox_pred, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights, loss_weight=1,
                      smooth_l1_loss_param=dict(sigma=3.0))

        # roi proposal
        n.rpn_cls_prob = L.Softmax(n.rpn_cls_score_reshape)
        n.rpn_cls_prob_reshape = L.Reshape(n.rpn_cls_prob, reshape_param=dict(shape=dict(dim=[0,18,-1,0])))
        if cpp_version:
            assert deploy, "cannot generate cpp version prototxt for training. deploy must be set to True"
            n['rois'] = L.RPNProposal(n.rpn_cls_prob_reshape, n.rpn_bbox_pred, n.im_info, ntop=1,
                      rpn_proposal_param=dict(feat_stride=16))
        else:
            n['rois' if deploy else 'rpn_rois'] = \
                L.Python(n.rpn_cls_prob_reshape, n.rpn_bbox_pred, n.im_info, ntop=1,
                      python_param=dict(module='rpn.proposal_layer', layer='ProposalLayer', param_str="'feat_stride': 16"))
        if not deploy:
            n.rois, n.labels, n.bbox_targets, n.bbox_inside_weights, n.bbox_outside_weights = \
                L.Python(n.rpn_rois, n.gt_boxes, ntop=5,
                      python_param=dict(module='rpn.proposal_target_layer', layer='ProposalTargetLayer', param_str="'num_classes': %d"%num_classes))

        # rcnn
        roi_size = cnnmodel.roi_size()
        n.roi_pool_conv5 = L.ROIPooling(bottom, n.rois, roi_pooling_param=dict(pooled_w=roi_size, pooled_h=roi_size, spatial_scale=0.0625)) # 0.0625=1/16

        cnnmodel.add_body_for_roi(n, n.roi_pool_conv5, lr=lr, deploy=deploy)

        bottom = last_layer(n)
        n.cls_score = fc(bottom, nout=num_classes, lr=lr, deploy=deploy,
                  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
        n.bbox_pred = fc(bottom, nout=num_classes*4, lr=lr, deploy=deploy,
                  weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0))

        if not deploy:
            n.loss_cls = L.SoftmaxWithLoss(n.cls_score, n.labels, propagate_down=[1,0], loss_weight=1, loss_param=dict(ignore_label=-1, normalize=True))
            n.loss_bbox = L.SmoothL1Loss(n.bbox_pred, n.bbox_targets, n.bbox_inside_weights, n.bbox_outside_weights, loss_weight=1)
        else:
            n.cls_prob = L.Softmax(n.cls_score)
