if __name__ == "__main__":  # for debug
    import _init_paths

import os
import re
import caffe
import quickcaffe.modelzoo as mzoo
from caffe import layers as L, params as P, to_proto

model_dict = {
    'zf': mzoo.ZFNet(add_last_pooling_layer=False),
    'zfb': mzoo.ZFBNet(add_last_pooling_layer=False),
    'vgg': mzoo.VGG(add_last_pooling_layer=False),
    'resnet': mzoo.ResNet(add_last_pooling_layer=False),
    'squeezenet': mzoo.SqueezeNet(add_last_pooling_layer=False),
    }

def list_models():
    return ['zf', 'zfb', 'vgg16', 'vgg19', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet']

def gen_net_prototxt(basemodel, num_classes, deploy=False, cpp_version=False):
    assert basemodel.lower() in list_models(), 'Unsupported basemodel: %s' % basemodel

    model_parts = re.findall(r'\d+|\D+', basemodel)
    model_name = model_parts[0].lower()
    model_depth = -1 if len(model_parts) == 1 else int(model_parts[1])
    
    rcnn = mzoo.FasterRCNN()
    model = model_dict[model_name]

    n = caffe.NetSpec()
    if not deploy:
        rcnn.add_input_data(n, num_classes)
    else:
        # create a placeholder, and replace later
        n.data = caffe.layers.Layer()
        n.im_info = caffe.layers.Layer()

    model.add_body_for_feature(n, depth=model_depth, lr=1, deploy=deploy)
    rcnn.add_body(n, lr=1, num_classes=num_classes, cnnmodel=model, deploy=deploy, cpp_version=cpp_version)

    layers = str(n.to_proto()).split('layer {')[1:]
    layers = ['layer {' + x for x in layers]
    if deploy:
        layers[0] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n  dim: {}\n  dim: {}\n}}\n'.format('"data"', 1, 3, 224, 224)
        layers[1] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n}}\n'.format('"im_info"', 1, 3)
    proto_str = ''.join(layers)
    proto_str = proto_str.replace("\\'", "'")
    
    return 'name: "Faster-RCNN-%s"\n' % basemodel + proto_str


def gen_rpn_prototxt(basemodel, num_classes, deploy=False, cpp_version=False):
    assert basemodel.lower() in list_models(), 'Unsupported basemodel: %s' % basemodel

    model_parts = re.findall(r'\d+|\D+', basemodel)
    model_name = model_parts[0].lower()
    model_depth = -1 if len(model_parts) == 1 else int(model_parts[1])
    
    rcnn = mzoo.FasterRCNN()
    model = model_dict[model_name]

    n = caffe.NetSpec()
    if not deploy:
        rcnn.add_input_data(n, num_classes)
    else:
        # create a placeholder, and replace later
        n.data = caffe.layers.Layer()
        n.im_info = caffe.layers.Layer()

    model.add_body_for_feature(n, depth=model_depth, lr=1, deploy=deploy)
    bottom = mzoo.last_layer(n);
    lr = 1.0
    
    # rpn
    n['rpn_conv/3x3'], n['rpn_relu/3x3'] = mzoo.conv_relu(bottom, nout=256, ks=3, stride=1, pad=1, lr=lr, deploy=deploy,
              weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

    n.rpn_cls_score =mzoo.conv(n['rpn_relu/3x3'], nout=2*9, ks=1, stride=1, pad=0, lr=lr, deploy=deploy,
              weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.rpn_bbox_pred = mzoo.conv(n['rpn_relu/3x3'], nout=4*9, ks=1, stride=1, pad=0, lr=lr, deploy=deploy,
              weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

    n.rpn_cls_score_reshape = L.Reshape(n.rpn_cls_score, reshape_param=dict(shape=dict(dim=[0,2,-1,0])))

    if not deploy:
        n.rpn_labels, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights = \
            L.Python(n.rpn_cls_score, n.gt_boxes, n.im_info, n.data, ntop=4, 
                  python_param=dict(module='rpn.anchor_target_layer', layer='AnchorTargetLayer', param_str="'feat_stride': 16"))

        n.rpn_loss_cls = L.SoftmaxWithLoss(n.rpn_cls_score_reshape, n.rpn_labels, propagate_down=[1,0], loss_weight=1, loss_param=dict(ignore_label=-1, normalize=True))
        n.rpn_loss_bbox = L.SmoothL1Loss(n.rpn_bbox_pred, n.rpn_bbox_targets, n.rpn_bbox_inside_weights, n.rpn_bbox_outside_weights, loss_weight=1,
                  smooth_l1_loss_param=dict(sigma=3.0))
    else:
        # roi proposal
        n.rpn_cls_prob = L.Softmax(n.rpn_cls_score_reshape)
        n.rpn_cls_prob_reshape = L.Reshape(n.rpn_cls_prob, reshape_param=dict(shape=dict(dim=[0,18,-1,0])))
        if cpp_version:
            assert deploy, "cannot generate cpp version prototxt for training. deploy must be set to True"
            n['rois'] = L.RPNProposal(n.rpn_cls_prob_reshape, n.rpn_bbox_pred, n.im_info, ntop=1, rpn_proposal_param=dict(feat_stride=16))
        else:
            n['rois' if deploy else 'rpn_rois'] = \
                L.Python(n.rpn_cls_prob_reshape, n.rpn_bbox_pred, n.im_info, ntop=1,
                      python_param=dict(module='rpn.proposal_layer', layer='ProposalLayer', param_str="'feat_stride': 16"))
        if not deploy:
            n.rois, n.labels, n.bbox_targets, n.bbox_inside_weights, n.bbox_outside_weights = \
                L.Python(n.rpn_rois, n.gt_boxes, ntop=5,
                      python_param=dict(module='rpn.proposal_target_layer', layer='ProposalTargetLayer', param_str="'num_classes': %d"%num_classes))    

    layers = str(n.to_proto()).split('layer {')[1:]
    layers = ['layer {' + x for x in layers]
    if deploy:
        layers[0] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n  dim: {}\n  dim: {}\n}}\n'.format('"data"', 1, 3, 224, 224)
        layers[1] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n}}\n'.format('"im_info"', 1, 3)
    proto_str = ''.join(layers)
    proto_str = proto_str.replace("\\'", "'")
    
    return 'name: "Faster-RCNN-%s"\n' % basemodel + proto_str
    
def sgd_solver_param(model_name):
    solver_param = {
        'base_lr': 0.001,
        'lr_policy': "step",
        'gamma': 0.1,
        'stepsize': 50000,
        'display': 20,
        'average_loss': 100,
        'momentum': 0.9,
        'weight_decay': 0.0005,

        'snapshot': 0,
        'iter_size': 2,
        }
    return solver_param

def gen_solver_prototxt(basemodel, train_net_file):

    model_parts = re.findall(r'\d+|\D+', basemodel)
    model_name = model_parts[0].lower()

    solver_param = sgd_solver_param(model_name)

    # Create solver.
    solver = caffe.proto.caffe_pb2.SolverParameter(
            snapshot_prefix='%s_faster_rcnn' % basemodel.lower(),
            **solver_param)

    solver_str = 'net: "%s"\n' % train_net_file.replace('\\', '\\\\')
    solver_str += str(solver)
    return solver_str

def generate_prototxt(basemodel_file, num_images, num_classes, output_path):
    basemodel = os.path.splitext(os.path.basename(basemodel_file))[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_net_file = os.path.join(output_path, 'train.prototxt')
    deploy_net_file = os.path.join(output_path, 'test.prototxt')
    deploy_cpp_net_file = os.path.join(output_path, 'test_cpp.prototxt')
    with open(train_net_file, 'w') as f:
        f.write(gen_net_prototxt(basemodel, num_classes, deploy=False, cpp_version=False))
    with open(deploy_net_file, 'w') as f:
        f.write(gen_net_prototxt(basemodel, num_classes, deploy=True, cpp_version=False))
    with open(deploy_cpp_net_file, 'w') as f:
        f.write(gen_net_prototxt(basemodel, num_classes, deploy=True, cpp_version=True))

    solver_file = os.path.join(output_path, 'solver.prototxt')
    with open(solver_file, 'w') as f:
        f.write(gen_solver_prototxt(basemodel, train_net_file))

def generate_rpn_prototxt(basemodel_file, num_images, num_classes, output_path):
    basemodel = os.path.splitext(os.path.basename(basemodel_file))[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_net_file = os.path.join(output_path, 'train.prototxt')
    deploy_net_file = os.path.join(output_path, 'test.prototxt')
    deploy_cpp_net_file = os.path.join(output_path, 'test_cpp.prototxt')
    with open(train_net_file, 'w') as f:
        f.write(gen_rpn_prototxt(basemodel, num_classes, deploy=False, cpp_version=False))
    with open(deploy_net_file, 'w') as f:
        f.write(gen_rpn_prototxt(basemodel, num_classes, deploy=True, cpp_version=False))
    with open(deploy_cpp_net_file, 'w') as f:
        f.write(gen_rpn_prototxt(basemodel, num_classes, deploy=True, cpp_version=True))

    solver_file = os.path.join(output_path, 'solver.prototxt')
    with open(solver_file, 'w') as f:
        f.write(gen_solver_prototxt(basemodel, train_net_file))        
        
