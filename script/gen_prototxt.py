if __name__ == "__main__":  # for debug
    import _init_paths

import os
import re
import caffe
import quickcaffe.modelzoo as mzoo

model_dict = {
    'zfb': mzoo.ZFNet(add_last_pooling_layer=False, rcnn_mode=True),
    'vgg': mzoo.VGG(add_last_pooling_layer=False, rcnn_mode=True),
    'resnet': mzoo.ResNet(add_last_pooling_layer=False, rcnn_mode=True),
    'squeezenet': mzoo.SqueezeNet(add_last_pooling_layer=False, rcnn_mode=True),
    }

def list_models():
    return ['zfb', 'vgg', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet']

def gen_net_prototxt(basemodel, num_classes, deploy=False):
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

    model.add_body(n, lr=1, depth=model_depth, deploy=deploy)
    rcnn.add_body(n, lr=1, num_classes=num_classes, deploy=deploy)

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
    with open(train_net_file, 'w') as f:
        f.write(gen_net_prototxt(basemodel, num_classes, deploy=False))
    with open(deploy_net_file, 'w') as f:
        f.write(gen_net_prototxt(basemodel, num_classes, deploy=True))

    solver_file = os.path.join(output_path, 'solver.prototxt')
    with open(solver_file, 'w') as f:
        f.write(gen_solver_prototxt(basemodel, train_net_file))

if __name__ == "__main__":
    generate_prototxt(r'models\ZF.caffemodel', r'data\voc20', 5000, 21, r'output\voc20_ZF_test')
    print('Done.')