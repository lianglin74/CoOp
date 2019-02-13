import caffe
import os
import logging
import numpy as np
import multiprocessing as mp
from multiprocessing import Process
from google.protobuf import text_format


def load_net(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    return load_net_from_str(all_line)

def solve(proto, snapshot, weights, gpus, timing, uid, rank, extract_blob,
        blob_queue):
    caffe.init_glog(str(os.path.join(os.path.dirname(proto),
        'log_rank_{}_'.format(str(rank)))))
    caffe.set_device(gpus[rank])
    caffe.set_mode_gpu()
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(str(proto))
    logging.info('solve: {}'.format(str(proto)))
    assert solver is not None
    has_snapshot = snapshot and len(snapshot) != 0
    has_weight = weights and len(weights) != 0
    if has_snapshot:
        solver.restore(str(snapshot))
    elif has_weight:
        solver.net.copy_from(str(weights), ignore_shape_mismatch=True)

    if extract_blob is not None:
        logging.info('extract_blob = {}'.format(extract_blob))
        assert extract_blob in solver.net.blobs

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        register_timing(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    iters = solver.param.max_iter - solver.iter
    solver.step(iters)
    if rank == 0:
        solver.snapshot()

    if extract_blob is not None:
        blob_queue.put(solver.net.blobs[extract_blob].data)


def register_timing(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))

def caffe_train(solver_prototxt, gpu=-1, pretrained_model=None, restore_snapshot=None):
    if gpu >= 0:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    #caffe.set_random_seed(777)
    solver = caffe.SGDSolver(solver_prototxt)
    if pretrained_model:
        solver.net.copy_from(pretrained_model, ignore_shape_mismatch=True)
    if restore_snapshot:
        solver.restore(restore_snapshot)
    solver.solve()

def caffe_net_check(net):
    result = {}
    result['param'] = {}
    for key in net.params:
        value = net.params[key]
        result['param'][key] = []
        for i, v in enumerate(value):
            result['param'][key].append((float(np.mean(v.data)),
                float(np.std(v.data))))

    result['blob'] = {}
    for key in net.blobs:
        v = net.blobs[key]
        result['blob'][key] = (float(np.mean(v.data)), float(np.std(v.data)))

    return result

def caffe_param_check(caffenet, caffemodel):
    if not os.path.exists(caffenet) or not os.path.exists(caffemodel):
        return {}

    caffe.set_mode_cpu()
    net = caffe.Net(str(caffenet), str(caffemodel), caffe.TEST)

    return caffe_net_check(net)

def get_solver(solver_prototxt, restore_snapshot=None):
    solver = caffe.SGDSolver(solver_prototxt)
    if restore_snapshot:
        solver.restore(restore_snapshot)
    return solver

def parallel_train(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        weights,
        gpus,  # list of device ids
        timing=False,  # show timing info for compute and communications
        extract_blob=None
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.log('Using devices %s' % str(gpus))

    procs = []
    blob_queue = mp.Queue()
    logging.info('train on {}'.format(map(str, gpus)))
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver, snapshot, weights, gpus, timing, uid, rank,
                        extract_blob, blob_queue))
        p.daemon = True
        p.start()
        procs.append(p)

    r = None
    if extract_blob:
        for rank in range(len(gpus)):
            if r is None:
                r = blob_queue.get()
            else:
                r = r + blob_queue.get()

    for p in procs:
        p.join()

    return r

def correct_caffemodel_file_path(solverstate_file, out_file):
    solverstate = caffe.proto.caffe_pb2.SolverState()
    with open(solverstate_file, 'r') as fp:
        contents = fp.read()
    solverstate.ParseFromString(contents)
    changed = False
    if not os.path.exists(solverstate.learned_net):
        basename = os.path.basename(contents.learned_net)
        directory = os.path.dirname(solverstate)
        caffemodel = os.path.join(directory, basename)
        if os.path.exists(caffemodel):
            solverstate.learned_net = caffemodel
            changed = True
        else:
            assert False
    if changed or True:
        with open(out_file, 'w') as fp:
            fp.write(solverstate.SerializeToString())

def caffemodel_num_param(model_file):
    param = caffe.proto.caffe_pb2.NetParameter()
    with open(model_file, 'r') as fp:
        model_context = fp.read()
    param.ParseFromString(model_context)
    result = 0
    for l in param.layer:
        for b in l.blobs:
            if len(b.double_data) > 0:
                result += len(b.double_data)
            elif len(b.data) > 0:
                result += len(b.data)
    return result

def update_kernel_active2(net, **kwargs):
    kernel_active = kwargs['kernel_active']
    kernel_active_skip = kwargs.get('kernel_active_skip', 0)
    kernel_active_type = kwargs.get('kernel_active_type', 'SEQ')
    shrink_group_if_group_e_out = kwargs.get('shrink_group_if_group_e_out',
            False)
    logging.info('type: {}'.format(kernel_active_type))
    c = 0
    skipped = 0
    logging.info('{}-{}'.format(kernel_active, kernel_active_skip));
    layers = []
    bottom_map = {}
    for l in net.layer:
        if l.type == 'Convolution':
            assert l.convolution_param.kernel_h == 0
            assert l.convolution_param.kernel_w == 0
            assert len(l.convolution_param.kernel_size) == 1
            if l.convolution_param.kernel_size[0] > 1:
                if skipped < kernel_active_skip:
                    skipped = skipped + 1
                    logging.info('skiping to update active kernel')
                else:
                    assert len(l.bottom) == 1
                    ks = l.convolution_param.kernel_size[0]
                    bottom = l.bottom[0]
                    if bottom not in bottom_map:
                        shift_layer = caffe.proto.caffe_pb2.LayerParameter()
                        shift_name = 'shift_{}'.format(bottom)
                        shift_layer.name = shift_name
                        shift_layer.bottom.append(bottom)
                        shift_layer.type = 'Shift'
                        shift_layer.top.append(shift_name)
                        sp = shift_layer.shift_param
                        sp.sparsity = kernel_active
                        sp.kernel_s = ks
                        if kernel_active_type != 'SEQ':
                            if kernel_active_type == 'SEQ_1x1':
                                sp.type = caffe.params.Shift.SEQ_1x1
                            elif kernel_active_type == 'UNIFORM_1x1':
                                sp.type = caffe.params.Shift.UNIFORM_1x1
                            else:
                                assert False
                        layers.append(shift_layer)
                        bottom_map[bottom] = shift_name
                    else:
                        shift_name = bottom_map[bottom]
                    l.ClearField('bottom')
                    l.bottom.append(shift_name)
                    assert len(l.convolution_param.pad) == 1
                    assert  l.convolution_param.pad[0] == \
                            l.convolution_param.kernel_size[0] / 2
                    l.convolution_param.ClearField('pad')
                    l.convolution_param.kernel_size[0] = 1
                    num_output = l.convolution_param.num_output
                    if l.convolution_param.group == num_output:
                        if shrink_group_if_group_e_out:
                            assert num_output / ks / ks > 0
                            l.convolution_param.group = num_output / (ks * ks)
                    else:
                        assert l.convolution_param.group == 1
                    c = c + 1
        layers.append(l)
    logging.info('update {} layers'.format(c))
    net.ClearField('layer')
    net.layer.extend(layers)

def update_crop_type(net, crop_type, inception_crop_kl=None):
    c = 0
    for l in net.layer:
        if l.type == 'TsvData':
            l.tsv_data_param.crop_type = crop_type
            if crop_type == caffe.params.TsvData.InceptionStyle:
                l.tsv_data_param.color_kl_file = inception_crop_kl
            c = c + 1
    logging.info('updated {} layers for crop type'.format(c))

def exist_bn(net, conv_top):
    for l in net.layer:
        if l.type == 'BatchNorm':
            assert len(l.bottom) == 1
            if l.bottom[0] == conv_top:
                return True
    return False

def update_bn(net):
    layers = []
    for l in net.layer:
        layers.append(l)
        if l.type == 'Convolution':
            assert len(l.top) == 1
            conv_top = l.top[0]
            if not exist_bn(net, conv_top):
                bn = caffe.proto.caffe_pb2.LayerParameter()
                bn.name = 'bn_{}'.format(conv_top)
                bn.bottom.append(conv_top)
                bn.type = 'BatchNorm'
                bn.top.append(conv_top)
                for i in range(3):
                    p = bn.param.add()
                    p.lr_mult = 0
                    p.decay_mult = 0
                layers.append(bn)
                scale = caffe.proto.caffe_pb2.LayerParameter()
                scale.name = 'scale_{}'.format(conv_top)
                scale.bottom.append(conv_top)
                scale.top.append(conv_top)
                scale.type = 'Scale'
                scale.scale_param.bias_term = True
                for i in range(2):
                    p = scale.param.add()
                    p.lr_mult = 1
                    p.decay_mult = 1 - i
                layers.append(scale)

    net.ClearField('layer')
    net.layer.extend(layers)
def load_binary_net(file_name):
    with open(file_name, 'r') as fp:
        c = fp.read()
    param = caffe.proto.caffe_pb2.NetParameter()
    param.ParseFromString(c)
    return param

def load_net_from_str(all_line):
    net_param = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(all_line, net_param)
    return net_param

def load_solver(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    solver_param = caffe.proto.caffe_pb2.SolverParameter()
    text_format.Merge(all_line, solver_param)
    return solver_param

def calculate_macc(prototxt):
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffe.TEST)
    net_proto = load_net(prototxt)
    macc = []
    ignore_layers = ['BatchNorm', 'Scale', 'ReLU', 'Softmax',
            'Pooling', 'Eltwise', 'Shift', 'Concat']
    for layer in net_proto.layer:
        if layer.type == 'Convolution':
            assert len(layer.bottom) == 1
            input_shape = net.blobs[layer.bottom[0]].data.shape
            assert len(layer.top) == 1
            output_shape = net.blobs[layer.top[0]].data.shape
            assert len(input_shape) == 4
            assert len(output_shape) == 4
            assert input_shape[0] == 1
            assert output_shape[0] == 1
            m = output_shape[3] * output_shape[1] * output_shape[2]
            assert layer.convolution_param.kernel_h == 0
            assert layer.convolution_param.kernel_w == 0
            assert len(layer.convolution_param.kernel_size) == 1
            m = m * layer.convolution_param.kernel_size[0] * \
                    layer.convolution_param.kernel_size[0]
            m = m * input_shape[1]
            m = m / layer.convolution_param.group
            macc.append((layer.name, m/1000000.))

        elif layer.type == 'InnerProduct':
            assert len(layer.bottom) == 1
            assert len(layer.top) == 1
            input_shape = net.blobs[layer.bottom[0]].data.shape
            output_shape = net.blobs[layer.top[0]].data.shape
            assert input_shape[0] == 1
            assert output_shape[0] == 1
            m = reduce(lambda x,y:x*y, input_shape)
            m = m * reduce(lambda x,y:x*y, output_shape)
            macc.append((layer.name, m/1000000.))
        #elif layer.type == 'Scale':
            #assert len(layer.bottom) == 1
            #input_shape = net.blobs[layer.bottom[0]].data.shape
            #m = reduce(lambda x,y:x*y, input_shape)
            #macc = macc + m
        else:
            assert layer.type in ignore_layers, layer.type
            pass

    return macc


def yolo_new_to_old(new_proto, new_model, old_model):
    assert op.isfile(new_proto)
    assert op.isfile(new_model)

    ensure_directory(op.dirname(old_model))

    # infer the number of anchors and the number of classes
    proto = load_net(new_proto)
    target_layers = [l for l in proto.layer if l.type == 'RegionTarget']
    if len(target_layers) == 1:
        target_layer = target_layers[0]
        biases_length = len(target_layer.region_target_param.biases)
    else:
        assert len(target_layers) == 0
        target_layers = [l for l in proto.layer if l.type == 'YoloBBs']
        assert len(target_layers) == 1
        target_layer = target_layers[0]
        biases_length = len(target_layer.yolobbs_param.biases)

    # if it is tree-based structure, we need to re-organize the classification
    # layout
    yolo_tree = any(l for l in proto.layer if 'SoftmaxTree' in l.type)

    num_anchor = biases_length / 2
    assert num_anchor * 2 == biases_length

    last_conv_layers = [l for l in proto.layer if l.name == 'last_conv']
    assert len(last_conv_layers) == 1
    last_conv_layer = last_conv_layers[0]
    num_classes = (last_conv_layer.convolution_param.num_output / num_anchor -
        5)
    assert last_conv_layer.type == 'Convolution'
    assert last_conv_layer.convolution_param.num_output == ((5 + num_classes) *
        num_anchor)
    net = caffe.Net(new_proto, new_model, caffe.TRAIN)
    target_param = net.params[last_conv_layer.name]
    new_weight = target_param[0].data
    old_weight = np.zeros_like(new_weight)
    has_bias = len(target_param) > 1
    all_new = [new_weight]
    all_old = [old_weight]
    if has_bias:
        new_bias = target_param[1].data
        old_bias = np.zeros_like(new_bias)
        all_new.append(new_bias[:, np.newaxis])
        all_old.append(old_bias[:, np.newaxis])

    for i in range(num_anchor):
        for new_p, old_p in zip(all_new, all_old):
            x = new_p[i + 0 * num_anchor, :]
            y = new_p[i + 1 * num_anchor, :]
            w = new_p[i + 2 * num_anchor, :]
            h = new_p[i + 3 * num_anchor, :]
            o = new_p[i + 4 * num_anchor, :]
            if not yolo_tree:
                new_cls_start = i * num_classes + 5 * num_anchor
                cls = new_p[new_cls_start : (new_cls_start + num_classes), :]
            else:
                new_cls_start = i + 5 * num_anchor
                cls = new_p[new_cls_start::num_anchor, :]

            old_p[0 + i * (5 + num_classes), :] = x
            old_p[1 + i * (5 + num_classes), :] = y
            old_p[2 + i * (5 + num_classes), :] = w
            old_p[3 + i * (5 + num_classes), :] = h
            old_p[4 + i * (5 + num_classes), :] = o
            old_cls_start = 5 + i * (5 + num_classes)
            old_p[old_cls_start: (old_cls_start + num_classes),
                    :] = cls

    for new_p, old_p in zip(all_new, all_old):
        new_p[...] = old_p

    net.save(old_model)

def yolo_old_to_new(old_proto, old_model, new_model):
    '''
    input: old_proto, old_model
    old_proto: train or test proto
    output: new_model
    '''
    assert op.isfile(old_proto)
    assert op.isfile(old_model)

    ensure_directory(op.dirname(new_model))

    # infer the number of anchors and the number of classes
    proto = load_net(old_proto)
    last_layer = proto.layer[-1]
    if last_layer.type == 'RegionLoss':
        num_classes = last_layer.region_loss_param.classes
        biases_length = len(last_layer.region_loss_param.biases)
    else:
        assert last_layer.type == 'RegionOutput'
        num_classes = last_layer.region_output_param.classes
        biases_length = len(last_layer.region_output_param.biases)
    num_anchor = biases_length / 2
    assert num_anchor * 2 == biases_length

    target_layer = proto.layer[-2]
    assert target_layer.type == 'Convolution'
    assert target_layer.convolution_param.num_output == ((5 + num_classes) *
        num_anchor)
    net = caffe.Net(old_proto, old_model, caffe.TRAIN)
    target_param = net.params[target_layer.name]
    old_weight = target_param[0].data
    new_weight = np.zeros_like(old_weight)
    has_bias = len(target_param) > 0
    all_old = [old_weight]
    all_new = [new_weight]
    if has_bias:
        old_bias = target_param[1].data
        new_bias = np.zeros_like(old_bias)
        all_old.append(old_bias[:, np.newaxis])
        all_new.append(new_bias[:, np.newaxis])

    for i in range(num_anchor):
        for old_p, new_p in zip(all_old, all_new):
            x = old_p[0 + i * (5 + num_classes), :]
            y = old_p[1 + i * (5 + num_classes), :]
            w = old_p[2 + i * (5 + num_classes), :]
            h = old_p[3 + i * (5 + num_classes), :]
            o = old_p[4 + i * (5 + num_classes), :]
            old_cls_start = 5 + i * (5 + num_classes)
            cls = old_p[old_cls_start: (old_cls_start + num_classes),
                    :]
            new_p[i + 0 * num_anchor, :] = x
            new_p[i + 1 * num_anchor, :] = y
            new_p[i + 2 * num_anchor, :] = w
            new_p[i + 3 * num_anchor, :] = h
            new_p[i + 4 * num_anchor, :] = o
            new_cls_start = i * num_classes + 5 * num_anchor
            new_p[new_cls_start : (new_cls_start + num_classes), :] = cls
    
    for old_p, new_p in zip(all_old, all_new):
        old_p[...] = new_p

    net.save(new_model)

def test_correct_caffe_file_path():
    solverstate = \
    './output/cifar100_vggstyle_0.1A_bn_16_1_1_1_1/snapshot/model_iter_1000.solverstate'
    out_file = solverstate + '_2'
    correct_caffemodel_file_path(solverstate, out_file)
    s1 = caffe.proto.caffe_pb2.SolverState()
    s2 = caffe.proto.caffe_pb2.SolverState()
    with open(solverstate, 'r') as fp:
        s1.ParseFromString(fp.read())
    with open(out_file, 'r') as fp:
        s2.ParseFromString(fp.read())
    assert s1.learned_net == s2.learned_net
    assert s1.iter == s2.iter
    assert s1.current_step == s2.current_step
    assert len(s1.history) == len(s2.history)
    for s1_h, s2_h in zip(s1.history, s2.history):
        assert len(s1_h.data) == len(s2_h.data)
        for s1_d, s2_d in zip(s1_h.data, s2_h.data):
            assert s1_d == s2_d

def update_yolo_test_proto(input_test, test_data, map_file, output_test):
    from .tsv_io import TSVDataset
    from .taxonomy import labels2noffsets, load_label_parent
    from .qd_common import write_to_file
    import os.path as op

    dataset = TSVDataset(test_data)
    if op.isfile(dataset.get_noffsets_file()):
        test_noffsets = dataset.load_noffsets()
    else:
        test_noffsets = labels2noffsets(dataset.load_labelmap())
    test_map_id = []
    net = load_net(input_test)
    for l in net.layer:
        if l.type == 'RegionOutput':
            tree_file = l.region_output_param.tree
            r = load_label_parent(tree_file)
            noffset_idx, noffset_parentidx, noffsets = r
            for noffset in test_noffsets:
                test_map_id.append(noffset_idx[noffset])
            write_to_file('\n'.join(map(str, test_map_id)), map_file)
            l.region_output_param.map = map_file
            l.region_output_param.thresh = 0.005
    assert len(test_noffsets) == len(test_map_id)
    write_to_file(str(net), output_test)


