import os.path as op
import numpy as np
import argparse
import logging
from numpy import linalg
try:
    import progressbar
except ImportError:
    progressbar = None
# noinspection PyUnresolvedReferences
import _init_paths
import caffe
from qd_common import init_logging
from yoloinit import last_linear_layer_name, read_model_proto, calc_epsilon, weight_normalize, EPS


# noinspection PyPep8Naming
def ncc2_train(X, Y, cmin, cmax, avgnorm2):
    if X.shape[0] < 2:
        logging.error("There are very few samples for nodes: {}->{}".format(cmin, cmax - 1))
    epsilon = calc_epsilon(X.shape[0]/X.shape[1])
    means = np.zeros((cmax - cmin, X.shape[1]), dtype=np.float32)
    for i, y in enumerate(range(cmin, cmax)):
        idxs = np.where(Y == y)
        if len(idxs[0]) == 0:
            logging.error("Ignore class: {} with no data".format(y))
            idxs = np.where(Y != y)
        means[i, :] = np.average(X[idxs, :], axis=1)
        X[idxs, :] -= means[i, :]
    cov = np.dot(X.T, X)/(X.shape[0] - 1 + EPS) + epsilon * np.identity(X.shape[1])
    W = linalg.solve(cov, means.T).T
    B = -0.5*np.add.reduce(W*means, axis=1)
    return weight_normalize(W, B, avgnorm2)


def softmax_tree_path(model):
    """Find the tree path in a Yolov2 model
    :param model: caffe prototxt model
    :rtype: str
    """
    n_layer = len(model.layer)
    for i in reversed(range(n_layer)):
        layer = model.layer[i]
        tree_file = layer.softmaxtree_param.tree
        if tree_file:
            return tree_file
    raise Exception('Could not find the tree file')


def number_of_anchor_boxes(model):
    """Find the number of anchors in a Yolov2 model
    :param model: caffe prototxt model
    :rtype: int
    """
    n_layer = len(model.layer)
    for i in reversed(range(n_layer)):
        layer = model.layer[i]
        if layer.type == 'RegionTarget':
            return len(layer.region_target_param.biases) / 2
        if layer.type == 'YoloBBs':
            return len(layer.yolobbs_param.biases) / 2
    raise Exception('Could not find the anchor number')


def extract_training_data(new_net, anchor_num, lname, cid_groups, cid_hier_func,
                          chunk_groups=None, total_chunk_size=0,
                          tr_cnt=200, max_iters=10000):
    """Extract training data
    :param new_net: training network to extract data from
    :type new_net: caffe.Net
    :param anchor_num: number of anchors
    :type anchor_num: int
    :param lname: name of the linear layer
    :param cid_groups: list of groups for each node
    :type cid_groups: list
    :param cid_hier_func: iterator for parents hierarchy
    :param chunk_groups: groups to extract for
    :param total_chunk_size: size of the group chunk under extraction
    :param tr_cnt: max number of trainign data to extract
    :param max_iters: maximum number of iterations to look for the required number of trainign data
    """
    feature_blob_name = new_net.bottom_names[lname][0]
    # feature_blob_shape = new_net.blobs[feature_blob_name].data.shape
    feature_outdim = new_net.params[lname][1].data.shape[0]
    class_num = feature_outdim // anchor_num - 5
    assert class_num == len(cid_groups), "class number: {} must match the groups len: {}".format(
        class_num, len(cid_groups))

    chunked = False
    if chunk_groups:
        assert total_chunk_size > 0
        logging.info("Extracting groups {}->{} with {} total nodes".format(
            chunk_groups[0], chunk_groups[-1], total_chunk_size))
        chunked = True
    else:
        total_chunk_size = class_num
        chunk_groups = cid_groups

    max_count = tr_cnt * total_chunk_size
    pbar = None
    if progressbar:
        widgets = ['Extraction: ', progressbar.AnimatedMarker(),
                   ' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=max_count).start()

    wcnt = np.zeros(class_num, dtype=np.uint32)
    x_g = {}
    y_g = {}
    iters = 0
    while iters < max_iters:
        count = np.sum(wcnt)
        if iters % 100 == 0:
            logging.info("Iterations: {} nodes: {}/{} total: {}/{}={}*{}".format(
                iters, np.sum(wcnt > 0), total_chunk_size, count, max_count, tr_cnt, total_chunk_size)
            )
        if pbar:
            pbar.update(count)
        iters += 1
        new_net.forward(end=lname)
        feature_map = new_net.blobs[feature_blob_name].data.copy()
        fh = feature_map.shape[2] - 1
        fw = feature_map.shape[3] - 1
        labels = new_net.blobs['label'].data
        batch_size = labels.shape[0]
        max_num_bboxes = labels.shape[1] / 5
        for i in range(batch_size):
            for j in range(max_num_bboxes):
                cid = int(labels[i, j * 5 + 4])
                if np.sum(labels[i, (j * 5):(j * 5 + 5)]) == 0:  # no more foreground objects
                    break
                assert cid < class_num, "Invalid label: {} >= class_num: {}".format(cid, class_num)
                bbox_x = int(labels[i, j * 5] * fw + 0.5)
                bbox_y = int(labels[i, j * 5 + 1] * fh + 0.5)
                for c in cid_hier_func(cid):
                    # balanced data between classes
                    if wcnt[c] >= tr_cnt:
                        continue
                    g = cid_groups[c]
                    if chunked and g not in chunk_groups:
                        continue
                    if g not in x_g:
                        x_g[g] = []
                        y_g[g] = []
                    xlist = x_g[g]
                    ylist = y_g[g]
                    xlist += [feature_map[i, :, bbox_y, bbox_x]]
                    ylist += [c]
                    wcnt[c] += 1
        if np.min(wcnt) >= tr_cnt:
            break
    if iters >= max_iters:
        logging.error("Could not find enough data for the some classes after {} iterations".format(
            iters
        ))
    for g in chunk_groups:
        if g not in x_g:
            logging.error("No data found for group: {}".format(g))
            continue
        x_g[g] = np.vstack(x_g[g]).astype(float)
        y_g[g] = np.array(y_g[g]).astype(int)
    return x_g, y_g


def lift_hier(parents):
    """lift each node to include the hiearchy
    :param parents: parent of each node
    """
    def _hier(n):
        yield n
        p = parents[n]
        while p >= 0:
            yield p
            p = parents[p]
    return _hier


def read_softmax_tree(tree_file):
    """Simple parsing of softmax tree with subgroups
    :param tree_file: path to the tree file
    :type tree_file: str
    """
    group_offsets = []
    group_sizes = []
    cid_groups = []
    parents = []
    last_p = -1
    last_sg = -1
    groups = 0
    size = 0
    n = 0
    with open(tree_file, 'r') as f:
        for line in f.readlines():
            tokens = [t for t in line.split(' ') if t]
            assert len(tokens) == 2 or len(tokens) == 3, "invalid tree: {} node: {} line: {}".format(
                tree_file, n, line)
            p = int(tokens[1])
            parents.append(p)
            sg = -1
            if len(tokens) == 3:
                sg = int(tokens[2])
            new_group = new_sub_group = False
            if p != last_p:
                last_p = p
                last_sg = sg
                new_group = True
            elif sg != last_sg:
                last_sg = sg
                new_sub_group = True
            if new_group or new_sub_group:
                group_sizes.append(size)
                group_offsets.append(n - size)
                groups += 1
                size = 0
            n += 1
            size += 1
            cid_groups.append(groups)
    group_sizes.append(size)
    group_offsets.append(n - size)

    assert len(cid_groups) == len(parents)
    assert len(group_offsets) == len(group_sizes) == max(cid_groups) + 1
    return group_offsets, group_sizes, cid_groups, parents


def data_dependent_init(pretrained_weights_filename, pretrained_prototxt_filename, new_prototxt_filename,
                        node_chunk=100, tr_cnt=100, max_iters=10000):
    """Rong's data-dependent init for Yolov2
    :param pretrained_weights_filename: path of the pre-trained .caffemodel file
    :param pretrained_prototxt_filename:  path of the test.prototxt used for the above weights
    :param new_prototxt_filename: path to the new train.prototxt to be initialized
    :param node_chunk: how many nodes to train at a time (smaller chunks is slower but needs less memory)
    :param tr_cnt: max number of trainign data to extract
    :param max_iters: maximum number of iterations to look for the required number of trainign data
    """
    if node_chunk < 0:
        node_chunk = np.inf
    assert node_chunk > 0, "Invalid chunk size: {}".format(node_chunk)
    pretrained_net = caffe.Net(pretrained_prototxt_filename, pretrained_weights_filename, caffe.TEST)
    new_net = caffe.Net(new_prototxt_filename, caffe.TRAIN)
    new_net.copy_from(pretrained_weights_filename, ignore_shape_mismatch=True)

    model_from_pretrain_proto = read_model_proto(pretrained_prototxt_filename)
    model_from_new_proto = read_model_proto(new_prototxt_filename)
    pretrain_last_layer_name = last_linear_layer_name(model_from_pretrain_proto)  # last layer name
    new_last_layer_name = last_linear_layer_name(model_from_new_proto)  # last layername

    anchor_num = number_of_anchor_boxes(model_from_new_proto)
    pretrain_anchor_num = number_of_anchor_boxes(model_from_pretrain_proto)
    if anchor_num != pretrain_anchor_num:
        raise ValueError('The anchor numbers mismatch between the new model and the pretrained model (%s vs %s).' % (
            anchor_num, pretrain_anchor_num))
    logging.info("# of anchors: %s" % anchor_num)
    # model surgery 1, copy bbox regression
    conv_w, conv_b = [p.data for p in pretrained_net.params[pretrain_last_layer_name]]
    featuredim = conv_w.shape[1]
    conv_w = conv_w.reshape(-1, anchor_num, featuredim)
    conv_b = conv_b.reshape(-1, anchor_num)

    new_w, new_b = [p.data for p in new_net.params[new_last_layer_name]]

    new_w = new_w.reshape(-1, anchor_num, featuredim)
    new_b = new_b.reshape(-1, anchor_num)
    new_w[:5, :, :] = conv_w[:5, :, :]
    new_b[:5, :] = conv_b[:5, :]

    tree_file = softmax_tree_path(model_from_new_proto)
    group_offsets, group_sizes, cid_groups, parents = read_softmax_tree(tree_file)

    pbar = None
    if progressbar:
        widgets = ['Training: ', progressbar.AnimatedMarker(),
                   ' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(parents)).start()

    sliced_groups = [[]]
    for group in range(len(group_sizes)):
        last_chunk = sliced_groups[-1]
        chunk_size = np.sum([group_sizes[g] for g in last_chunk])
        if chunk_size < node_chunk:
            last_chunk.append(group)
        else:
            sliced_groups.append([group])

    for chunk_groups in sliced_groups:
        chunk_group_offsets = [group_offsets[idx] for idx in chunk_groups]
        chunk_group_sizes = [group_sizes[idx] for idx in chunk_groups]

        total_chunk_size = int(np.sum(chunk_group_sizes))
        # noinspection PyPep8Naming
        X, Y = extract_training_data(new_net, anchor_num, new_last_layer_name, cid_groups, lift_hier(parents),
                                     chunk_groups=chunk_groups, total_chunk_size=total_chunk_size,
                                     tr_cnt=tr_cnt, max_iters=max_iters)

        for g, offset, size in zip(chunk_groups, chunk_group_offsets, chunk_group_sizes):
            if g not in X:
                logging.error("Ignore group {} with no training data".format(g))
                continue
            logging.info("Training softmax group {}".format(g))

            if size == 1:
                w = 0
                b = 0
            else:
                # calculate the empirical norm of the yolo classification weights
                base_cw = conv_w[5 + offset:5 + offset + size, :, :].reshape(-1, featuredim)
                base_avgnorm2 = np.average(np.add.reduce(base_cw * base_cw, axis=1))
                w, b = ncc2_train(X[g], Y[g], offset, offset + size, base_avgnorm2)

            for i in range(anchor_num):
                new_w[5+offset:5+offset+size, i] = w
                new_b[5+offset:5+offset+size, i] = b

            if pbar:
                pbar.update(offset + size)

    new_net.params[new_last_layer_name][0].data[...] = new_w.reshape(-1, featuredim, 1, 1)
    new_net.params[new_last_layer_name][1].data[...] = new_b.reshape(-1)

    return new_net


def parse_args():
    parser = argparse.ArgumentParser(description='Initialzie a model')
    parser.add_argument('-g', '--gpuid', help='GPU device id to be used.', type=int, default='0')
    parser.add_argument('-o', '--outputdir', help='Output directory', default='.',
                        required=False)
    parser.add_argument('--train', required=True, type=str, help='train.prototxt path')
    parser.add_argument('--test', required=True, type=str, help='test.prototxt path')
    parser.add_argument('--weights', required=False, type=str, help='initial .caffemodel path (optional)')
    parser.add_argument('--chunk', required=False, type=int,
                        help=('approximate node chunk size for extraction.'
                              'Larger chunks are faster but need more memory.'
                              'Use -1 for a single pass (fastest)'),
                        default=-1)
    parser.add_argument('--tr_cnt', required=False, type=int,
                        help='training count for each node',
                        default=100)
    parser.add_argument('--max_iters', required=False, type=int,
                        help='Max number of iterations to look for training data for each chunk',
                        default=10000)

    return vars(parser.parse_args())


if __name__ == "__main__":
    init_logging()
    args = parse_args()

    caffe.set_device(args['gpuid'])
    caffe.set_mode_gpu()
    caffe.init_glog(args['outputdir'])

    pretrained_weights = args['weights']
    pretrained_proto = args['test']
    new_proto = args['train']

    init_net = data_dependent_init(pretrained_weights, pretrained_proto, new_proto,
                                   node_chunk=args['chunk'], tr_cnt=args['tr_cnt'], max_iters=args['max_iters'])
    init_net.save(op.join(args['outputdir'], 'init.caffemodel'))
