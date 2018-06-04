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


def ncc2_train_with_covariance_and_mean(cov, means, nsample, avgnorm2):
    n_features = cov.shape[0]
    epsilon = calc_epsilon(nsample * 1.0 / n_features)
    cov = cov + epsilon * np.identity(n_features)
    w = linalg.solve(cov, means.T).T
    b = -0.5 * np.add.reduce(w * means, axis=1)
    return weight_normalize(w, b, avgnorm2)


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


def online_cov_calculate(previous_cov, mean, previous_n, x):
    delta_n = np.array(x - mean)
    delta_n = delta_n.reshape((delta_n.size, 1))
    n = previous_n + 1
    cov = (n - 2.0) / (n - 1.0) * previous_cov + np.dot(delta_n, np.transpose(delta_n)) / n

    return cov


def compute_covariance_by_group(new_net, anchor_num, lname, cid_groups, cid_hier_func, class_wise_mean_by_group,
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
    :param class_wise_mean_by_group: class-wise feature means of each group
    :type class_wise_mean_by_group: dict of dict
    :param tr_cnt: max number of trainign data to extract
    :param max_iters: maximum number of iterations to look for the required number of trainign data
    """
    feature_blob_name = new_net.bottom_names[lname][0]
    # feature_blob_shape = new_net.blobs[feature_blob_name].data.shape
    feature_outdim = new_net.params[lname][1].data.shape[0]
    class_num = feature_outdim // anchor_num - 5
    assert class_num == len(cid_groups), "class number: {} must match the groups len: {}".format(
        class_num, len(cid_groups))

    max_count = tr_cnt * class_num
    pbar = None
    if progressbar:
        widgets = ['Cov Extraction: ', progressbar.AnimatedMarker(),
                   ' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=max_count).start()

    wcnt = np.zeros(class_num, dtype=np.uint32)
    x_g = {}
    cov_by_group = {}
    cnt_g = {}
    iters = 0
    while iters < max_iters:
        count = np.sum(wcnt)
        if iters % 100 == 0:
            logging.info("Iterations: {} nodes: {}/{} total: {}/{}={}*{}".format(
                iters, np.sum(wcnt > 0), class_num, count, max_count, tr_cnt, class_num)
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
                if np.any(labels[i, (j * 5):(j * 5 + 5)] > 900000):
                    # already complained when finding the mean
                    continue
                assert cid < class_num, "Invalid label: {} >= class_num: {}".format(cid, class_num)

                bbox_x = int(labels[i, j * 5] * fw + 0.5)
                bbox_y = int(labels[i, j * 5 + 1] * fh + 0.5)
                features = feature_map[i, :, bbox_y, bbox_x]
                for c in cid_hier_func(cid):
                    # balanced data between classes
                    if wcnt[c] >= tr_cnt:
                        continue

                    g = cid_groups[c]
                    x = features - class_wise_mean_by_group[g][c]

                    if g not in cnt_g:
                        cnt_g[g] = 0
                        x_g[g] = x
                    elif cnt_g[g] == 1:
                        x_vec = np.array([x_g[g], x])
                        cov_by_group[g] = np.dot(x_vec.T, x_vec) / (x_vec.shape[0] - 1 + EPS)
                        x_g[g] = {}
                    else:
                        cov_by_group[g] = online_cov_calculate(cov_by_group[g], class_wise_mean_by_group[g][c],
                                                               cnt_g[g], x)

                    cnt_g[g] += 1
                    wcnt[c] += 1

        if np.min(wcnt) >= tr_cnt:
            break

    if iters >= max_iters:
        logging.error("Could not find enough data for the some classes after {} iterations".format(
            iters
        ))
    else:
        logging.info('using {}'.format(iters))

    return cov_by_group, cnt_g


def compute_class_wise_feature_mean_by_group(new_net, anchor_num, lname, cid_groups, cid_hier_func,
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
    :param tr_cnt: max number of trainign data to extract
    :param max_iters: maximum number of iterations to look for the required number of trainign data
    """
    feature_blob_name = new_net.bottom_names[lname][0]
    # feature_blob_shape = new_net.blobs[feature_blob_name].data.shape
    feature_outdim = new_net.params[lname][1].data.shape[0]
    class_num = feature_outdim // anchor_num - 5
    assert class_num == len(cid_groups), "class number: {} must match the groups len: {}".format(
        class_num, len(cid_groups))

    max_count = tr_cnt * class_num
    pbar = None
    if progressbar:
        widgets = ['Mean Extraction: ', progressbar.AnimatedMarker(),
                   ' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=max_count).start()

    wcnt = np.zeros(class_num, dtype=np.uint32)
    class_wise_feature_means_by_group = {}
    cnt_g = {}
    iters = 0
    while iters < max_iters:
        count = np.sum(wcnt)
        if iters % 100 == 0:
            logging.info("Iterations: {} nodes: {}/{} total: {}/{}={}*{}".format(
                iters, np.sum(wcnt > 0), class_num, count, max_count, tr_cnt, class_num)
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
                if np.any(labels[i, (j * 5):(j * 5 + 5)] > 900000):
                    # ignore invalidated bbox
                    logging.warning("ignore invalid bbox at iters: {} batch: {} box: {}".format(
                        iters, i, j
                    ))
                    continue
                assert cid < class_num, "Invalid label: {} >= class_num: {}".format(cid, class_num)

                bbox_x = int(labels[i, j * 5] * fw + 0.5)
                bbox_y = int(labels[i, j * 5 + 1] * fh + 0.5)
                x = feature_map[i, :, bbox_y, bbox_x]
                for c in cid_hier_func(cid):
                    # balanced data between classes
                    if wcnt[c] >= tr_cnt:
                        continue
                    g = cid_groups[c]

                    if g not in class_wise_feature_means_by_group:
                        class_wise_feature_means_by_group[g] = {}
                        cnt_g[g] = {}

                    if c not in class_wise_feature_means_by_group[g]:
                        class_wise_feature_means_by_group[g][c] = x
                        cnt_g[g][c] = 1
                    else:
                        cnt_g[g][c] += 1
                        class_wise_feature_means_by_group[g][c] = class_wise_feature_means_by_group[g][c] * (
                                (cnt_g[g][c] - 1.0) / (cnt_g[g][c])) + x / cnt_g[g][c]

                    wcnt[c] += 1
        if np.min(wcnt) >= tr_cnt:
            break
    if iters >= max_iters:
        logging.error("Could not find enough data for the some classes after {} iterations".format(
            iters
        ))
    else:
        logging.info('using {}'.format(iters))

    return class_wise_feature_means_by_group


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

def data_dependent_init2(pretrained_weights_filename,
        pretrained_prototxt_filename, new_prototxt_filename, new_weight,
        tr_cnt=100, max_iters=1000):

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = data_dependent_init(pretrained_weights_filename, 
            pretrained_prototxt_filename, 
            new_prototxt_filename,
            tr_cnt=tr_cnt, max_iters=max_iters)
    net.save(new_weight)

def data_dependent_init(pretrained_weights_filename, pretrained_prototxt_filename, new_prototxt_filename,
                        tr_cnt=100, max_iters=10000):
    """Rong's data-dependent init for Yolov2
    :param pretrained_weights_filename: path of the pre-trained .caffemodel file
    :param pretrained_prototxt_filename:  path of the test.prototxt used for the above weights
    :param new_prototxt_filename: path to the new train.prototxt to be initialized
    :param tr_cnt: max number of trainign data to extract
    :param max_iters: maximum number of iterations to look for the required number of trainign data
    """
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

    class_wise_feature_mean_by_group = \
        compute_class_wise_feature_mean_by_group(new_net, anchor_num, new_last_layer_name, cid_groups,
                                                 lift_hier(parents), tr_cnt=tr_cnt, max_iters=max_iters)

    # import pickle
    # with open("mean.npy", "wb") as fp:
    #     pickle.dump(class_wise_feature_mean_by_group, fp)

    covariance_by_group, cnt_g = \
        compute_covariance_by_group(new_net, anchor_num, new_last_layer_name, cid_groups,
                                    lift_hier(parents), class_wise_feature_mean_by_group,
                                    tr_cnt=tr_cnt, max_iters=max_iters)

    # with open("cov.npy", "wb") as fp:
    #     pickle.dump([covariance_by_group, cnt_g], fp)

    n_group = len(group_sizes)
    feature_blob_name = new_net.bottom_names[new_last_layer_name][0]
    feature_size = new_net.blobs[feature_blob_name].data.shape[1]

    # calculate the empirical norm of the yolo classification weights
    base_cw = conv_w[5:, :, :].reshape(-1, featuredim)
    base_avgnorm2 = np.average(np.add.reduce(base_cw * base_cw, axis=1))
    for g, offset, size in zip(range(n_group), group_offsets, group_sizes):
        if size == 1:
            w = 0
            b = 0
        else:
            feature_means = np.zeros((size, feature_size))
            c0 = offset
            for c in class_wise_feature_mean_by_group[g]:
                feature_means[c - c0, ] = class_wise_feature_mean_by_group[g][c]

            w, b = ncc2_train_with_covariance_and_mean(covariance_by_group[g], feature_means, cnt_g[g], base_avgnorm2)

        for i in range(anchor_num):
            new_w[5 + offset:5 + offset + size, i] = w
            new_b[5 + offset:5 + offset + size, i] = b

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
    parser.add_argument('--train', required=True, type=str, help='new train.prototxt path')
    parser.add_argument('--test', required=True, type=str, help='pretrained test.prototxt path')
    parser.add_argument('--weights', required=False, type=str, help='pretrained .caffemodel path')
    parser.add_argument('--tr_cnt', required=False, type=int,
                        help='training count for each node',
                        default=100)
    parser.add_argument('--max_iters', required=False, type=int,
                        help='Max number of iterations to look for training data for each run',
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

    init_net = data_dependent_init(pretrained_weights, pretrained_proto, new_proto, tr_cnt=args['tr_cnt'],
                                   max_iters=args['max_iters'])
    init_net.save(op.join(args['outputdir'], 'init.caffemodel'))
