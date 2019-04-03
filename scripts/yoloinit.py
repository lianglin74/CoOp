import sys, os, os.path as op
import time
import glob
import numpy as np
import argparse
import _init_paths
from shutil import copyfile
from google.protobuf import text_format
from numpy import linalg as LA
import logging
from tqdm import tqdm
try:
    from itertools import izip as zip
except:
    # python3
    pass
from pathos.multiprocessing import ProcessingPool as Pool
import pathos.multiprocessing as mp

EPS = np.finfo(float).eps

def data_dependent_init_ncc1(pretrained_weights_filename,
        pretrained_prototxt_filename, new_prototxt_filename, new_weight,
        tr_cnt=100, max_iters=1000):
    '''
    ncc version 1
    '''

    import caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = data_dependent_init_tree_ncc(pretrained_weights_filename,
            pretrained_prototxt_filename,
            new_prototxt_filename,
            tr_cnt=tr_cnt,
            max_iters=max_iters)

    net.save(new_weight)

def data_dependent_init_ncc2(pretrained_weights_filename,
        pretrained_prototxt_filename, new_prototxt_filename, new_weight,
        tr_cnt=100, max_iters=1000):

    import caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = data_dependent_init_tree_online(pretrained_weights_filename,
            pretrained_prototxt_filename,
            new_prototxt_filename,
            tr_cnt=tr_cnt,
            max_iters=max_iters)

    net.save(new_weight)

def extract_training_data_mean(new_net,anchor_num, lname,
        cid_groups, group_offsets, parents, group_sizes,
        tr_cnt, max_iter):
    feature_blob_name = new_net.bottom_names[lname][0]
    feature_dim = new_net.params[lname][0].data.shape[1]
    feature_outdim = new_net.params[lname][1].data.shape[0]
    class_num = feature_outdim//anchor_num -5
    wcnt = np.zeros(class_num, dtype=np.float32)
    logging.info('begin to collect the features')
    num_group = len(group_offsets)
    sum_count_n = []
    for g in range(num_group):
        # the first one is sum x_i * x_i^T
        # the second one is for sum x_i for each category
        # the third one is the count for each category on each group
        # the last one is the number of images within that group
        sum_count_n.append([np.zeros((group_sizes[g], feature_dim)),
            np.zeros((group_sizes[g], 1)),
            np.zeros(1)])
    for _ in range(max_iter):
        if (_ % 100) == 0:
            logging.info('{}/{}'.format(_, max_iter))
        new_net.forward(end=lname)
        feature_map = new_net.blobs[feature_blob_name].data.copy()
        fh = feature_map.shape[2]
        fw = feature_map.shape[3]
        labels = new_net.blobs['label'].data;
        batch_size = labels.shape[0];
        max_num_bboxes = labels.shape[1]/5;
        for i in range(batch_size):
            for j in range(max_num_bboxes):
                cid =int(labels[i, j*5+4]);
                if np.sum(labels[i,(j*5):(j*5+5)])==0:          #no more foreground objects
                    break;
                bbox_x = int(labels[i,j*5]*fw)
                bbox_y = int(labels[i,j*5+1]*fh)
                x = feature_map[i,:,bbox_y,bbox_x]
                x = x.reshape((-1, 1))
                y = cid
                while y >= 0:
                    wcnt[y]+=1;
                    curr_group_idx = cid_groups[y]
                    y_in_group = y - group_offsets[curr_group_idx]
                    sum_count_n[curr_group_idx][0][y_in_group] += \
                        x.reshape(len(x))
                    sum_count_n[curr_group_idx][1][y_in_group] += 1
                    sum_count_n[curr_group_idx][2] += 1
                    y = parents[y]
        if  np.min(wcnt) > tr_cnt:    break;

    logging.info('using {}'.format(_))
    means = []
    for sum_x_each, count_each, n in sum_count_n:
        # we need to add 1 to avoid 0
        for i in range(len(count_each)):
            count_each[i] = 1 if count_each[i] == 0 else count_each[i]
        mean_x_each = sum_x_each / count_each
        means.append((mean_x_each, n))
    return means

def extract_training_data_convmean(new_net,anchor_num, lname,
        cid_groups, group_offsets, parents, group_sizes,
        tr_cnt, max_iter):
    feature_blob_name = new_net.bottom_names[lname][0]
    feature_dim = new_net.params[lname][0].data.shape[1]
    feature_outdim = new_net.params[lname][1].data.shape[0]
    class_num = feature_outdim//anchor_num -5
    wcnt = np.zeros(class_num, dtype=np.float32)
    logging.info('begin to collect the features')
    num_group = len(group_offsets)
    conv_sum_count_n = []
    online = False
    if not online:
        xs_each_group = []
    for g in range(num_group):
        if not online:
            xs_each_group.append([])
        # the first one is sum x_i * x_i^T
        # the second one is for sum x_i for each category
        # the third one is the count for each category on each group
        # the last one is the number of images within that group
        conv_sum_count_n.append([np.zeros((feature_dim, feature_dim)),
            np.zeros((group_sizes[g], feature_dim)),
            np.zeros((group_sizes[g], 1)),
            np.zeros(1)])
    for _ in range(max_iter):
        if (_ % 100) == 0:
            logging.info('{}/{}'.format(_, max_iter))
        new_net.forward(end=lname)
        feature_map = new_net.blobs[feature_blob_name].data.copy()
        fh = feature_map.shape[2]
        fw = feature_map.shape[3]
        labels = new_net.blobs['label'].data;
        batch_size = labels.shape[0];
        max_num_bboxes = labels.shape[1]/5;
        for i in range(batch_size):
            for j in range(max_num_bboxes):
                cid =int(labels[i, j*5+4]);
                if np.sum(labels[i,(j*5):(j*5+5)])==0:          #no more foreground objects
                    break;
                bbox_x = int(labels[i,j*5]*fw)
                bbox_y = int(labels[i,j*5+1]*fh)
                x = feature_map[i,:,bbox_y,bbox_x]
                x = x.reshape((-1, 1))
                if online:
                    xxT = None
                y = cid
                while y >= 0:
                    if online:
                        if xxT is None:
                            xxT = np.dot(x, x.T)
                    wcnt[y]+=1;
                    curr_group_idx = cid_groups[y]
                    y_in_group = y - group_offsets[curr_group_idx]
                    if not online:
                        xs_each_group[curr_group_idx].append(x)
                    else:
                        conv_sum_count_n[curr_group_idx][0] += xxT
                    conv_sum_count_n[curr_group_idx][1][y_in_group] += \
                        x.reshape(len(x))
                    conv_sum_count_n[curr_group_idx][2][y_in_group] += 1
                    conv_sum_count_n[curr_group_idx][3] += 1
                    y = parents[y]
        if  np.min(wcnt) > tr_cnt:    break;

    if not online:
        for g, xs in enumerate(xs_each_group):
            if len(xs) == 0:
                continue
            else:
                xs_mat = np.hstack(xs)
            xsxsT = np.dot(xs_mat, xs_mat.T)
            conv_sum_count_n[g][0] = xsxsT
    logging.info('using {}'.format(_))
    conv_means = []
    for sum_xxt, sum_x_each, count_each, n in conv_sum_count_n:
        nmeanmeant = np.zeros((feature_dim, feature_dim))
        # we need to add 1 to avoid 0
        for i in range(len(count_each)):
            count_each[i] = 1 if count_each[i] == 0 else count_each[i]
        mean_x_each = sum_x_each / count_each
        for mean_x, c in zip(mean_x_each, count_each):
            mean_x = mean_x.reshape((-1, 1))
            nmeanmeant += c * np.dot(mean_x, mean_x.T)
        if n > 1:
            conv_x = (sum_xxt - nmeanmeant) / (n - 1)
        else:
            conv_x = np.zeros((feature_dim, feature_dim))
        conv_means.append((conv_x, mean_x_each, n))
    return conv_means

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

def get_softmax_tree_path(model):
    """Find the tree path in a Yolov2 model
    :param model: caffe prototxt model
    :rtype: str
    """
    n_layer = len(model.layer)
    result = None
    for i in reversed(range(n_layer)):
        layer = model.layer[i]
        tree_file = layer.softmaxtree_param.tree
        if tree_file:
            assert result is None
            result = tree_file
    assert result is not None
    return result


def number_of_anchor_boxex2(model):
    num_anchor = None
    for l in model.layer:
        if l.type == 'RegionTarget':
            assert num_anchor is None
            num_anchor = len(l.region_target_param.biases) / 2
        elif l.type == 'RegionLoss':
            assert num_anchor is None
            num_anchor = len(l.region_loss_param.biases) / 2
        elif l.type == 'RegionOutput':
            assert num_anchor is None
            num_anchor = len(l.region_output_param.biases) / 2
    return num_anchor

def ncc2_train_with_mean(means, nsample, avgnorm2):
    if nsample <= 1:
        return
    w = means
    b = np.zeros(means.shape[0])
    return weight_normalize(w, b, avgnorm2)

def ncc2_train_with_covariance_and_mean(cov, means, nsample, avgnorm2):
    if nsample <= 1:
        return
    n_features = cov.shape[0]
    epsilon = calc_epsilon(nsample * 1.0 / n_features)
    cov = cov + epsilon * np.identity(n_features)
    w = LA.solve(cov, means.T).T
    b = -0.5 * np.add.reduce(w * means, axis=1)
    return weight_normalize(w, b, avgnorm2)

def data_dependent_init_tree_ncc(pretrained_weights_filename,
        pretrained_prototxt_filename, new_prototxt_filename, tr_cnt=20,
        max_iters=1000):
    import caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()
    pretrained_net = caffe.Net(pretrained_prototxt_filename, pretrained_weights_filename, caffe.TEST)
    new_net = caffe.Net(new_prototxt_filename, caffe.TEST)
    new_net.copy_from(pretrained_weights_filename, ignore_shape_mismatch=True)

    model_from_pretrain_proto = read_model_proto(pretrained_prototxt_filename)
    model_from_new_proto = read_model_proto(new_prototxt_filename)
    pretrain_last_layer_name = last_linear_layer_name(model_from_pretrain_proto)        #last layer name
    new_last_layer_name = last_linear_layer_name(model_from_new_proto)    #last layername

    anchor_num = number_of_anchor_boxex2(model_from_new_proto)
    pretrain_anchor_num = number_of_anchor_boxex2(model_from_pretrain_proto)
    if anchor_num != pretrain_anchor_num:
        raise ValueError('The anchor numbers mismatch between the new model and the pretrained model (%s vs %s).' % (anchor_num, pretrain_anchor_num))
    print("# of anchors: %s" % anchor_num)
    #model surgery 1, copy bbox regression
    conv_w, conv_b = [p.data for p in pretrained_net.params[pretrain_last_layer_name]]
    featuredim = conv_w.shape[1]
    assert conv_w.shape[2] == 1 and conv_w.shape[3] == 1
    conv_w = conv_w.reshape(-1, anchor_num, featuredim)
    conv_b = conv_b.reshape(-1, anchor_num)

    new_w, new_b = [p.data for p in new_net.params[new_last_layer_name]];
    assert new_w.shape[2] == 1 and new_w.shape[3] == 1

    # save the param for the bounding box regression and objectiveness
    new_w = new_w.reshape(-1, anchor_num, featuredim)
    new_b = new_b.reshape(-1, anchor_num)
    new_w[:5, :, :] = conv_w[:5, :, :]
    new_b[:5, :] = conv_b[:5, :]

    tree_file = get_softmax_tree_path(model_from_new_proto)
    group_offsets, group_sizes, cid_groups, parents = read_softmax_tree(tree_file)
    assert len(cid_groups) == len(parents)
    start_time = time.time()
    means = extract_training_data_mean(new_net, anchor_num, new_last_layer_name,
            cid_groups, group_offsets, parents, group_sizes,
            tr_cnt=tr_cnt, max_iter=max_iters)
    logging.info('time cost to extract the training data: {}s'.format(
        time.time() - start_time))

    start_time = time.time()

    #calculate the empirical norm of the yolo classification weights
    base_cw= conv_w[5:, :, :]
    base_avgnorm2 = np.average(np.add.reduce(base_cw*base_cw,axis=1))

    logging.info('begin sequential')
    for i, (mean_x_each, c) in enumerate(means):
        if c <= 1:
            # single data have no conv
            logging.info('no training data')
            # no need to set the parameters
            continue
        else:
            W, B = ncc2_train_with_mean(mean_x_each, c,
                    base_avgnorm2)
            assert not np.isnan(np.mean(np.abs(W[:])))
            assert not np.isnan(np.mean(np.abs(B[:])))
        offset = group_offsets[i]
        size = group_sizes[i]
        for a in range(anchor_num):
            new_w[5 + offset:5 + offset + size, a] = W
            new_b[5 + offset:5 + offset + size, a] = B
    logging.info('end sequential')

    new_net.params[new_last_layer_name][0].data[...] = new_w.reshape(-1, featuredim, 1, 1)
    new_net.params[new_last_layer_name][1].data[...] = new_b.reshape(-1)
    logging.info('time cost to learn the parameter: {}s'.format(time.time() -
        start_time))

    return new_net

def data_dependent_init_tree_online(pretrained_weights_filename,
        pretrained_prototxt_filename, new_prototxt_filename, tr_cnt=20,
        max_iters=1000):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    pretrained_net = caffe.Net(pretrained_prototxt_filename, pretrained_weights_filename, caffe.TEST)
    new_net = caffe.Net(new_prototxt_filename, caffe.TEST)
    new_net.copy_from(pretrained_weights_filename, ignore_shape_mismatch=True)

    model_from_pretrain_proto = read_model_proto(pretrained_prototxt_filename)
    model_from_new_proto = read_model_proto(new_prototxt_filename)
    pretrain_last_layer_name = last_linear_layer_name(model_from_pretrain_proto)        #last layer name
    new_last_layer_name = last_linear_layer_name(model_from_new_proto)    #last layername

    anchor_num = number_of_anchor_boxex2(model_from_new_proto)
    pretrain_anchor_num = number_of_anchor_boxex2(model_from_pretrain_proto)
    if anchor_num != pretrain_anchor_num:
        raise ValueError('The anchor numbers mismatch between the new model and the pretrained model (%s vs %s).' % (anchor_num, pretrain_anchor_num))
    print("# of anchors: %s" % anchor_num)
    #model surgery 1, copy bbox regression
    conv_w, conv_b = [p.data for p in pretrained_net.params[pretrain_last_layer_name]]
    featuredim = conv_w.shape[1]
    assert conv_w.shape[2] == 1 and conv_w.shape[3] == 1
    conv_w = conv_w.reshape(-1, anchor_num, featuredim)
    conv_b = conv_b.reshape(-1, anchor_num)

    new_w, new_b = [p.data for p in new_net.params[new_last_layer_name]];
    assert new_w.shape[2] == 1 and new_w.shape[3] == 1

    # save the param for the bounding box regression and objectiveness
    new_w = new_w.reshape(-1, anchor_num, featuredim)
    new_b = new_b.reshape(-1, anchor_num)
    new_w[:5, :, :] = conv_w[:5, :, :]
    new_b[:5, :] = conv_b[:5, :]

    tree_file = get_softmax_tree_path(model_from_new_proto)
    group_offsets, group_sizes, cid_groups, parents = read_softmax_tree(tree_file)
    assert len(cid_groups) == len(parents)
    start_time = time.time()
    conv_means = extract_training_data_convmean(new_net, anchor_num, new_last_layer_name,
            cid_groups, group_offsets, parents, group_sizes,
            tr_cnt=tr_cnt, max_iter=max_iters)
    logging.info('time to extract data: {}s'.format(time.time() - start_time))
    start_time = time.time()
    #calculate the empirical norm of the yolo classification weights
    base_cw= conv_w[5:, :, :]
    base_avgnorm2 = np.average(np.add.reduce(base_cw*base_cw,axis=1))

    for i, (conv_x, mean_x_each, c) in enumerate(conv_means):
        if c <= 1:
            # single data have no conv
            logging.info('no training data')
            # no need to set the parameters
            continue
        else:
            W, B = ncc2_train_with_covariance_and_mean(conv_x, mean_x_each, c,
                    base_avgnorm2)
            assert not np.isnan(np.mean(np.abs(W[:])))
            assert not np.isnan(np.mean(np.abs(B[:])))
        offset = group_offsets[i]
        size = group_sizes[i]
        for a in range(anchor_num):
            new_w[5 + offset:5 + offset + size, a] = W
            new_b[5 + offset:5 + offset + size, a] = B
    logging.info('time for ncc2: {}s'.format(time.time() - start_time))

    new_net.params[new_last_layer_name][0].data[...] = new_w.reshape(-1, featuredim, 1, 1)
    new_net.params[new_last_layer_name][1].data[...] = new_b.reshape(-1)

    return new_net

def read_model_proto(proto_file_path):
    import caffe
    with open(proto_file_path) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        text_format.Parse(f.read(), model)
        return model

def last_linear_layer_name(model):
    n_layer = len(model.layer)
    for i in reversed(range(n_layer)):
        if model.layer[i].type=='InnerProduct' or model.layer[i].type=='Convolution' :
            return model.layer[i].name
    return None

def parse_labelfile_path(model):
    return model.layer[0].box_data_param.labelmap

def number_of_anchor_boxex(model):
    last_layer = model.layer[len(model.layer)-1]
    return max(len(last_layer.region_loss_param.biases), len(last_layer.region_output_param.biases))/2

def load_labelmap(label_file):
    with open(label_file) as f:
        cnames = [line.split('\t')[0].strip() for line in f]
    cnames.insert(0, '__background__')    # always index 0
    return dict(zip(cnames, xrange(len(cnames))))

def weight_normalize(W,B,avgnorm2):
    W -= np.average(W, axis=0)
    B -= np.average(B)
    W_normavg = np.average(np.add.reduce(W*W, axis=1)) + EPS
    alpha = np.sqrt(avgnorm2/W_normavg)
    return alpha*W, alpha*B

def calc_epsilon(dnratio):
    if dnratio>10: return 0.1
    elif dnratio<2:  return 10
    else: return 1;

def ncc2_train2(X,Y, avgnorm2, cmax=None):
    '''
    X: num_sample x num_feature
    '''
    if cmax is None:
        cmax = np.max(Y)+1;
    assert len(Y.shape) == 1
    epsilon = calc_epsilon(X.shape[0]/X.shape[1]);
    means = np.zeros((cmax, X.shape[1]), dtype=np.float32)
    for i in range(cmax):
        idxs = np.where(Y==i)[0]
        if len(idxs) > 1:
            means[i,:] = np.average(X[idxs,:], axis=0)
            X[idxs,:] -= means[i,:]
    cov = np.dot(X.T,X)/(X.shape[0]-1) + epsilon*np.identity(X.shape[1])
    W = LA.solve(cov,means.T).T
    B = -0.5*np.add.reduce(W*means,axis=1)
    return weight_normalize(W,B,avgnorm2)

def ncc2_train(X,Y, avgnorm2, cmax=None):
    '''
    X: num_sample x num_feature
    '''
    if cmax is None:
        cmax = np.max(Y)+1;
    assert len(Y.shape) == 1
    epsilon = calc_epsilon(X.shape[0]/X.shape[1]);
    means = np.zeros((cmax, X.shape[1]), dtype=np.float32)
    for i in range(cmax):
        idxs = np.where(Y==i)
        if len(idxs) > 1:
            means[i,:] = np.average(X[idxs,:], axis=1)
            X[idxs,:] -= means[i,:]
    cov = np.dot(X.T,X)/(X.shape[0]-1) + epsilon*np.identity(X.shape[1])
    W = LA.solve(cov,means.T).T
    B = -0.5*np.add.reduce(W*means,axis=1)
    return weight_normalize(W,B,avgnorm2)

def extract_training_data( new_net,anchor_num, lname, tr_cnt=200):
    feature_blob_name = new_net.bottom_names[lname][0]
    feature_blob_shape = new_net.blobs[feature_blob_name].data.shape
    feature_outdim = new_net.params[lname][1].data.shape[0]
    class_num = feature_outdim//anchor_num -5
    wcnt = np.zeros(class_num, dtype=np.float32)
    xlist =[]
    ylist = []
    while True:
        new_net.forward(end=lname)
        feature_map = new_net.blobs[feature_blob_name].data.copy()
        fh = feature_map.shape[2]-1
        fw = feature_map.shape[3]-1
        labels = new_net.blobs['label'].data;
        batch_size = labels.shape[0];
        max_num_bboxes = labels.shape[1]/5;
        for i in range(batch_size):
            for j in range(max_num_bboxes):
                cid =int(labels[i, j*5+4]);
                if np.sum(labels[i,(j*5):(j*5+5)])==0:          #no more foreground objects
                    break;
                bbox_x = int(labels[i,j*5]*fw+0.5)
                bbox_y = int(labels[i,j*5+1]*fh+0.5)
                xlist += [feature_map[i,:,bbox_y,bbox_x]]
                ylist += [cid]
                wcnt[cid]+=1;
        if  np.min(wcnt) > tr_cnt:    break;
    return np.vstack(xlist).astype(float), np.array(ylist).astype(int);


def data_dependent_init(pretrained_weights_filename, pretrained_prototxt_filename, new_prototxt_filename):
    import caffe
    pretrained_net = caffe.Net(pretrained_prototxt_filename, pretrained_weights_filename, caffe.TEST)
    new_net = caffe.Net(new_prototxt_filename, caffe.TRAIN)
    new_net.copy_from(pretrained_weights_filename, ignore_shape_mismatch=True)

    model_from_pretrain_proto = read_model_proto(pretrained_prototxt_filename)
    model_from_new_proto = read_model_proto(new_prototxt_filename)
    pretrain_last_layer_name = last_linear_layer_name(model_from_pretrain_proto)        #last layer name
    new_last_layer_name = last_linear_layer_name(model_from_new_proto)    #last layername

    anchor_num = number_of_anchor_boxex(model_from_new_proto)
    pretrain_anchor_num = number_of_anchor_boxex(model_from_pretrain_proto)
    if anchor_num != pretrain_anchor_num:
        raise ValueError('The anchor numbers mismatch between the new model and the pretrained model (%s vs %s).' % (anchor_num, pretrain_anchor_num))
    print("# of anchors: %s" % anchor_num)
    #model surgery 1, copy bbox regression
    conv_w, conv_b = [p.data for p in pretrained_net.params[pretrain_last_layer_name]]
    featuredim = conv_w.shape[1]
    conv_w = conv_w.reshape(anchor_num,-1,featuredim)
    conv_b = conv_b.reshape(anchor_num,-1)

    new_w, new_b = [p.data for p in new_net.params[new_last_layer_name]];

    new_w = new_w.reshape(anchor_num,-1,featuredim)
    new_b = new_b.reshape(anchor_num,-1)
    new_w[:,:5,:] = conv_w[:,:5,:]
    new_b[:,:5] = conv_b[:,:5]

    #data dependent model init
    class_to_ind = load_labelmap(parse_labelfile_path(model_from_new_proto))

    X,Y = extract_training_data(new_net,anchor_num, new_last_layer_name, tr_cnt=60  )
    #calculate the empirical norm of the yolo classification weights
    base_cw= conv_w[:,5:,:].reshape(-1,featuredim)
    base_avgnorm2 = np.average(np.add.reduce(base_cw*base_cw,axis=1))

    W,B = ncc2_train(X,Y,base_avgnorm2)

    for i in range(anchor_num):
        new_w[i,5:] = W
        new_b[i,5:] = B

    new_net.params[new_last_layer_name][0].data[...] = new_w.reshape(-1, featuredim, 1,1)
    new_net.params[new_last_layer_name][1].data[...] = new_b.reshape(-1)

    return new_net

def parse_args():
    parser = argparse.ArgumentParser(description='Initialzie a model')
    parser.add_argument('-g', '--gpuid',  help='GPU device id to be used.',  type=int, default='0')
    parser.add_argument('-n', '--net', required=True, type=str, help='CNN archiutecture')
    parser.add_argument('-j', '--jobfolder', required=True, type=str, help='job folder')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    import caffe
    caffe.set_device(args.gpuid)
    caffe.set_mode_gpu()

    pretrained_weights = op.join('models', args.net+".caffemodel")
    pretrained_proto = op.join('models', args.net+"_test.prototxt")

    new_proto = op.join(args.jobfolder, "train.prototxt");
    new_net = data_dependent_init(pretrained_weights, pretrained_proto, new_proto)
    new_net.save(op.join(args.jobfolder,'init.caffemodel'))
