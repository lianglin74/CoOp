import sys, os, os.path as op
import time
import glob
import numpy as np
import argparse
import _init_paths
import caffe
from shutil import copyfile
from google.protobuf import text_format
from numpy import linalg as LA
import logging

EPS = np.finfo(float).eps

def read_model_proto(proto_file_path):
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

def ncc2_train(X,Y, avgnorm2, cmax=None):
    if cmax is None:
        cmax = np.max(Y)+1;
    epsilon = calc_epsilon(X.shape[0]/X.shape[1]);
    means = np.zeros((cmax, X.shape[1]), dtype=np.float32)
    for i in range(cmax):
        idxs = np.where(Y==i)
        if len(idxs) > 0:
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

    caffe.set_device(args.gpuid)
    caffe.set_mode_gpu()

    pretrained_weights = op.join('models', args.net+".caffemodel")
    pretrained_proto = op.join('models', args.net+"_test.prototxt")

    new_proto = op.join(args.jobfolder, "train.prototxt");
    new_net = data_dependent_init(pretrained_weights, pretrained_proto, new_proto)
    new_net.save(op.join(args.jobfolder,'init.caffemodel'))
