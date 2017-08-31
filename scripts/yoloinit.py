import sys, os, os.path as op
import time
import glob
import numpy as np
import argparse
import _init_paths
import caffe
from shutil import copyfile
import google.protobuf as pb
from numpy import linalg as LA

def read_model_proto(proto_file_path):
    with open(proto_file_path) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Parse(f.read(), model)
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
    conv_w = conv_w.reshape(anchor_num,-1,conv_w.shape[1])
    conv_b = conv_b.reshape(anchor_num,-1)

    new_w, new_b = [p.data for p in new_net.params[new_last_layer_name]];    
    new_w = new_w.reshape(anchor_num,-1,new_w.shape[1])
    new_b = new_b.reshape(anchor_num,-1)
    new_w[:,:5,:] = conv_w[:,:5,:]
    new_b[:,:5] = conv_b[:,:5]

    #data dependent model init   
    class_to_ind = load_labelmap(parse_labelfile_path(model_from_new_proto))
    
    feature_blob_name = new_net.bottom_names[new_last_layer_name][0]
    feature_blob_shape = new_net.blobs[feature_blob_name].data.shape
    featuredim = feature_blob_shape[1]

    new_class_num = new_w.shape[1] - 5
    wsum = np.zeros((new_class_num, featuredim), dtype=np.float32)
    wcnt = np.zeros((new_class_num), dtype=np.float32)
    
    while True:
        new_net.forward(end=new_last_layer_name)
        feature_map = new_net.blobs[feature_blob_name].data.copy()
        fh = feature_map.shape[2]-1    
        fw = feature_map.shape[3]-1
        labels = new_net.blobs['label'].data
        batch_size = labels.shape[0]
        for i in range(batch_size):
            cid =int(labels[i,4])
            if np.sum(labels[i,:])==0:          #no more foreground objects
                break
            bbox_x = int(labels[i,0]*fw+0.5)
            bbox_y = int(labels[i,1]*fh+0.5)
            wsum[cid,:] += feature_map[i,:,bbox_y,bbox_x]
            wcnt[cid]+=1
        min_cnt = np.min(wcnt)
        if  min_cnt > 20:
            break

    class_means = wsum/wcnt[:,None]
    norm_avg = np.average(LA.norm(conv_w[:,5:,:], axis=2))
    class_norms = LA.norm(class_means, axis=1)
    class_b = -0.5*(class_norms**2)
    class_normavg = np.average(class_norms)
    class_means *= norm_avg/class_normavg
    class_b *= norm_avg/class_normavg
    class_b -= np.average(class_b)
    
    for i in range(anchor_num):
        new_w[i,5:] = class_means
        new_b[i,5:] = class_b
        
    new_net.params[new_last_layer_name][0].data[...] = new_w.reshape(-1, featuredim, 1,1)
    new_net.params[new_last_layer_name][1].data[...] = new_b.reshape(-1)

    return new_net

def parse_args():
    parser = argparse.ArgumentParser(description='Initialzie a model')
    parser.add_argument('-g', '--gpuid',  help='GPU device id to be used.',  type=int, default='0')
    parser.add_argument('-n', '--net', required=True, type=str.lower, help='CNN archiutecture')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    caffe.set_device(args.gpuid)
    caffe.set_mode_gpu()

    pretrained_weights = op.join('models', args.net+".caffemodel")
    pretrained_proto = op.join('models', args.net+"_test.prototxt")

    new_proto = "yolo_voc20_train.prototxt"
    new_net = data_dependent_init(pretrained_weights, pretrained_proto, new_proto)
    new_net.save('models/darknet19_voc20b.caffemodel')
