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
#import tables as tb

#def saveh (hdf5_file, **kwargs):
#    with tb.open_file(hdf5_file,'w') as f:
#        for key,value in kwargs.items():
#            ds = f.create_carray(f.root, key, obj=value)
import h5py

def saveh (hdf5_file, **kwargs):
    with h5py.File(hdf5_file,'w') as f:
        for key,value in kwargs.items():
            key = key.replace('/','\\')
            ds = f.create_dataset(key, data=value)
            
def extract_training_data( new_net, feature_layers, iters):
    features = {lname:[] for lname in feature_layers}
    for i in range(iters):
        new_net.forward()
        for lname in feature_layers:
            features[lname] += [new_net.blobs[lname].data.copy()]
    features = {key:np.vstack(value) for key,value in features.items() }
    return features

def parse_args():
    parser = argparse.ArgumentParser(description='Initialzie a model')
    parser.add_argument('-g', '--gpuid',  help='GPU device id to be used.',  type=int, default='0')
    parser.add_argument('-n', '--net', required=True, type=str.lower, help='CNN archiutecture')
    parser.add_argument('-p', '--proto', required=True, help='network prototxt')
    parser.add_argument('-f', '--feature_layers', required=True, help='name of feature layers')
    parser.add_argument('-o', '--output', required=False, help='name of feature layer')
    parser.add_argument('-t', '--iters', type=int, required=True, help='Iterations required')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    caffe.set_device(args.gpuid)
    caffe.set_mode_gpu()

    pretrained_weights = args.net
    new_proto = args.proto
    new_net = caffe.Net(new_proto,pretrained_weights, caffe.TEST)
    feaout = args.output if args.output is not None else op.join(op.split(new_proto)[0],'features.h5');
    #new_proto = "dark19_voc20_train.prototxt"
    #X, Y = extract_training_data(new_net, 'res5c', 'label', 784 )  #5 epoches
    features = extract_training_data(new_net, args.feature_layers.split(','), args.iters )  #5 epoches = 784
    for k,v in features.items(): print(k,v.shape)
    saveh(feaout, **features)
