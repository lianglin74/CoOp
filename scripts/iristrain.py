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
import time
import h5py

def saveh (hdf5_file, **kwargs):
    with h5py.File(hdf5_file,'w') as f:
        for key,value in kwargs.items():
            key = key.replace('/','\\')        
            ds = f.create_dataset(key, data=value)

def readh (hdf5_file):
    arrays = dict();
    with h5py.File(hdf5_file,'r') as f:
        for key in f:
            arrays[key.replace('\\','/')] = np.array(f[key])
    return arrays;

class Timer:
    def __init__(self,tag):
        self.tag=tag;
    def __enter__(self):
        self.start = time.clock()
        return self
    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print('%s used %.03f sec.'%(self.tag,self.interval));

def extract_training_data( new_net, feature_bname, label_bname, iters):
    xlist =[]
    ylist = []
    for i in range(iters):
        new_net.forward()
        xlist += [new_net.blobs[feature_bname].data.copy()]
        ylist += [new_net.blobs[label_bname].data.copy()]
    return np.vstack(xlist), np.array(ylist);

def parse_args():
    parser = argparse.ArgumentParser(description='Initialzie a model')
    parser.add_argument('-g', '--gpuid',  help='GPU device id to be used.',  type=int, default='0')
    parser.add_argument('-n', '--net', required=True, type=str.lower, help='CNN archiutecture')
    parser.add_argument('-j', '--jobpath', required=True, help='job path')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64, help='batch size, default=64')
    
    return parser.parse_args()

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

def weight_normalize(W,B,avgnorm2):
    W -= np.average(W, axis=0);
    B -=  np.average(B)
    W_normavg = np.average(np.add.reduce(W*W,axis=1));
    alpha = np.sqrt(avgnorm2/W_normavg)
    return alpha*W, alpha*B

def ncc2_train(X, Y, avgnorm2, epsilon=0.001):
    cmax = np.max(Y)+1;
    means = np.zeros((cmax, X.shape[1]), dtype=np.float32)
    for i in range(cmax):
        idxs = np.where(Y==i)
        means[i,:] = np.average(X[idxs,:], axis=1)
        X[idxs,:] -= means[i,:]
    cov = np.dot(X.T,X)/(X.shape[0]-1)
    W = LA.solve(cov+epsilon*np.identity(X.shape[1]),means.T).T
    #W = LA.solve(cov,means.T).T
    B = -0.5*np.add.reduce(W*means,axis=1)
    return weight_normalize(W,B,avgnorm2)

def extract_training_data( new_net,anchor_num, lname, trainX, trainY, datalayer, tr_cnt=200):
    feature_blob_name = new_net.bottom_names[lname][0]
    feature_blob_shape = new_net.blobs[feature_blob_name].data.shape
    feature_outdim = new_net.params[lname][1].data.shape[0]
    class_num = feature_outdim//anchor_num -5
    wcnt = np.zeros(class_num, dtype=np.float32)
    xlist =[]
    ylist = []
    idx = 0;
    while True:
        new_net.blobs[datalayer].data[...]=trainX[idx,:,:]
        labels = trainY[idx,:]
        new_net.blobs['label'].data[...]=labels
        idx = (idx+1)%trainX.shape[0];
        new_net.forward(end=lname)
        feature_map = new_net.blobs[feature_blob_name].data.copy()
        fh = feature_map.shape[2]-1
        fw = feature_map.shape[3]-1
        batch_size = labels.shape[0];
        for i in range(batch_size):
            for j in range(labels.shape[1]//5):
                if np.sum(labels[i,5*j:])==0:          #no more foreground objects
                    break;
                cid =int(labels[i,5*j+4]);                    
                bbox_x = int(labels[i,5*j]*fw+0.5)
                bbox_y = int(labels[i,5*j+1]*fh+0.5)
                xlist += [feature_map[i,:,bbox_y,bbox_x]]
                ylist += [cid]
                wcnt[cid]+=1;
        if  np.min(wcnt) > tr_cnt:    break;
    return np.vstack(xlist).astype(float), np.array(ylist).astype(int);


def data_dependent_init(pretrained_weights_filename, pretrained_prototxt_filename, new_prototxt_filename, X, Y, datalayer):
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

    trainX,trainY = extract_training_data(new_net,anchor_num, new_last_layer_name, X, Y,datalayer, tr_cnt=60  )
    #calculate the empirical norm of the yolo classification weights
    base_cw= conv_w[:,5:,:].reshape(-1,featuredim)
    base_avgnorm2 = np.average(np.add.reduce(base_cw*base_cw,axis=1))

    W,B = ncc2_train(trainX,trainY,base_avgnorm2)

    for i in range(anchor_num):
        new_w[i,5:] = W
        new_b[i,5:] = B

    new_net.params[new_last_layer_name][0].data[...] = new_w.reshape(-1, featuredim, 1,1)
    new_net.params[new_last_layer_name][1].data[...] = new_b.reshape(-1)

    return new_net

def savemodel( deploy_net, new_net, outputmodel):
    if not op.exists(outputmodel): new_net.save(outputmodel);
    deploy_net.copy_from(outputmodel)
    deploy_net.save(outputmodel)

def train(new_weight_init,deploy_net, solver_proto, X, Y,datalayer):
    solver = caffe.SGDSolver(solver_proto)
    solver.net.copy_from(new_weight_init)
    snapshot_path = op.split(new_weight_init)[0]
    while solver.iter< solver.param.max_iter:
        if solver.iter % solver.param.display == 0:
            model_path = snapshot_path + '/model_iter_%d.caffemodel'%solver.iter
            savemodel(deploy_net, solver.net, model_path )
        solver.net.blobs[datalayer].data[...] =  X[solver.iter%X.shape[0],:,:,:,:]
        solver.net.blobs['label'].data[...] =  Y[solver.iter%Y.shape[0],:,:]
        solver.step(1)
    model_path = snapshot_path + '/model_iter_%d.caffemodel'%solver.iter
    savemodel(deploy_net, solver.net, model_path )
    return model_path
    
if __name__ == "__main__":
    args = parse_args()

    caffe.set_device(args.gpuid)
    caffe.set_mode_gpu()

    pretrained_weights = args.net
    pretrained_proto =  op.splitext(args.net)[0]+"_test.prototxt"

    job_path = args.jobpath
    new_proto = op.join(job_path,'train.prototxt')
    solver_proto = op.join(job_path,'solver.prototxt')
    deploy_proto = op.join(job_path,'test.prototxt')
    new_weight_init = op.join(job_path,'snapshot/model_iter_0.caffemodel')
    feature_path = op.join(job_path,'features.h5')    
    deploy_net = caffe.Net(deploy_proto, caffe.TRAIN)
    deploy_net.copy_from(pretrained_weights, ignore_shape_mismatch=True)

    with Timer('loading data') as t:
        features = readh(feature_path)
    Y = features['label']
    for key,value in features.items():
        if key!='label':   datalayer,X = key,value;
    X = X.reshape(-1, args.batch_size, X.shape[1],X.shape[2],X.shape[3])
    Y = Y.reshape(-1, args.batch_size, Y.shape[1])
    new_net = data_dependent_init(pretrained_weights, pretrained_proto, new_proto, X,Y, datalayer)
    new_net.save(new_weight_init)
    new_net = train(new_weight_init,deploy_net, solver_proto, X, Y, datalayer)
