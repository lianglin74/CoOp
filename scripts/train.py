# Usage:
# python -u ./experiments/scripts/faster_rcnn_end2end.py ^
#   --GPU gpu --NET net --DATASET dataset [cfg args to {train,test}_net.py]
# where DATASET is either pascal_voc or coco.
#
# Example:
# python -u experiments/scripts/faster_rcnn_end2end.py ^
#   --GPU 0 --NET VGG_CNN_M_1024 --DATASET pascal_voc ^
#   EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"
#
# Notes:
#   1) the line-continuation symbol is ^ for cmd, use ` for powershell.
#   2) "-u" flag stands for unbuffered std output
import _init_paths
import sys, os, os.path as op
import time
import glob
import numpy as np
import argparse
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.tsv import tsv
import datasets.imdb
import caffe
from shutil import copyfile
from pprint import pprint
from tsvdet import tsvdet,setup_paths;
import deteval;
#import deteval_voc;
import gen_prototxt;

def at_fcnn(x):
    return op.realpath(op.join(FRCN_ROOT, x))

#tee python stdout        
class PyTee(object):
    def __init__(self, logstream, stream_name):
        valid_streams = ['stderr','stdout'];
        if  stream_name not in valid_streams:
            raise IOError("valid stream names are %s" % ', '.join(valid_streams))
        self.logstream =  logstream
        self.stream_name = stream_name;
    def __del__(self):
        pass;
    def write(self, data):  #tee stdout
        self.logstream.write(data);
        self.fstream.write(data);
        self.logstream.flush();
        self.fstream.flush();        
    def __enter__(self):
        if self.stream_name=='stdout' :
            self.fstream   =  sys.stdout
            sys.stdout = self;
        else:
            self.fstream   =  sys.stderr
            sys.stderr = self;
        self.fstream.flush();
    def __exit__(self, _type, _value, _traceback):
        if self.stream_name=='stdout' :
            sys.stdout = self.fstream;
        else:
            sys.stderr = self.fstream;

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('-g', '--gpu', dest='GPU_ID', help='GPU device id to use [0].',  default=0, type=int)
    parser.add_argument('-n', '--net', required=True, type=str.lower, help='CNN archiutecture')
    parser.add_argument('-ts', '--train_sizes', help='image target sizes and max size', default=[600,1000], nargs='+', required=False, type=int)
    parser.add_argument('-t', '--iters', dest='max_iters',  help='number of iterations to train', default=70000, required=True, type=int)
    parser.add_argument('-d', '--data', help='the name of the dataset', required=True)
    parser.add_argument('-e', '--expid', help='the experiment id', required=True)
    parser.add_argument(
        '-c', '--model_config', default="", type=str, required=False,
        help='model config path, default: models/faster_rcnn_end2end.yml')
    parser.add_argument('--precth', required=False, type=float, nargs='+', default=[0.8,0.9,0.95], help="get precision, recall, threshold above given precision threshold")
    parser.add_argument('--ovth', required=False, type=float, nargs='+', default=[0.3,0.4,0.5], help="get precision, recall, threshold above given precision threshold")
    parser.add_argument('-sg', '--skip_genprototxt', default=False, action='store_true', help='Flag to skip generating prototxt, default: False')

    return parser.parse_args()

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def latest_model(model_pattern):
    searchedfile = glob.glob(model_pattern)
    assert (len(searchedfile)>0), "0 file matched by %s!"%(model_pattern)
    files = sorted( searchedfile, key = lambda file: os.path.getmtime(file));
    return files[-1];

if __name__ == "__main__":
    args = parse_args()
    path_env = setup_paths( args.net, args.data, args.expid)
    # Get model train config copy it to the output path.
    cfg_config = args.model_config or path_env["cfg"]
    cfg_from_file(cfg_config)
    copyfile(cfg_config, os.path.join(path_env["output"], "config.yml"))
    cfg.GPU_ID = args.GPU_ID
    cfg.DATA_DIR = path_env['data_root']
    cfg.TRAIN.SCALES = args.train_sizes[:-1]
    cfg.TRAIN.MAX_SIZE = args.train_sizes[-1]

    # fix the random seeds (numpy and caffe) for reproducibility
    #np.random.seed(cfg.RNG_SEED)
    #caffe.set_random_seed(cfg.RNG_SEED)

    # redirect output to the LOG file
    print 'Logging output to %s' % path_env['log']
    start = time.time()
    with open(path_env['log'],'w') as pylog, PyTee(pylog,'stdout'):
        print 'Setting GPU device %d for training' % cfg.GPU_ID
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)
        caffe.init_glog(path_env['caffe_log']);

        imdb = tsv(args.data, 'train')
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)

        #generate training/testing prototxt
        if not args.skip_genprototxt :
            gen_prototxt.generate_prototxt(path_env['basemodel'].split('_')[0], imdb.num_images, imdb.num_classes, path_env['output']);

        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)   #imdb.gt_roidb()
        
        print '{:d} roidb entries'.format(len(roidb))
        train_net(path_env['solver'], roidb, path_env['snapshot'], pretrained_model=path_env['basemodel'], max_iters=args.max_iters)
        print 'training finished in %s seconds' % (time.time() - start)
        
        #get final models
        labelmap_src = op.join(path_env['data'],"labelmap.txt")    
        net_final = latest_model( path_env['model_pattern'] )
        
        labelmap_src = op.join(path_env['data'],"labelmap.txt")
        proto_src =  op.join (path_env['output'],"test.prototxt")
        model_dst = op.join(path_env['deploy'],op.basename(net_final));
        proto_dst = op.splitext(model_dst)[0]+".prototxt";
        labelmap_dst = op.splitext(model_dst)[0]+".labelmap";
        
        copyfile(net_final,model_dst);
        copyfile(proto_src, proto_dst);
        copyfile(labelmap_src, labelmap_dst)
        
        #do evaluation when test data is available
        intsv_file = op.join(path_env['data'], "test.tsv");
        outtsv_file = path_env['eval'];
        assert op.isfile(intsv_file), "test file %s not find" % intsv_file  #this is a test file, do det and evaluation
        start = time.time()            
        nimgs = tsvdet(model_dst, intsv_file, 0,2,outtsv_file, proto = proto_src, cmap = labelmap_src);
        time_used = time.time() - start
        print ( 'detect %d images, used %g s (avg: %g s)' % (nimgs,time_used, time_used/nimgs ) )  
        reports = deteval.get_report(deteval.load_truths(intsv_file), deteval.load_dets(outtsv_file), args.ovth, True)
        deteval.print_reports(reports, args.precth)  
