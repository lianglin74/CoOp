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
from datetime import datetime
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

def at_fcnn(x):
    return op.realpath(op.join(FRCN_ROOT, x))

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
    def __enter__(self):
        pass
    def __exit__(self, _type, _value, _traceback):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='GPU_ID', help='GPU device id to use [0].',  default=0, type=int)
    parser.add_argument('--net', required=True, help='CNN archiutecture')    
    parser.add_argument('--iters', dest='max_iters',  help='number of iterations to train', default=70000,    required=True, type=int)
    parser.add_argument('--data', help='the name of the dataset', required=True);
    #parser.add_argument('--cfg', dest='cfg_file',  help='optional config file',    default=None, type=str)
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

def createpath( pparts ):
    fpath = op.join(*pparts);
    if not os.path.exists(fpath):
        os.makedirs(fpath);
    return fpath;   
if __name__ == "__main__":
    args = parse_args()
    
    
    proj_root = op.dirname(op.dirname(op.realpath(__file__)));
    solver_file = op.join (proj_root,"models",args.net,args.data,"solver.prototxt");
    basemodel_file =  op.join (proj_root,"models",args.net, '%s.caffemodel'%args.net);
    default_cfg = op.join(proj_root,"models","faster_rcnn_end2end.yml")
    data_path = op.join(proj_root,"data");
    snapshot_path = createpath([proj_root,"output",args.data,'snapshot']);
    DATE = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = op.join(proj_root,"output",args.data,'%s_%s.log' %(args.net, DATE));
    
    cfg_from_file(default_cfg)
    cfg.GPU_ID = args.GPU_ID;
    cfg.DATA_DIR = data_path;
    #print('Using config:')
    #pprint.pprint(cfg)

    # fix the random seeds (numpy and caffe) for reproducibility
    #np.random.seed(cfg.RNG_SEED)
    #caffe.set_random_seed(cfg.RNG_SEED)

    # redirect output to the LOG file
    print 'Logging output to %s' % log_file
    start = time.time()
    with Tee(log_file,'w'):
        print 'Setting GPU device %d for training' % cfg.GPU_ID
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)

        imdb = tsv(args.data, 'train')
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        #imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        #print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)   #imdb.gt_roidb()
        
        print '{:d} roidb entries'.format(len(roidb))

        train_net(solver_file, roidb, snapshot_path,
                  pretrained_model=basemodel_file,
                  max_iters=args.max_iters)
        print 'training finished in %s seconds' % (time.time() - start)

        model_pattern = "%s/%s_faster_rcnn_iter_*.caffemodel"%(snapshot_path,args.net.lower());
        searchedfile = glob.glob(model_pattern)
        assert (len(searchedfile)>0), "0 file matched by %s!"%(model_pattern)
        files = sorted( searchedfile, key = lambda file: os.path.getctime(file));
        net_final = files[-1];
        labelmap_src = op.join(data_path,args.data,"labelmap.txt")
        proto_src =  op.join (proj_root,"models",args.net,args.data,"test.prototxt")
        model_dst = op.join(snapshot_path,"..",op.basename(net_final));
        proto_dst = op.splitext(model_dst)[0]+".prototxt";
        labelmap_dst = op.splitext(model_dst)[0]+".labelmap";
        
        copyfile(net_final,model_dst);
        copyfile(proto_src, proto_dst);
        copyfile(labelmap_src, labelmap_dst)
