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

import os, os.path as op
from os.path import dirname as parent
import argparse
from datetime import datetime
import sys
import time
import glob

def at_fcnn(x):
    FRCN_ROOT = parent(parent(parent(op.realpath(__file__))))
    return op.realpath(op.join(FRCN_ROOT, x))
sys.path.insert(0, at_fcnn('tools'))   # add 'tools' dir to the path
import train_net, test_net

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', default='0', type=str, help='GPU id, or -1 for CPU')
    parser.add_argument('--NET', required=True, help='CNN archiutecture')
    parser.add_argument('--DATASET', required=True, help='either "pascal_voc" or "coco"')
    parser.add_argument('--ITERS', help='number of iter., default for pascal_voc = 70K, for coco = 490K')
    parser.add_argument('EXTRA_ARGS', nargs='*',  help='optional cfg for {train,test}_net.py (without "--")')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args();
    EXTRA_ARGS_SLUG = '_'.join(args.EXTRA_ARGS)
    if args.DATASET == 'pascal_voc':
        TRAIN_IMDB = "voc_2007_trainval"
        TEST_IMDB = "voc_2007_test"
        PT_DIR = "pascal_voc"
        ITERS = args.ITERS if args.ITERS is not None else 70000
    elif args.DATASET == 'coco':
        # This is a very long and slow training schedule
        # You can probably use fewer iterations and reduce the
        # time to the LR drop (set in the solver to 350,000 iterations).
        TRAIN_IMDB = "coco_2014_train"
        TEST_IMDB = "coco_2014_minival"
        PT_DIR = "coco"
        ITERS = args.ITERS if args.ITERS is not None else 490000
    else:
        TRAIN_IMDB = "tsv_%s_train"%args.DATASET;
        TEST_IMDB = "tsv_%s_test"%args.DATASET;
        PT_DIR = args.DATASET;
        ITERS = args.ITERS if args.ITERS is not None else 70000
    # redirect output to the LOG file
    DATE = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    LOG = at_fcnn('experiments/logs/faster_rcnn_end2end_%s_%s_%s.txt' %(args.NET, EXTRA_ARGS_SLUG, DATE))
    print 'Logging output to %s' % LOG
    with Tee(LOG,'w'):
        # run training
        start = time.time()
        train_net.main([
            '--gpu', args.GPU, 
            '--solver', at_fcnn('models/%s/%s/faster_rcnn_end2end/solver.prototxt' % (PT_DIR, args.NET)),
            '--weights', at_fcnn('data/imagenet_models/%s.v2.caffemodel' % args.NET),
            '--imdb', TRAIN_IMDB,
            '--iters', str(ITERS),
            '--cfg', at_fcnn('experiments/cfgs/faster_rcnn_end2end.yml')] +
            ([] if not args.EXTRA_ARGS else ['--set'] + args.EXTRA_ARGS))
        print 'tools/train_net.py finished in %s seconds' % (time.time() - start)
        
        model_pattern = "output/faster_rcnn_end2end/%s/%s_faster_rcnn_iter_*.caffemodel"%(TRAIN_IMDB,args.NET.lower());
        searchedfile = glob.glob(model_pattern)
        assert (len(searchedfile)>0), "0 file matched by %s!"%(model_pattern)
        files = sorted( searchedfile, key = lambda file: os.path.getctime(file));
        NET_FINAL = files[-1];

        # run testing
        start = time.time()
        test_net.main([
            '--gpu', args.GPU,
            '--def', at_fcnn('models/%s/%s/faster_rcnn_end2end/test.prototxt' % (PT_DIR, args.NET)),
            '--net', NET_FINAL,
            '--imdb', TEST_IMDB,
            '--cfg', at_fcnn('experiments/cfgs/faster_rcnn_end2end.yml')] + \
            ([] if not args.EXTRA_ARGS is not None else ['--set'] + args.EXTRA_ARGS))
        print 'tools/test_net.py finished in %s seconds' % (time.time() - start)
