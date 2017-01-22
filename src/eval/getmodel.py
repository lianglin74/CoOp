import sys,os;
import argparse;
import glob;
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser(description='Get py-faster-rcnn models')
    parser.add_argument('--data', required=True, help='the name of dataset')
    parser.add_argument('--basemodel', required=False, default="ZF", help='the name of the base model')
    parser.add_argument('--iter', required=False, type=int, default=0, help='the model@iteration, if iter==0, use the latest one')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args();
    
    basemodel = args.basemodel;
    dataname = args.data;
    iter = args.iter;
    
    prototxt_src = "models/%s/%s/faster_rcnn_end2end/test.prototxt"%(dataname,basemodel);
    labelmap_src = "data/%s/image.labelmap"%(dataname)
    model_path = "output/faster_rcnn_end2end/tsv_%s_train/%s_faster_rcnn_iter_"%(dataname,basemodel.lower());

    if iter>0:
        model_src = "%s%d.caffemodel"%(model_path,iter);
        assert os.path.exists(model_src), "%s is not exists!"%(model_src);
    else:
        model_pattern = model_path+"*.caffemodel";
        searchedfile = glob.glob(model_pattern)
        assert (len(searchedfile)>0), "0 file matched by %s!"%(model_pattern)
        files = sorted( searchedfile, key = lambda file: os.path.getctime(file));
        model_src = files[-1];
        iter = int(os.path.splitext(model_src)[0].split("_")[-1])
    
    base_dst = "%s_%s_%d"%(dataname,basemodel,iter);
    
    copyfile(model_src, base_dst+".caffemodel")
    copyfile(prototxt_src, base_dst+".prototxt")
    copyfile(labelmap_src, base_dst+".labelmap")
