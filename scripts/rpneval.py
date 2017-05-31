#!python3

import os, os.path as op;
import sys
import json
import argparse
import numpy as np;
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def  gen_truthslist(truths):
    truths_small = dict()
    truths_medium = dict()
    truths_large = dict()
    
    for (key,rects) in truths.items():
        crects_small = [];
        crects_medium = [];
        crects_large = [];
        for rect in rects:
            area = (rect[2]-rect[0])*(rect[3]-rect[1]);
            if area>32*32:
                if area>96*96:
                    crects_large +=[rect];
                else:
                    crects_medium +=[rect];
            else:
                crects_small +=[rect];
        if len(crects_small)> 0 : truths_small[key] = crects_small;
        if len(crects_medium)> 0 : truths_medium[key] = crects_medium;
        if len(crects_large)> 0 : truths_large[key] = crects_large;
    return [('small',truths_small), ('medium',truths_medium), ('large',truths_large), ('overall',truths)];  
    
#load ground-truth, organized by classes
def load_truths(filein):
    retdict = dict();
    with open(filein, "r") as tsvin:
        for line in tsvin:
            cols = [x.strip() for x in line.split("\t")]
            if len(cols)<2:
                continue;
            key = cols[0]
            rects = json.loads(cols[1]);
            retdict[key]=[];
            for rect in rects:
                bbox = [ x+1 for x in rect['rect'] ];
                retdict[key]+=[bbox];
    return gen_truthslist(retdict);

#load the detection results, organized by classes
def load_dets(filein):
    retlist = [];
    with open(filein, "r") as tsvin:
        for line in tsvin:
            cols = [x.strip() for x in line.split("\t")]
            if len(cols)<2:
                continue;
            key = cols[0]
            rects = [ rect['rect'] for rect in json.loads(cols[1])];
            retlist += [ (key,rects) ]
    return retlist;

def rect_area(rc):
    return (rc[2]-rc[0] + 1)*(rc[3]-rc[1] + 1);

#calculate the Jaccard similarity between two rectangles
def IoU(rc1, rc2):
    rc_inter =  [max(rc1[0],rc2[0]), max(rc1[1],rc2[1]),min(rc1[2],rc2[2]), min(rc1[3],rc2[3]) ]
    iw = rc_inter [2] - rc_inter [0] + 1;
    ih = rc_inter [3] - rc_inter [1] + 1;
    return (float(iw))*ih/(rect_area(rc1)+rect_area(rc2)-iw*ih) if (iw>0 and ih>0) else 0;

#evaluate the detection results
def eval(truths, detects, maxdet, ovthresh):
    #calculate npos
    sumarray = np.zeros(maxdet, dtype=np.float32);
    npos = sum([len(x[1]) for x in truths.items()]);
        
    for (key,rects) in detects:
        if key in truths :
            truthrects = truths[key];
            truth_cnt = len(truthrects);
            tags = np.zeros(truth_cnt);
            for j in range(len(rects)):
                overlaps = np.array([ IoU(rects[j],truthrects[i]) if tags[i]==0 else 0 for i in range(truth_cnt) ]);       # get overlaps with truth rectangles
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                if ovmax> ovthresh :
                    sumarray[j]+=1;
                    tags[jmax]=1;
    return np.cumsum(sumarray)/npos;

#split the file path into (directory, basename, ext)
def splitpath (filepath) :
    (dir,fname) = os.path.split(filepath);
    (basename,ext) = os.path.splitext(fname);
    return (dir,basename,ext);
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection RPN evaluation')
    parser.add_argument('--truths', required=True,   help='import groundtruth and baseline files')
    parser.add_argument('--dets', required=True,   help='import detection results')
    parser.add_argument('--name', default="", required=False,   help='the name of the experiment')
    parser.add_argument('--ovs', required=False, type=float, nargs='+', default=[0.5,0.4,0.3], help="IoU overlap thresholds")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # parse arguments
    args = parse_args();
    detsfile = args.dets
    #Load data
    truths_list = load_truths(args.truths);

    detresults = load_dets(detsfile);
    maxdet = max([len(x[1]) for x in detresults]);
    for truths in truths_list:
        fig =   plt.figure()    
        outputfig = '.'.join([op.splitext(detsfile)[0], truths[0],"png"]);
        for ovthresh in args.ovs:    
           coverage = eval(truths[1], detresults, maxdet,ovthresh);
           plt.plot(list(coverage), lw=2, label='%s IoU=%0.3g)' % (truths[0],ovthresh))
        plt.xlim([0.0, maxdet])
        plt.ylim([0.0, 1.05])
        plt.xlabel('# of RPN proposals')
        plt.ylabel('Coverage')
        plt.title('RPN Coverage Curve')
        plt.legend(loc="lower right")
        fig.savefig(outputfig,dpi=fig.dpi)       
