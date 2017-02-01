#!python3

import os
import sys
import json
import argparse
import numpy as np;
from sklearn import metrics;
import matplotlib.pyplot as plt
import glob;

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
            for rect in rects:
                label = rect['class'];
                if label not in retdict:
                    retdict[label]=dict();
                if key not in retdict[label]:
                    retdict[label][key]=[];
                bbox = [ x+1 for x in rect['rect'] ];
                retdict[label][key]+=[(rect['diff'],bbox)];
    return retdict;

#load the detection results, organized by classes
def load_dets(filein,truths):
    retdict = { key:[] for key in truths}
    with open(filein, "r") as tsvin:
        for line in tsvin:
            cols = [x.strip() for x in line.split("\t")]
            if len(cols)<2:
                continue;
            key = cols[0]
            rects = json.loads(cols[1]);
            for rect in rects:
                label = rect['class'];
                bbox =  rect['rect'];
                retdict[label] += [ (key,rect['conf'],bbox)]
    for key in retdict:
        retdict[key] = sorted(retdict[key], key=lambda x:-x[1])
    return retdict;

def rect_area(rc):
    return (rc[2]-rc[0] + 1)*(rc[3]-rc[1] + 1);

#calculate the Jaccard similarity between two rectangles
def IoU(rc1, rc2):
    rc_inter =  [max(rc1[0],rc2[0]), max(rc1[1],rc2[1]),min(rc1[2],rc2[2]), min(rc1[3],rc2[3]) ]
    iw = rc_inter [2] - rc_inter [0] + 1;
    ih = rc_inter [3] - rc_inter [1] + 1;
    return (float(iw))*ih/(rect_area(rc1)+rect_area(rc2)-iw*ih) if (iw>0 and ih>0) else 0;

#evaluate the detection results
def evaluate (c_detects, c_truths):
    #calculate npos
    npos = 0;
    for key in c_truths:
        npos += len([x for x in c_truths[key] if x[0]==0])
    nd = len(c_detects)
    y_trues = [];
    y_scores = [];
    dettag = set();
    for i in range(nd):
        det = c_detects[i]
        y_true = 0;
        if det[0] in c_truths :
            overlaps = np.array([ IoU(det[2],truthrect[1]) for truthrect in  c_truths[det[0]]]);       # get overlaps with truth rectangles
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            if ovmax> args.ovthresh:
                if c_truths[det[0]][jmax][0]==0 :   #if the truth label is not difficult
                    if (det[0],jmax) not in dettag:
                        y_true=1;
                        dettag.add((det[0],jmax))
                else:   #skip difficult examples
                    continue;
        y_trues += [ y_true ];
        y_scores += [ det[1] ];
    return (np.array(y_scores),np.array(y_trues), npos);

#split the file path into (directory, basename, ext)
def splitpath (filepath) :
    (dir,fname) = os.path.split(filepath);
    (basename,ext) = os.path.splitext(fname);
    return (dir,basename,ext);
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection evaluation')
    parser.add_argument('--truthfolder', required=True,   help='import groundtruth and baseline files')
    parser.add_argument('--dets', required=True,   help='import detection results')
    parser.add_argument('--name', default="", required=False,   help='the name of the experiment')
    parser.add_argument('--ovthresh', required=False, type=float, default=0.5,  help='IoU overlap threshold, default=0.5')
    args = parser.parse_args()
    return args

def load_baseline(truthfolder) :
    file_pattern = truthfolder+"/*.report";
    baseline_files = glob.glob(file_pattern)    
    baseline_files = sorted(baseline_files);
    retdict = dict();
    for file in baseline_files:
        (truth_dir, expname, ext) = splitpath(file); 
        with open(file,"r") as fin:
            report_metrics = json.load(fin);
            retdict[expname] = report_metrics;
    return retdict;
    
if __name__ == '__main__':
    # parse arguments
    args = parse_args();
    truthsfile = args.truthfolder + "/truths.tsv";
    assert  os.path.isfile(truthsfile), truthsfile + " is not found"
    baselines = load_baseline(args.truthfolder);
    detsfile = args.dets
    (report_dir, fbase, ext) = splitpath(detsfile);
    exp_name = args.name if args.name !="" else fbase;
    exp_name = '%s_%g'%(exp_name,args.ovthresh);    
    report_name = exp_name if report_dir=='' else '/'.join([report_dir,exp_name]);
    report_fig = report_name + ".png";
    report_file = report_name + ".report" 
    dataset_name = os.path.basename(args.truthfolder);
    
    #Load data
    truths = load_truths(truthsfile);
    detresults = load_dets(detsfile,truths);
    
    #calculate metrics
    y_scores = [];
    y_trues = [];
    npos = 0;
    apdict = dict();
    for label in sorted(truths.keys()):
        c_detects = detresults[label];        #detection results for current class
        c_truths = truths[label];             #truths for current class
        (c_y_scores, c_y_trues, c_npos) = evaluate(c_detects,c_truths);
        ap = metrics.average_precision_score(c_y_trues,c_y_scores) * np.sum(c_y_trues)/c_npos;
        y_scores += [c_y_scores];
        y_trues += [c_y_trues];
        npos += c_npos;
        apdict[label] = ap;

    for item in apdict.items():
        print(item[0],item[1]);
    print('------------------------')
    map = sum(apdict.values())/len(truths);
    print("MAP",map );

    y_trues = np.hstack(y_trues)
    y_scores = np.hstack(y_scores)
    coverage_ratio = float(np.sum(y_trues))/npos;
    precision, recall, thresholds = metrics.precision_recall_curve(y_trues,y_scores);
    ap = metrics.average_precision_score(y_trues,y_scores) * coverage_ratio;

    #save the evaluation result to the report file, which can be used as baseline
    with open(report_file,"w") as fout:
        report = {'class_ap':apdict,
            'MAP': map,
            'precision' : list(precision),
            'recall' : list(recall),
            'thresholds' : list(thresholds),
            'coverage_ratio' : coverage_ratio,
            'ap' : ap
            }
        fout.write(json.dumps(report,indent=4, sort_keys=True));
    
    #plot the PR-curve, and compare with baseline results
    recall *= coverage_ratio;
    fig =   plt.figure()
    plt.plot(recall, precision, 
             lw=2, label='%s (ap=%0.3g)' % (exp_name,ap))
    for exp in baselines:
        precision = np.array(baselines[exp]['precision']);
        recall = np.array(baselines[exp]['recall']) * baselines[exp]['coverage_ratio']
        plt.plot(recall, precision, lw=2, label='%s (ap=%.3g)'%(exp,baselines[exp]['ap']));

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Object detection PR Curve on %s dataset'%dataset_name)
    plt.legend(loc="lower right")
    fig.savefig(report_fig,dpi=fig.dpi)