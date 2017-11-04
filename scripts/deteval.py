#!python3
import os
import sys
import json
import argparse
import numpy as np;
from sklearn import metrics;
import matplotlib.pyplot as plt
import glob;
from pytablemd import write_tablemd
from functools import partial

def load_truths(filein):
    '''
    Return: dict [class][image id] => bboxes
    '''
    retdict = dict();
    with open(filein, "r") as tsvin:
        for line in tsvin:
            cols = [x.strip() for x in line.split("\t")]
            if len(cols)<2:
                continue;
            key = cols[0]
            rects = json.loads(cols[1]) if cols[1]!='' else [];
            for rect in rects:
                label = rect['class'].strip();
                if label not in retdict:
                    retdict[label]=dict();
                if key not in retdict[label]:
                    retdict[label][key]=[];
                # groundtruth coords are 0-based. +1
                bbox = [ x+1 for x in rect['rect'] ];
                retdict[label][key]+=[(rect['diff'] if 'diff' in rect else 0,bbox)];
    return retdict;
    
def load_voc_dets(folderin):
    searchedfile = glob.glob(folderin+'/*.txt')
    assert (len(searchedfile)>0), "0 file matched by %s!"%(model_pattern)
    retdict = dict();
    for file in searchedfile:
        cname = file.split('_')[-1].split('.')[0];
        clist = []
        with open(file,'r') as tsvin:
            for line in tsvin:
                cols = [x.strip() for x in line.split(' ')]
                if len(cols)<6: continue
                key = cols[0];
                conf = float(cols[1])
                # voc coords are 1-based. no need to +1.
                rect = [float(x) for x in cols[2:]];
                clist += [(key, conf, rect)];
        retdict[cname]= sorted(clist, key=lambda x:-x[1]);
    return retdict;
    
#load the detection results, organized by classes
def load_dets(filein):
    '''
    Return: dict [class] => list of (image id, conf, bbox), in ascending order of conf
    '''
    retdict = dict();
    with open(filein, "r") as tsvin:
        for line in tsvin:
            cols = [x.strip() for x in line.split("\t")]
            if len(cols)<2:
                continue;
            key = cols[0]
            rects = json.loads(cols[1]);
            for rect in rects:
                label = rect['class'].strip();
                # coords +1 as we did for load_truths
                bbox = [ x+1 for x in rect['rect'] ];
                if label not in retdict:
                    retdict[label]=[]
                retdict[label] += [ (key,rect['conf'],bbox)]
    for label in retdict:
        retdict[label] = sorted(retdict[label], key=lambda x:-x[1])
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
def evaluate_(c_detects, c_truths, ovthresh):
    '''
    For each detection in a class, check whether it hits a ground truth box or not
    Return: (a list of confs, a list of hit or miss, number of ground truth boxes) 
    '''
    #calculate npos
    c_truths_not_hard = dict()
    npos = 0;
    for img_id in c_truths:
        npos += len([difficulty_gtbox for difficulty_gtbox in c_truths[img_id] if difficulty_gtbox[0] == 0])

    y_trues = []
    y_scores = []

    dettag = set()

    for i in range(len(c_detects)):
        det = c_detects[i]
        y_true = 0;
        img_id = det[0]
        conf = det[1]
        bbox = det[2]
        if img_id in c_truths:
            # get overlaps with truth rectangles
            overlaps = np.array([IoU(bbox, gtbox[1]) for gtbox in c_truths[img_id]])
            bbox_idx_max = np.argmax(overlaps)
            if overlaps[bbox_idx_max] > ovthresh:
                # if a detection hits a difficult gt_box, skip this detection
                if c_truths[img_id][bbox_idx_max][0] != 0:
                    continue

                if (img_id, bbox_idx_max) not in dettag:
                    y_true = 1
                    dettag.add((img_id, bbox_idx_max))

        y_trues += [ y_true ]
        y_scores += [ conf ]
    return (np.array(y_scores), np.array(y_trues), npos)

#split the file path into (directory, basename, ext)
def splitpath (filepath) :
    (dir,fname) = os.path.split(filepath);
    (basename,ext) = os.path.splitext(fname);
    return (dir,basename,ext);
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection evaluation')
    parser.add_argument('-t', '--truth', required=True,   help='import groundtruth')
    parser.add_argument('-b', '--baselinefolder', required=False,   help='import baseline files')
    parser.add_argument('-d', '--dets', required=False, default='',  help='import detection results')
    parser.add_argument('-v', '--vocdets', required=False, default='',  help='import voc2007 detection results')
    parser.add_argument('-n', '--name', default="", required=False,   help='the name of the experiment')
    parser.add_argument('-o', '--ovthresh', required=False, type=float, nargs='+', default=[0.3,0.4,0.5],  help='IoU overlap threshold, default=0.5')
    parser.add_argument('-p', '--precth', required=False, type=float, nargs='+', default=[0.8,0.9,0.95], help="get precision, recall, threshold above given precision threshold")
    parser.add_argument('-ms', '--multiscale', default=False, action='store_true', help='Flag to enable report metrics on small, medium, large object, default: False')
    parser.add_argument('-c', '--classap', required=False, type=float,  help='the per class precision on given IOU')
    
    args = parser.parse_args()
    return args

def load_baseline(baselinefolder) :
    file_pattern = baselinefolder+"/*.report";
    baseline_files = glob.glob(file_pattern)    
    baseline_files = sorted(baseline_files);
    retdict = dict();
    for file in baseline_files:
        (truth_dir, expname, ext) = splitpath(file); 
        with open(file,"r") as fin:
            report_metrics = json.load(fin);
            retdict[expname] = report_metrics;
    return retdict;

def _eval(truths, detresults, ovthresh):
    #calculate metrics
    y_scores = [];
    y_trues = [];
    npos = 0;
    apdict = dict();
    for label in sorted(truths.keys()):
        if label not in detresults:
            continue;
        c_detects = detresults[label];        #detection results for current class
        c_truths = truths[label];             #truths for current class
        (c_y_scores, c_y_trues, c_npos) = evaluate_(c_detects, c_truths, ovthresh)
        if len(c_detects)>0:
            c_true_sum = np.sum(c_y_trues)
            ap = metrics.average_precision_score(c_y_trues, c_y_scores) * c_true_sum/c_npos if c_true_sum > 0 else 0
            y_scores += [c_y_scores];
            y_trues += [c_y_trues];
            apdict[label] = ap;
        else:
            apdict[label]=0;
        npos += c_npos;
    map = sum(apdict.values())/len(truths);
    y_trues = np.hstack(y_trues) if len(y_trues) != 0 else np.array([0])
    y_scores = np.hstack(y_scores) if len(y_scores) != 0 else np.array([0])
    coverage_ratio = float(np.sum(y_trues))/npos if npos != 0 else 0
    precision, recall, thresholds = metrics.precision_recall_curve(y_trues,y_scores);
    precision = list(precision);
    thresholds = list(thresholds);
    if len(thresholds)<len(precision):
        thresholds += [thresholds[-1]];
    #plot the PR-curve, and compare with baseline results
    recall *= coverage_ratio;
    recall = list(recall);    
    return  { 'class_ap':apdict,
            'map': map,
            'precision' : precision,
            'recall' : recall,
            'thresholds' : thresholds,
            'npos': npos,
            'coverage_ratio' : coverage_ratio
            }

def eval(truthsfile, detsfile, ovthresh):
    truths = load_truths(truthsfile);
    detresults = load_dets(detsfile);
    return _eval(truths, detresults, ovthresh);
    
def get_pr(report, thresh):
    idx = np.where(np.array(report['precision'])>thresh);
    recall_ = np.array(report['recall'])[idx];
    maxid = np.argmax(np.array(recall_));
    maxid = idx[0][maxid]
    return report['thresholds'][maxid], report['precision'][maxid], report['recall'][maxid]
'''    
def print_pr(report, thresh):
    th,prec,rec = get_pr(report, thresh)
    print("\t%9.6f\t%9.6f\t%9.6f"%(th,prec,rec));
    

def drawfigs(report, baselines, exp_name,report_fig):
    #plot the PR-curve, and compare with baseline results
    fig =   plt.figure()
    plt.plot(report['recall'], report['precision'], lw=2, label='%s (ap=%0.3g)' % (exp_name,report['map']))
    for exp in baselines:
        precision = np.array(baselines[exp]['precision']);
        recall = np.array(baselines[exp]['recall']) 
        plt.plot(recall, precision, lw=2, label='%s (ap=%.3g)'%(exp,baselines[exp]['map']));

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Object detection PR Curve on %s dataset'%dataset_name)
    plt.legend(loc="lower right")
    fig.savefig(report_fig,dpi=fig.dpi)
'''    
def  gen_truthslist(truths):
    truths_small = dict()
    truths_medium = dict()
    truths_large = dict()
    
    for label in truths:
        if label not in truths_small:
            truths_small[label] =dict();
            truths_medium[label] =dict();
            truths_large[label] =dict();
        for key in truths[label]:
            crects_small = [];
            crects_medium = [];
            crects_large = [];
            for item in truths[label][key]:
                rect = item[1];
                area = (rect[2]-rect[0])*(rect[3]-rect[1]);
                tags = tagm = tagl = 1;
                if not item[0]:
                    if area>32*32:
                        if area>96*96:
                            tagl=0;
                        else:
                            tagm=0;
                    else:
                        tags=0;
                crects_small  += [(tags,rect)]
                crects_medium += [(tagm,rect)]
                crects_large  += [(tagl,rect)]
            truths_small[label][key] = crects_small;
            truths_medium[label][key] = crects_medium;
            truths_large[label][key] = crects_large;
    return {'small':truths_small, 'medium':truths_medium, 'large':truths_large, 'overall':truths};  

def get_report (truths, dets, ovths, msreport):
    truths_list = gen_truthslist(truths) if msreport==True else {'overall':truths};
    reports = dict();
    for part in truths_list:
        reports[part] = dict()
        for ov_th in ovths:    
            reports[part][ov_th] = _eval(truths_list[part], dets, ov_th);
    return reports;   #return the overal reports

def print_reports(reports, precths):
    for key in reports:
        table = []
        headings = ['IOU', 'MAP']
        mv_report = reports[key]
        for precth in precths:
            headings += ['Th@%g'%precth, 'Prec@%g'%precth, 'Rec@%g'%precth ]
        for ov_th in sorted(mv_report.keys()):
            report = mv_report[ov_th]
            data = [ov_th, report['map']]
            for precth in precths:
                pr = get_pr(report, precth)
                data += list(pr);
            data = tuple([round(x,4) for x in data])
            table +=[data]
        fields = list(range(len(data)));
        align = [('^', '<')] + [('^', '^')]*len(data)
        print('Results on %s objects (%d)'% (key, mv_report[mv_report.keys()[0]]['npos']))
        write_tablemd(sys.stdout, table,fields,headings,align)
   
def deteval(truth='', dets='', vocdets='', name='', 
        precth=[0.8,0.9,0.95], multiscale=False, ovthresh=[0.3,0.4,0.5],
        classap=None, baselinefolder=None):
    truthsfile = truth
    # parse arguments
    assert  os.path.isfile(truthsfile), truthsfile + " is not found"

    #Load data
    if dets!='' :
        detsfile = dets
        (report_dir, fbase, ext) = splitpath(detsfile);
    elif vocdets!='':
        report_dir = vocdets
        fbase = 'voc2007'
    else:
        assert False, "argument dets/vocdets is missing!"

    #save the evaluation result to the report file, which can be used as baseline
    exp_name = name if name !="" else fbase;
    report_name = exp_name if report_dir=='' else '/'.join([report_dir,exp_name]);
    report_file = report_name + ".report" 

    if os.path.isfile(report_file):
        print 'skip to evaluate since the report file exists, ', report_file
        return report_file

    if dets!='' :
        detresults = load_dets(detsfile);
    elif vocdets!='':
        detresults = load_voc_dets(vocdets);
    else:
        assert False, "argument dets/vocdets is missing!"
        
    truths = load_truths(truthsfile);
    #brief report on different object size
    reports = get_report(truths, detresults, ovthresh, multiscale)
    # detail report with p-r curve
    
    with open(report_file,"w") as fout:
        fout.write(json.dumps(reports,indent=4, sort_keys=True));
    
    print_reports(reports, precth)
    if classap is not None and classap in ovthresh:
        caplist = sorted(reports['overall'][classap]['class_ap'].items(), key=lambda x:-x[1])
        for pair in caplist:
            print('%s\t%.4g'%pair)

    return report_file

if __name__ == '__main__':
    args = parse_args();
    deteval(**vars(args))

