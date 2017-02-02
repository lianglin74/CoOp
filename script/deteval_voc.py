import os
import sys
import json
import argparse
import numpy as np;

#load the detection results, organized by classes
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
                label = rect['class'].strip();
                bbox =  rect['rect'];
                retdict[label] += [ (key,rect['conf'],bbox)]
    for key in retdict:
        retdict[key] = sorted(retdict[key], key=lambda x:-x[1])
    return retdict;

def IoU_lei(rc1, rc2):
    overlap = 0.0
    iw = min(rc1[2], rc2[2] + 1) - max(rc1[0], rc2[0] + 1)
    ih = min(rc1[3], rc2[3] + 1) - max(rc1[1], rc2[1] + 1)
    iw = max(0, iw)
    ih = max(0, ih)
    rc1_area = (rc1[2] - rc1[0] + 1) * (rc1[3] - rc1[1] + 1)
    rc2_area = (rc2[2] - rc2[0] + 1) * (rc2[3] - rc2[1] + 1)
    inter = float(iw * ih)
    overlap = inter / (rc1_area + rc2_area - inter)
    return overlap

def rect_area(rc):
    return (rc[2]-rc[0] + 1)*(rc[3]-rc[1] + 1);


def IoU(rc1, rc2):
    rc_inter =  [max(rc1[0],rc2[0]), max(rc1[1],rc2[1]),min(rc1[2],rc2[2]), min(rc1[3],rc2[3]) ]
    iw = rc_inter [2] - rc_inter [0] + 1;
    ih = rc_inter [3] - rc_inter [1] + 1;
    return (float(iw))*ih/(rect_area(rc1)+rect_area(rc2)-iw*ih) if (iw>0 and ih>0) else 0;


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calcuate_tpfp (c_detects, c_truths,ovthresh):
        #calculate npos
        npos = 0;
        for key in c_truths:
            npos += len([x for x in c_truths[key] if x[0]==0])
        nd = len(c_detects)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        thresholds = np.zeros(nd)
        dettag = set();
        for i in range(nd):
            det = c_detects[i]
            thresholds[i] = det[1];
            if det[0] not in c_truths :
                fp[i]=1;
                continue;
            overlaps = np.array([ IoU(det[2],truthrect[1]) for truthrect in  c_truths[det[0]]]);       # get overlaps with truth rectangles
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            if ovmax> ovthresh:
                if c_truths[det[0]][jmax][0]==0 :
                    if (det[0],jmax) not in dettag:
                        tp[i]=1;
                        dettag.add((det[0],jmax))
                    else:
                        fp[i]=1;
            else:
                fp[i]=1;
        return (tp,fp,thresholds,npos);

def calculate_AP (tp,fp,npos,use_07_metric):
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap;
        

def eval(truths_file, det_file, ovthresh, use_07_metric):

    truths = load_truths(truths_file);
    detresults = load_dets(det_file,truths);
    map = 0;
    all_fp = [];
    all_tp = [];
    all_thesholds = [];
    npos_sum = 0;
    for label in sorted(truths.keys()):
        # go down dets and mark TPs and FPs
        c_detects = detresults[label];        #detection results for current class
        c_truths = truths[label];             #truths for current class
        (tp,fp,thresholds,npos) = calcuate_tpfp(c_detects,c_truths, ovthresh);
        all_tp += [tp];
        all_fp += [fp];
        all_thesholds += [thresholds];
        npos_sum += npos;
        ap = calculate_AP(tp, fp, npos, use_07_metric)
        map += ap;
        print(label,ap);

    print('------------------------')
    print("MAP", map/len(truths));

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Pascal VOC2007 evaluation')
    parser.add_argument('--truths', required=True,   help='import groundtruth file')
    parser.add_argument('--dets', required=True,   help='import detection results')
    parser.add_argument('--ovthresh', required=False, type=float, default=0.5,  help='IoU overlap threshold, default=0.5')
    parser.add_argument('-v', required=False, choices=['2007','2012'], default="2007",  help='2007: VOC2007 metric, 2012: VOC2012 metric')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args();
    use_07_metric = (args.v == '2007');
    eval(args.truths, args.dets, args.ovthresh, use_07_metric);
    
