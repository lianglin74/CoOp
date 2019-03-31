#!python3
import os
import sys
import json
import argparse
import numpy as np
from .taxonomy import LabelTree
from sklearn import metrics;
import matplotlib.pyplot as plt
import glob;
from pytablemd import write_tablemd
from functools import partial
from .qd_common import FileProgressingbar, worth_create
import logging
import copy
from .tsv_io import tsv_writer
from tqdm import tqdm

ENTITY_LABEL = "__entity"
BACKGROUND_LABEL = "__background"

def load_truths_iter(rows, region_only=False):
    logging.info('loading gt')
    '''
    Return: dict [class][image id] => bboxes
    '''
    retdict = dict();
    for cols in tqdm(rows):
        if len(cols)<2:
            continue;
        key = cols[0]
        try:
            rects = json.loads(cols[1]) if cols[1]!='' else [];
        except:
            logging.info('invalid grouth truth: {}'.format(cols[0]))
            continue
        if type(rects) is int:
            label = str(rects)
            if label not in retdict:
                retdict[label] = dict()
            if key not in retdict[label]:
                retdict[label][key] = []
            retdict[label][key] += [(0, None)]
            continue
        for rect in rects:
            if region_only:
                label = ENTITY_LABEL
            else:
                label = rect['class'].strip().lower();
            if label not in retdict:
                retdict[label]=dict();
            if key not in retdict[label]:
                retdict[label][key] = []
            # groundtruth coords are 0-based. +1
            if 'rect' in rect:
                bbox = [ x+1 for x in rect['rect'] ]
            else:
                bbox = None
            retdict[label][key]+=[(rect['diff'] if 'diff' in rect else 0,bbox)];
    return retdict;

def load_truths(filein):
    logging.info('loading gt')
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
            try:
                rects = json.loads(cols[1]) if cols[1]!='' else [];
            except:
                logging.info('invalid grouth truth: {}'.format(cols[0]))
                continue
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
def load_dets(filein, region_only=False):
    '''
    Return: dict [class] => list of (image id, conf, bbox), in ascending order of conf
    '''
    retdict = dict();
    with open(filein, "r") as tsvin:
        # the file.tell() is disabled by the next function and the
        # FileProgressBar here is not supported in python3.
        #bar = FileProgressingbar(tsvin, 'load-dets')
        for line in tqdm(tsvin):
            cols = [x.strip() for x in line.split("\t")]
            if len(cols)<2:
                continue;
            key = cols[0]
            rects = json.loads(cols[1]);
            for rect in rects:
                if region_only:
                    label = ENTITY_LABEL
                else:
                    label = rect['class'].strip().lower();
                if label == BACKGROUND_LABEL:
                    continue
                if 'rect' in rect:
                    # coords +1 as we did for load_truths
                    bbox = [ x+1 for x in rect['rect'] ];
                else:
                    bbox = None
                if label not in retdict:
                    retdict[label]=[]
                retdict[label] += [ (key,rect['conf'],bbox)]
            #bar.update()
    for label in retdict:
        retdict[label] = sorted(retdict[label], key=lambda x:-x[1])
    return retdict;

def load_dets_iter(rows, region_only=False):
    '''
    Return: dict [class] => list of (image id, conf, bbox), in ascending order of conf
    '''
    retdict = dict();
    for cols in rows:
        if len(cols)<2:
            continue;
        key = cols[0]
        rects = json.loads(cols[1]);
        for rect in rects:
            if region_only:
                label = ENTITY_LABEL
            else:
                label = rect['class'].strip().lower();
            if label == BACKGROUND_LABEL:
                continue
            if 'rect' in rect:
                # coords +1 as we did for load_truths
                bbox = [ x+1 for x in rect['rect'] ];
            else:
                bbox = None
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
    if rc1 and rc2:
        rc_inter =  [max(rc1[0],rc2[0]), max(rc1[1],rc2[1]),min(rc1[2],rc2[2]), min(rc1[3],rc2[3]) ]
        iw = rc_inter [2] - rc_inter [0] + 1;
        ih = rc_inter [3] - rc_inter [1] + 1;
        return (float(iw))*ih/(rect_area(rc1)+rect_area(rc2)-iw*ih) if (iw>0 and ih>0) else 0;
    else:
        return 0

#evaluate the detection results
def evaluate_(c_detects, c_truths, ovthresh):
    '''
    For each detection in a class, check whether it hits a ground truth box or not
    Return: (a list of confs, a list of hit or miss, number of ground truth boxes)
    '''
    #calculate npos
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
    parser.add_argument('-r', '--report_file', required=True, type=str,
            help='the report file')


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

def _eval(truths, detresults, ovthresh, confs=None, label_to_keys=None):
    if confs is None:
        confs = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    #calculate metrics
    y_scores = []
    y_trues = []
    npos = 0
    class_thresh = dict()
    apdict = dict()
    class_prec_recall_th = {}
    for label in sorted(truths.keys()):
        if label not in detresults:
            apdict[label] = 0
            continue;
        c_detects = detresults[label]        #detection results for current class
        c_truths = truths[label]             #truths for current class
        if label_to_keys is not None:
            valid_keys = label_to_keys.get(label, [])
            c_truths = {key: c_truths[key] for key in c_truths if key in
                valid_keys}
            c_detects = [(key, conf, bbox) for key, conf, bbox in c_detects
                    if key in valid_keys]
        (c_y_scores, c_y_trues, c_npos) = evaluate_(c_detects, c_truths, ovthresh)
        if confs and np.sum(c_y_trues):
            precision, recall, thresholds = metrics.precision_recall_curve(c_y_trues, c_y_scores)
            class_prec_recall_th[label] = [[float(p) for p in precision],
                    [float(r) for r in recall], [float(t) for t in thresholds]]
            for conf in confs:
                # precision is in ascending order
                indices, = np.where((precision > conf) & (recall > 0.0))
                if len(indices) == 0:
                    continue
                if label not in class_thresh:
                    class_thresh[label] = {}
                class_thresh[label].update({
                    conf: (thresholds[indices[0]], recall[indices[0]])
                })
        if len(c_detects) > 0:
            c_true_sum = np.sum(c_y_trues)
            ap = metrics.average_precision_score(c_y_trues, c_y_scores) * c_true_sum/c_npos if c_true_sum > 0 else 0
            y_scores += [c_y_scores]
            y_trues += [c_y_trues]
            apdict[label] = ap
        else:
            apdict[label] = 0
        npos += c_npos
    map = sum(apdict.values())/ (1 if len(truths) == 0 else len(truths))
    y_trues = np.hstack(y_trues) if len(y_trues) != 0 else np.array([0])
    y_scores = np.hstack(y_scores) if len(y_scores) != 0 else np.array([0])
    coverage_ratio = float(np.sum(y_trues))/npos if npos != 0 else 0
    if np.sum(y_trues) == 0:
        # in this case, metrics.precision_recall_curve will crash
        precision = np.asarray([0.])
        recall = np.asarray([0.])
        thresholds = np.asarray([0.])
    else:
        precision, recall, thresholds = metrics.precision_recall_curve(y_trues,y_scores);
    precision = list(precision);
    thresholds = list(thresholds);
    if len(thresholds)<len(precision):
        thresholds += [thresholds[-1]];
    #plot the PR-curve, and compare with baseline results
    recall *= coverage_ratio;
    recall = list(recall);
    return {
        'class_ap': apdict,
        'class_thresh': class_thresh,
        'class_prec_recall_th': class_prec_recall_th,
        'map': map,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'npos': npos,
        'coverage_ratio': coverage_ratio
    }

def eval(truthsfile, detsfile, ovthresh):
    truths = load_truths(truthsfile);
    detresults = load_dets(detsfile);
    return _eval(truths, detresults, ovthresh);

def get_pr(report, thresh):
    idx = np.where(np.array(report['precision'])>thresh);
    if len(idx) == 0:
        return 0, 0, 0
    recall_ = np.array(report['recall'])[idx];
    if len(recall_) == 0:
        return 0, 0, 0
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

def get_report (truths, dets, ovths, msreport, label_to_keys=None):
    truths_list = gen_truthslist(truths) if msreport==True else {'overall':truths};
    reports = dict();
    for part in truths_list:
        reports[part] = dict()
        for ov_th in ovths:
            reports[part][ov_th] = _eval(truths_list[part], dets, ov_th,
                    label_to_keys=label_to_keys);
    return reports;   #return the overal reports

def print_reports(reports, precths, report_file_table):
    fp = open(report_file_table, 'w')
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
        # mv_report.keys() does not return a list in python3. Thus, we have to
        # convert it to a list first
        note = ('Results on %s objects (%d)'% (key,
            mv_report[list(mv_report.keys())[0]]['npos']))
        write_tablemd(sys.stdout, table,fields,headings,align)
        fp.write(note + '\n')
        write_tablemd(fp, table,fields,headings,align)
    fp.close()

def lift_detects(detresults, label_tree):
    result = {}
    for label in detresults:
        dets = detresults[label]
        all_label = [label]
        nodes = label_tree.root.search_nodes(name=label)
        assert len(nodes) == 1
        node = nodes[0]
        for n in node.get_ancestors()[: -1]:
            all_label.append(n.name)
        for l in all_label:
            if l not in result:
                result[l] = copy.deepcopy(dets)
            else:
                result[l].extend(dets)
    return result

def lift_truths(truths, label_tree):
    result = {}
    expanded = {}
    for label in truths:
        imid_to_rects = truths[label]
        all_label = [label]
        nodes = label_tree.root.search_nodes(name=label)
        assert len(nodes) == 1
        node = nodes[0]
        for n in node.get_ancestors()[: -1]:
            all_label.append(n.name)
        logging.info('->{}'.format(','.join(all_label)))
        for l in all_label:
            if l not in result:
                result[l] = copy.deepcopy(imid_to_rects)
            else:
                r = result[l]
                for imid in imid_to_rects:
                    rects = imid_to_rects[imid]
                    if imid in r:
                        r[imid].extend(rects)
                    else:
                        r[imid] = rects
    return result

def has_negative_labels(truths):
    return any(l for l in truths if l.startswith('-'))

def remove_negative_labels(truths):
    neg_labels = [l for l in truths if l.startswith('-')]
    label_to_keys = {}
    for l in truths:
        pos_l = l if not l.startswith('-') else l[1:]
        if pos_l not in label_to_keys:
            label_to_keys[pos_l] = []
        label_to_keys[pos_l].extend(truths[l].keys())
    for l in neg_labels:
        del truths[l]
    return label_to_keys

def deteval_iter(truth_iter, dets='', vocdets='', name='',
        precth=[0.8,0.9,0.95], multiscale=False, ovthresh=[0.3,0.4,0.5],
        classap=None, baselinefolder=None, report_file=None,
        label_to_keys=None, **kwargs):
    '''
    truth_iter is an iterator
    '''
    assert report_file, 'report_file must be specified'
    #Load data
    if dets!='' :
        detsfile = dets
        (report_dir, fbase, ext) = splitpath(detsfile);
    elif vocdets!='':
        report_dir = vocdets
        fbase = 'voc2007'
    else:
        assert False, "argument dets/vocdets is missing!"

    if os.path.isfile(report_file) and \
            not kwargs.get('force_evaluate', False) and \
            not worth_create(detsfile, report_file):
        logging.info('skip to evaluate (exists) {}'.format(report_file))
        return report_file

    if dets!='' :
        detresults = load_dets(detsfile, kwargs.get("region_only", False));
    elif vocdets!='':
        detresults = load_voc_dets(vocdets);
    else:
        assert False, "argument dets/vocdets is missing!"

    truths = load_truths_iter(truth_iter, kwargs.get("region_only", False));
    if has_negative_labels(truths):
        assert label_to_keys is None
        label_to_keys = remove_negative_labels(truths)

    #brief report on different object size
    reports = get_report(truths, detresults, ovthresh, multiscale,
            label_to_keys=label_to_keys)
    # detail report with p-r curve

    with open(report_file,"w") as fout:
        fout.write(json.dumps(reports,indent=4, sort_keys=True));

    print_reports(reports, precth, report_file + '.table')

    if classap is not None and classap in ovthresh:
        caplist = sorted(reports['overall'][classap]['class_ap'].items(), key=lambda x:-x[1])
        for pair in caplist:
            print('%s\t%.4g'%pair)

    return report_file

def deteval(truth='', dets='', vocdets='', name='',
        precth=[0.8,0.9,0.95], multiscale=True, ovthresh=[0.3,0.4,0.5],
        classap=None, baselinefolder=None, **kwargs):
    '''
    use deteval_iter if possible, which is more general
    '''
    from tsv_io import tsv_reader
    return deteval_iter(tsv_reader(truth),
            dets, vocdets, name, precth, multiscale, ovthresh,
            classap, baselinefolder, **kwargs)

if __name__ == '__main__':
    args = parse_args();
    deteval(**vars(args))


