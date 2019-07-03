import os
import os.path as op
import json
from qd.qd_common import load_list_file
from qd.tsv_io import TSVFile, tsv_reader
from qd.evaluate.oid_hierarchical_labels_expansion_tsv import expand_labels
import numpy as np
import argparse
import time


def load_truths(tsv_file, imagelabel_tsv_file, shuf_file=None):
    """ Load ground-truth annotations in both image-level and box-level.
    Args:
        tsv_file: A string of the tsv file with all ground-truth annotations.
                 The format of each row is as follows:
                    row[0]: image key
                    row[1]: a json string of image-level positive/negative labels
                    row[2]: a json string of box-level annotations for positive labels
        shuf_file: A string of the shuffle file with a list of indexes.
                   This feature enables to evaluate on a subset of images in tsv_file.

    Assert:
        Check if there is conflict labels. Eg., a class is labeled as negative while
        there is also box-level labels.

    Returns:
        gt_dict: A dictionary of all annotations.
                 It is grouped by class label first and then by image key.
                 For positive label, gt_dict[label][key] is a list of tuple with (IsGroupOf, rect).
                 For negative label, gt_dict[label][key] = -1
    """
    tsv = TSVFile(tsv_file)
    imagelabel_tsv = TSVFile(imagelabel_tsv_file)
    if shuf_file is None:
        shuf_list = [_ for _ in range(tsv.num_rows())]
    else:
        shuf_list = [int(x) for x in load_list_file(shuf_file)]

    gt_dict = {}
    for idx in shuf_list:
        row = tsv.seek(idx)
        key = row[0]
        imagelabel_key, str_imagelabel = imagelabel_tsv.seek(idx)
        assert imagelabel_key == row[0]
        img_label = json.loads(str_imagelabel)
        box_label = json.loads(row[1])
        if img_label==[] and box_label==[]:
            continue
        for obj in img_label:
            if obj['conf'] == 0:  # negative labels
                label = obj['class']
                if label not in gt_dict:
                    gt_dict[label] = {}
                if key not in gt_dict[label]:
                    gt_dict[label][key] = -1
                else:
                    assert gt_dict[label][key] == -1
        for obj in box_label:
            label = obj['class']
            if label not in gt_dict:
                gt_dict[label] = {}
            if key not in gt_dict[label]:
                gt_dict[label][key] = []
            gt_dict[label][key] += [(int(obj['IsGroupOf']), obj['rect'])]
    return gt_dict


def load_dets(tsv_file, truths):
    """ Load detection results.
    Args:
        tsv_file: A string of the tsv file with all detection results.
                 The format of each row is as follows:
                    row[0]: image key
                    row[1]: a json string of bounding box predictions (including class, confidence, rect)
        truths: A dictionary of ground-truth annotations.
                Detections of unknown classes are not recorded.

    Returns:
        det_dict: A dictionary of detections of all classes in truths.
                 Any invalid detections are removed (i.e., x2<x1, y2<y1).
                 It is grouped by class label first and then by image key.
                 det_dict[label][key] is a list of tuple with (confidence, rect)
                 and it is sorted by confidence (high to low).
    """
    det_dict = {cls:{} for cls in truths}
    for i, row in enumerate(tsv_reader(tsv_file)):
        key = row[0]
        rects = json.loads(row[1])
        for obj in rects:
            label = obj['class']
            if label in det_dict:
                if is_valid_rect(obj['rect']):
                    if key not in det_dict[label]:
                        det_dict[label][key] = []
                    det_dict[label][key] += [(obj['conf'], obj['rect'])]
    for label in det_dict:
        for key in det_dict[label]:
            det_dict[label][key] = sorted(det_dict[label][key], key=lambda x:-x[0])
    return det_dict


def is_valid_rect(rc):
    # a rect is valid if x2 > x1 and y2 > y1
    return rc[2] > rc[0] and rc[3] > rc[1]


def rect_area(rc):
    return (float(rc[2]) - rc[0]) * (rc[3] - rc[1])


def IoU(rc1, rc2):
    rc_inter = [max(rc1[0], rc2[0]), max(rc1[1], rc2[1]), min(rc1[2], rc2[2]), min(rc1[3], rc2[3])]
    if is_valid_rect(rc_inter):
        return rect_area(rc_inter) / (rect_area(rc1) + rect_area(rc2) - rect_area(rc_inter))
    else:
        return 0


def IoA(rc1, rc2):
    """ Intersection over smaller box area, used in group-of box evaluation.
    Args:
        rc1: A list of the smaller box coordinates in xyxy mode
        rc2: A list of the group box coordinates in xyxy mode
    Returns:
        ioa: A float number of ioa score = intersection(rc1, rc2) / area(rc1)
    """
    rc_inter = [max(rc1[0], rc2[0]), max(rc1[1], rc2[1]), min(rc1[2], rc2[2]), min(rc1[3], rc2[3])]
    if is_valid_rect(rc_inter):
        return rect_area(rc_inter) / rect_area(rc1)
    else:
        return 0


def get_overlaps(det, gt):
    """ Calculate IoU and IoA for a list of detection boxes and ground-truth boxes.
    Args:
        det: A list of D detection results (from det_dict[label][key])
        gt: A list of G ground-truth results (from gt_dict[label][key]),
            and say there are G1 group-of box and G2 non group-of box

    Returns:
        ious: A float numpy array (D*G1) of IoU scores between detection and non group-of ground-truth boxes
        ioas: A float numpy array (D*G2) of IoA scores between detection and group-of ground-truth boxes
    """
    gt_is_group = [g for g in gt if g[0]!=0]
    gt_is_non_group = [g for g in gt if g[0]==0]
    ious = [[IoU(d[1], g[1]) for g in gt_is_non_group] for d in det]
    ioas = [[IoA(d[1], g[1]) for g in gt_is_group] for d in det]
    return np.array(ious), np.array(ioas)


def eval_per_class(c_dets, c_truths, overlap_threshold=0.5, count_group_of=True):
    """ Evaluation for each class.
    Args:
        c_dets: A dictionary of all detection results (from det_dict[label])
        c_truths: A dictionary of all ground-truth annotations (from gt_dict[label])
        overlap_threshold: A float indicates the threshold used in IoU and IoA matching
        count_group_of: A bool indicates whether to consider group-of box or not

    Returns:
        scores_all: A list of numpy float array collecting the confidence scores of both
            truth positives and false positives in each image(ignored detections are not included)
        tp_fp_labels_all: A list of numpy float array collecting the true positives (=1)
            and false positives (=0) labels in each image
        num_gt_all: An integer of the total number of valid ground-truth boxes

    Note: the IsGroupOf feature can be 0, 1, and -1 (unknown).
        Follow Google's implementation, unknown is considered as group-of.
    """
    num_gt_all = 0
    for key in c_truths:
        if c_truths[key] == -1:
            continue    # negative label does not count
        is_group_of = [1 if x[0]!=0 else 0 for x in c_truths[key]]
        if count_group_of:
            num_gt_all += len(is_group_of)
        else:
            num_gt_all += sum(is_group_of)

    scores_all = []
    tp_fp_labels_all = []
    for key in c_dets:
        img_det = c_dets[key]
        num_det = len(img_det)
        scores = np.array([det[0] for det in img_det])
        tp_fp_labels = np.zeros(num_det, dtype=float)
        is_matched_to_group_of_box = np.zeros(num_det, dtype=bool)
        if key not in c_truths:
            continue  # ignore missing labels

        img_gt = c_truths[key]
        if img_gt == -1:
            # for negative label, all detections are fp
            scores_all.append(scores)
            tp_fp_labels_all.append(tp_fp_labels)
        else:
            ######## This part is imported modified from Google's implementation ########
            # The evaluation is done in two stages:
            # 1. All detections are matched to non group-of boxes; true positives are
            #    determined and detections matched to difficult boxes are ignored.
            # 2. Detections that are determined as false positives are matched against
            #    group-of boxes and scored with weight w per ground truth box is matched.

            [ious, ioas] = get_overlaps(img_det, img_gt)
            # Tp-fp evaluation for non-group of boxes (if any).
            if ious.shape[1] > 0:
                max_overlap_gt_ids = np.argmax(ious, axis=1)
                is_gt_box_detected = np.zeros(ious.shape[1], dtype=bool)
                for i in range(num_det):
                    gt_id = max_overlap_gt_ids[i]
                    if ious[i, gt_id] >= overlap_threshold:
                        if not is_gt_box_detected[gt_id]:
                            tp_fp_labels[i] = True
                            is_gt_box_detected[gt_id] = True

            scores_group_of = np.zeros(ioas.shape[1], dtype=float)
            tp_fp_labels_group_of = int(count_group_of) * np.ones(ioas.shape[1], dtype=float)
            # Tp-fp evaluation for group of boxes.
            if ioas.shape[1] > 0:
                max_overlap_group_of_gt_ids = np.argmax(ioas, axis=1)
                for i in range(num_det):
                    gt_id = max_overlap_group_of_gt_ids[i]
                    if not tp_fp_labels[i] and ioas[i, gt_id] >= overlap_threshold:
                        is_matched_to_group_of_box[i] = True
                        scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])
                selector = np.where((scores_group_of > 0) & (tp_fp_labels_group_of > 0))
                scores_group_of = scores_group_of[selector]
                tp_fp_labels_group_of = tp_fp_labels_group_of[selector]
            scores_all.append(np.concatenate((scores[~is_matched_to_group_of_box], scores_group_of)))
            tp_fp_labels_all.append(np.concatenate((tp_fp_labels[~is_matched_to_group_of_box], tp_fp_labels_group_of)))
            ######## end ########

    return scores_all, tp_fp_labels_all, num_gt_all


def compute_precision_recall(scores, labels, num_gt):
    assert np.sum(labels) <= num_gt, "number of true positives must be no larger than num_gt."
    assert len(scores) == len(labels), "scores and labels must be of the same size."
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    tp_labels = labels[sorted_indices]
    fp_labels = (tp_labels <= 0).astype(float)
    cum_tp = np.cumsum(tp_labels)
    cum_fp = np.cumsum(fp_labels)
    precision = cum_tp.astype(float) / (cum_tp + cum_fp)
    recall = cum_tp.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    if not precision.size:
        return 0.0
    assert len(precision) == len(recall), "precision and recall must be of the same size."
    assert np.amin(precision) >= 0 and np.amax(precision) <= 1, "precision must be in the range of [0, 1]."
    assert np.amin(recall) >= 0 and np.amax(recall) <= 1, "recall must be in the range of [0, 1]."
    assert all(recall[i] <= recall[i+1] for i in range(len(recall)-1)), "recall must be a non-decreasing array"

    rec = np.concatenate([[0], recall, [1]])
    prec = np.concatenate([[0], precision, [0]])
    # pre-process precision to be a non-decreasing array
    for i in range(len(prec) - 2, -1, -1):
      prec[i] = np.maximum(prec[i], prec[i + 1])
    indices = np.where(rec[1:] != rec[:-1])[0] + 1
    ap = np.sum((rec[indices] - rec[indices - 1]) * prec[indices])
    return ap


def evaluate(truths, imagelabel_truths, dets, shuf_file=None, expand_label_gt=False, expand_label_det=False, apply_nms_gt=False, apply_nms_det=False,
             json_hierarchy_file=None, count_group_of=True, overlap_threshold=0.5, save_file=None):
    if expand_label_gt:
        assert json_hierarchy_file is not None, "need json hierarchy file for label expansion"
        if not apply_nms_gt:
            new_file = op.splitext(truths)[0] + '.expanded.tsv'
            new_imagelevel_truths = op.splitext(imagelabel_truths)[0] + '.expanded.tsv'
        else:
            new_file = op.splitext(truths)[0] + '.expanded.nms.tsv'
            new_imagelevel_truths = op.splitext(imagelabel_truths)[0] + '.expanded.nms.tsv'
        print('expanding labels for ground-truth file and save to: ' + new_file)
        if not (op.isfile(new_file) and op.isfile(new_imagelevel_truths)):
            expand_labels(truths, imagelabel_truths, json_hierarchy_file,
                    new_file, new_imagelevel_truths, True, apply_nms_gt)
        truths = new_file
        imagelabel_truths = new_imagelevel_truths

    if expand_label_det:
        assert json_hierarchy_file is not None, "need json hierarchy file for label expansion"
        if not apply_nms_det:
            new_file = op.splitext(dets)[0] + '.expanded.tsv'
        else:
            new_file = op.splitext(dets)[0] + '.expanded.nms.tsv'
        print('expanding labels for detection file and save to: ' + new_file)
        if not op.isfile(new_file):
            expand_labels(dets, None, json_hierarchy_file, new_file,
                    None, False, apply_nms_det)
        dets = new_file

    truths_dict = load_truths(truths, imagelabel_truths, shuf_file)
    dets_dict = load_dets(dets, truths_dict)
    class_ap = {}
    class_num_gt = {}
    for label in sorted(truths_dict.keys()):
        c_truths = truths_dict[label]
        c_dets = dets_dict[label]
        scores, tp_fp_labels, num_gt = eval_per_class(c_dets, c_truths, overlap_threshold, count_group_of)
        if scores and num_gt:
            scores = np.concatenate(scores)
            tp_fp_labels = np.concatenate(tp_fp_labels)
            precision, recall = compute_precision_recall(scores, tp_fp_labels, num_gt)
            ap = compute_average_precision(precision, recall)
        else:
            ap = 0.0 # there are cases when num_gt = 0 and there are false positives.
        class_ap[label] = ap
        class_num_gt[label] = num_gt
        print(label, ap)

    mean_ap = sum([class_ap[cls] for cls in class_ap]) / len(truths_dict)
    total_gt = sum([class_num_gt[cls] for cls in class_num_gt])

    return {'class_ap': class_ap,
            'map': mean_ap,
            'class_num_gt': class_num_gt,
            'total_gt': total_gt}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='OpenImage Challenge Evaluation')
    parser.add_argument('--truths', required=True,   help='import groundtruth file')
    parser.add_argument('--shuf_file', required=False, default=None,
                        help='shuffle list to select a subset of images')
    parser.add_argument('--dets', required=True,   help='import detection results')
    parser.add_argument('--overlap_threshold', required=False, type=float, default=0.5,
                        help='IoU overlap threshold, default=0.5')
    parser.add_argument('--count_group_of', required=False, type=bool, default=True,
                        help='include group-of box evaluation or ignore default=True')
    parser.add_argument('--expand_label_gt', required=False, default=False, action='store_true',
                        help='whether to expand labels for gt annotations or not default=False')
    parser.add_argument('--expand_label_det', required=False, default=False, action='store_true',
                        help='whether to expand labels for detection annotations or not default=False')
    parser.add_argument('--apply_nms_gt', required=False, default=False, action='store_true',
                        help='whether to apply nms after gt label expansion, default=False')
    parser.add_argument('--apply_nms_det', required=False, default=False, action='store_true',
                        help='whether to apply nms after det label expansion, default=True')
    parser.add_argument('--json_hierarchy_file', required=False, type=str, default=None,
                        help='json file used for label expansion default=None')
    parser.add_argument('--save_file', required=False, type=str, default=None,
                        help='filename to save evaluation results (class AP)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    This script is for OpenImage Detection (OID) Challenge evaluation.
    It follows Google's official implementation to produce the same result,
    but it is much faster compared to Google's evaluation (2 mins vs. 50 mins
    not consider label expansion).

    However, there are three known issues with Google's implementation.
        1) Box area calculation: should +1 for height and width computation
        2) Duplicate label expansion: child class is expanded to the same parent
           more than once. (eg. Shrimp->Shellfish->seafood, Shrimp->Shellfish->..->Animal)
        3) Dense box matching: not considered (should follow COCO's evaluation)
           For example, we should not assign a detection to be false positive immediately
           when it is matched to an already matched ground-truth box. Instead, we should
           look at the next largest iou score to see if it can be matched to another
           un-matched ground-truth box. This is helpful for crowd scenes like human crowd.
           (we observe a 1.5 AP improvement on "Person" class after turn on this feature)

    A few other notes:
        1) The IsGroupOf attribute can be 0, 1 and -1 (unknown). There are 1,166 boxes
           are unknown in the validation set, which are considered as group-of boxes.
    """
    start = time.time()

    args = parse_args()
    evaluate(**vars(args))

    end = time.time()
    print('Elapsed time: ' + str(end - start))
