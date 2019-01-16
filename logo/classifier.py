import base64
import collections
import datetime
import json
import logging
import numpy as np
import os
import torch

from sklearn.metrics import roc_curve, auc
import matplotlib
# use a non-interactive backend to generate images without having a window appear
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from tagging.scripts import extract, pred
from tagging.utils import accuracy
from scripts.qd_common import calculate_iou, write_to_yaml_file, load_from_yaml_file, init_logging, worth_create
from evaluation.eval_utils import DetectionFile
from scripts.tsv_io import TSVDataset, TSVFile, tsv_reader, tsv_writer
from scripts.yolotrain import yolo_predict


BACKGROUND_LABEL = "__background"
BBOX_POS_PAIR_IDS = "pos_pair_ids"
BBOX_NEG_PAIR_IDS = "neg_pair_ids"


class CropTaggingWrapper(object):
    def __init__(self, det_expid, tag_expid):
        self.det_expid = det_expid
        self._rootpath = "/raid/data/brand_output/{}/classifier/{}".format(det_expid, tag_expid)
        self.labelmap = os.path.join(self._rootpath, "labelmap.txt")
        self.tag_model_path = os.path.join(self._rootpath, "snapshot/model_best.pth.tar")
        self.log_file = os.path.join(self._rootpath, "prediction_log.txt")

        self.num_workers = 24

    def predict_on_known_class(self, dataset_name, split):
        """
        Two stage methods by combining detector with classifier
        Returns file of imgkey, list of bboxes
        """
        # get region proposal
        rp_file = self.det_predict(dataset_name, split)

        # get tagging results
        data_yaml = self._write_data_yaml(dataset_name, split, "test", rp_file)
        tag_file = self.tag_predict(data_yaml)

        # combine
        outfile = os.path.join(self._rootpath, "{}.{}.tag.predict.tsv".format(dataset_name, split))
        if worth_create(tag_file, outfile):
            parse_tagging_predict(tag_file, outfile)
        return outfile

    def predict_on_unknown_class(self, dataset_name, split, region_source="predict"):
        """
        Predicts on unknown classes by comparing similarity with reference database
        Returns file of imgkey, list of bboxes
        """
        # get region proposal
        if region_source == "predict":
            rp_file = self.det_predict(dataset_name, split)
            top_k_acc = None
        elif region_source == "gt":
            d = TSVDataset(dataset_name)
            rp_file = d.get_data(split, t='label', version=-1)
            top_k_acc = (1,5)
        else:
            raise ValueError("Invalid region source: {}".format(region_source))

        # get feature for prediction
        data_yaml = self._write_data_yaml(dataset_name, split, "test", rp_file)
        fea_file = self.extract_feature(data_yaml)

        # TODO: configure reference database
        # get feature for gt/canonical
        prototype_dataset = "logo40"
        prototype_split = "train"
        data_yaml = self._write_data_yaml(prototype_dataset, prototype_split, "test")
        gt_fea_file = self.extract_feature(data_yaml)

        # compare similarity
        outfile = os.path.join(self._rootpath,
                "{}.{}.fea.predict.region.{}.tsv".format(dataset_name, split, region_source))
        if worth_create(fea_file, outfile) or worth_create(gt_fea_file, outfile):
            acc_str = compare_similarity(gt_fea_file, fea_file, outfile, top_k_acc=top_k_acc)
            if acc_str:
                with open(self.log_file, 'a') as fp:
                    fp.write("\nTime: {}\t Method: {}\n" \
                            "Test dataset: {}({})\t RegionProposal: {}\t Prototype database: {}({})\n".format(
                            datetime.datetime.now(), "predict_on_unknown_class",
                            dataset_name, split, rp_file, prototype_dataset, prototype_split))
                    fp.write(acc_str)
                    fp.write('\n')
        return outfile

    def compare_pairs(self, dataset_name, split):
        """
        Internal metric for featurizer, calculate ROC for positive and negative pairs
        """
        # TODO: configure constant
        rp_file = "/raid/data/logo40_pair/test.label.tsv"
        data_yaml = self._write_data_yaml(dataset_name, split, "test", rp_file)
        fea_file = self.extract_feature(data_yaml)

        outfile = os.path.join(self._rootpath, "{}.{}.pair.roc.png".format(dataset_name, split))
        if worth_create(fea_file, outfile):
            compare_pair_features(fea_file, outfile)
        return outfile

    def det_predict(self, dataset_name, split):
        pred_file, _ = yolo_predict(full_expid=self.det_expid, test_data=dataset_name, test_split=split)
        return pred_file

    def tag_predict(self, datayaml, force_rewrite=False):
        outpath = "{}.tagging.tsv".format(datayaml.rsplit('.', 1)[0])
        if not force_rewrite and not worth_create(datayaml, outpath):
            logging.info("skip tagging, already exists: {}".format(outpath))
            return outpath
        pred.main([
            datayaml,
            "--model", self.tag_model_path,
            "--output", outpath,
            "--labelmap", self.labelmap,
            "--topk", str(1),
            "--workers", str(self.num_workers)
        ])
        return outpath

    def extract_feature(self, datayaml, force_rewrite=False):
        outpath = "{}.feature.tsv".format(datayaml.rsplit('.', 1)[0])
        if not force_rewrite and not worth_create(datayaml, outpath):
            logging.info("skip extracting feature, already exists: {}".format(outpath))
            return outpath
        extract.main([
            datayaml,
            "--model", self.tag_model_path,
            "--output", outpath,
            "--workers", str(self.num_workers)
        ])
        return outpath

    def _write_data_yaml(self, dataset_name, split, session, labelfile=None):
        dataset = TSVDataset(dataset_name)
        data_yaml_info = [dataset_name, split]
        if labelfile:
            data_yaml_info.append("label"+str(hash(labelfile)))
        data_yaml_info.append("dataset.yaml")
        data_yaml = os.path.join(self._rootpath, ".".join(data_yaml_info))
        if os.path.isfile(data_yaml):
            data_config = load_from_yaml_file(data_yaml)
        else:
            data_config = {}

        new_cfg = {
            "tsv": os.path.realpath(dataset.get_data(split)),
            "labelmap": self.labelmap
        }
        if labelfile:
            new_cfg["label"] = os.path.realpath(labelfile)

        if session in data_config:
            if self._is_config_equal(data_config[session], new_cfg):
                return data_yaml

        data_config.update({session: new_cfg})
        write_to_yaml_file(data_config, data_yaml)
        return data_yaml

    def _is_config_equal(self, cfg1, cfg2):
        if len(cfg1) != len(cfg2):
            return False
        for k1 in cfg1:
            if k1 not in cfg2:
                return False
            if cfg1[k1] != cfg2[k1]:
                return False
        return True


def compare_similarity(gt_fea_file, pred_fea_file, outfile, nms_type="cls_dep", top_k_acc=None):
    """
    Compares feature similarity (cosine distance)
    gt_fea is the feature of canonical images, pred_fea is the feature of real images
    gt_fea_file, pred_fea_file cols: imgkey, bbox, b64_feature_string
    outfile cols: imgkey (from pred_fea_file), list of bboxes
    top_k_acc: tuple of integers, calculate precision@k for the feature matching results
            against original labels in pred_fea_file
    """
    gt_feas = []
    class_indices = {}
    all_classes = []
    for cols in tsv_reader(gt_fea_file):
        bbox = json.loads(cols[1])
        fea = from_b64_to_fea(cols[2])
        c = bbox["class"]
        if c not in class_indices:
            class_indices[c] = len(all_classes)
            all_classes.append(c)
        gt_feas.append((class_indices[c], fea))

    pred_dict = collections.defaultdict(list)  # imgkey: list of bbox
    num_classes = len(class_indices)
    all_target = []  # batch_size * 1 (single label)
    all_pred = []  # batch_size * num_classes
    for cols in tsv_reader(pred_fea_file):
        bbox = json.loads(cols[1])
        cur_fea = from_b64_to_fea(cols[2])
        all_sim = np.array([cosine_similarity(cur_fea, f[1]) for f in gt_feas])

        if top_k_acc and bbox["class"] in class_indices:
            all_target.append([class_indices[bbox["class"]]])
            cur_pred = [0] * num_classes
            for i, score in enumerate(all_sim):
                c_idx = gt_feas[i][0]
                cur_pred[c_idx] = max(score, cur_pred[c_idx])
            all_pred.append(cur_pred)
        # get the nearest neighbor
        max_idx = np.argmax(all_sim)
        bbox["class"] = all_classes[gt_feas[max_idx][0]]
        obj_score = bbox["obj"] if "obj" in bbox else 1.0
        bbox["conf"] = obj_score * all_sim[max_idx]
        pred_dict[cols[0]].append(bbox)

    if nms_type:
        for key in pred_dict:
            pred_dict[key] = nms_wrapper(pred_dict[key], nms_type=nms_type)

    tsv_writer([[k, json.dumps(pred_dict[k])] for k in pred_dict], outfile)

    if top_k_acc:
        acc = accuracy.SingleLabelAccuracy(top_k_acc)
        acc.calc(torch.from_numpy(np.array(all_pred)), torch.tensor(all_target))
        return acc.result_str()
    else:
        return ""


def parse_tagging_predict(infile, outfile, nms_type="cls_dep", bg_skip=1):
    """ Convert two-stage prediction result to regular format
    Args:
        infile: imgkey, bbox, classification results (term:conf;)
        outfile: imgkey, bbox list
        nms_type: None (no nms), cls_dep, cls_indep
        bg_skip: int, if background is in the first [n] prediction, it is treated as bg
    """
    pred_dict = collections.defaultdict(list) # key: a list of bbox
    for cols in tsv_reader(infile):
        assert(len(cols) == 3)
        key = cols[0]
        bbox = json.loads(cols[1])
        # choose the first label
        label_conf_pair = [p.rsplit(':', 1) for p in cols[2].split(';')]

        is_bg = False
        for i, (label, conf) in enumerate(label_conf_pair):
            if i >= bg_skip:
                break
            if label == BACKGROUND_LABEL:
                is_bg = True
                break
        if is_bg:
            continue

        # use classification label as label, classification score * obj as conf score
        bbox["class"] = label_conf_pair[0][0]
        bbox["conf"] = float(label_conf_pair[0][1]) / 100 * bbox["obj"]

        pred_dict[key].append(bbox)
    if nms_type:
        for key in pred_dict:
            pred_dict[key] = nms_wrapper(pred_dict[key], nms_type=nms_type)

    tsv_writer([[k, json.dumps(pred_dict[k])] for k in pred_dict], outfile)


def compare_pair_features(fea_file, outfile):
    pos_pair_lineidx_dict = {}
    neg_pair_lineidx_dict = {}
    fea_tsv = TSVFile(fea_file)
    for i in range(fea_tsv.num_rows()):
        cols = fea_tsv.seek(i)
        bbox = json.loads(cols[1])
        # positive pairs
        if BBOX_POS_PAIR_IDS in bbox:
            for pid in bbox[BBOX_POS_PAIR_IDS]:
                if pid in pos_pair_lineidx_dict:
                    assert(len(pos_pair_lineidx_dict[pid]) == 1)
                    pos_pair_lineidx_dict[pid].append(i)
                else:
                    pos_pair_lineidx_dict[pid] = [i]

        # negative pairs
        if BBOX_NEG_PAIR_IDS in bbox:
            for pid in bbox[BBOX_NEG_PAIR_IDS]:
                if pid in neg_pair_lineidx_dict:
                    assert(len(neg_pair_lineidx_dict[pid]) == 1)
                    neg_pair_lineidx_dict[pid].append(i)
                else:
                    neg_pair_lineidx_dict[pid] = [i]

    truths = []
    scores = []
    for label, pair_lineidx_dict in [(1, pos_pair_lineidx_dict), (0, neg_pair_lineidx_dict)]:
        for pid in pair_lineidx_dict:
            feas = []
            for idx in pair_lineidx_dict[pid]:
                cols = fea_tsv.seek(idx)
                feas.append(from_b64_to_fea(cols[2]))
            assert(len(feas) == 2)
            truths.append(label)
            scores.append(cosine_similarity(feas[0], feas[1]))
    fpr, tpr, _ = roc_curve(truths, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    fig.savefig(outfile)


def from_b64_to_fea(s, normalize=True):
    fea = base64.decodestring(s)
    fea = np.frombuffer(fea, dtype=np.float32)
    if normalize:
        norm = np.linalg.norm(fea)
        fea = fea / norm
    return fea


def cosine_similarity(fea1, fea2):
    return max(0, np.dot(fea1, fea2))


def nms_wrapper(bboxes, iou_threshold=0.45, sort_by_key="conf", nms_type="cls_dep"):
    """
    :param bboxes: a list of bboxes
    :param nms_type: choose from None (no nms), cls_dep, cls_indep
    :return: a list of bboxes after nms
    """
    if not nms_type:
        return bboxes
    assert(nms_type=="cls_dep" or nms_type=="cls_indep")
    after_nms = []
    if nms_type == "cls_dep":
        bbox_dict = collections.defaultdict(list)
        for b in bboxes:
            bbox_dict[b["class"]].append(b)
        for label in bbox_dict:
            for b in nms_wrapper(bbox_dict[label], iou_threshold, sort_by_key, "cls_indep"):
                after_nms.append(b)
    else:
        dets = np.array([b["rect"] + [b[sort_by_key]] for b in bboxes]).astype(np.float32)
        keep = nms(dets, iou_threshold)
        for i in keep:
            after_nms.append(bboxes[i])

    return after_nms


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1].astype(np.int32)

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    keep = []
    for _i in range(ndets):
        # _i: normal index, i: ordered index
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        # the box i currently under consideration
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # variables for computing overlap with box j (lower scoring box)
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep


def prepare_training_data(det_expid, gt_dataset_name, outdir, gt_split="train"):
    # generate region proposal
    detpred_file, _ = yolo_predict(full_expid=det_expid, test_data=gt_dataset_name, test_split=gt_split)

    # merge region proposal and ground truth
    pos_iou = 0.5
    neg_iou = 0.05
    outfile = os.path.join(outdir, "region_proposal/{}.{}.gt_rp.{}_{}.tsv".format(gt_dataset_name, gt_split, pos_iou, neg_iou))
    dataset = TSVDataset(gt_dataset_name)
    merge_gt_rp(dataset.get_data(gt_split, t='label'), detpred_file, outfile, pos_iou, neg_iou)


def merge_gt_rp(gt_file, rp_file, outfile, pos_iou=0.5, neg_iou=0.05):
    """
    Merge ground truth bbox with region proposal bbox
    region proposal is annotated with the corresponding class if IoU>pos_iou,
    annotated as __background if max(IoU)<neg_iou
    """
    gt = DetectionFile(gt_file)
    rp = DetectionFile(rp_file)

    rp_candidates = collections.defaultdict(list)  # imgkey: list of bboxes

    count_class = collections.defaultdict(int)  # class: count
    bg_cands = []  # tuple of imgkey, bbox
    for imgkey in gt:
        gt_bboxes = gt[imgkey]
        rp_bboxes = rp[imgkey]
        for b in rp_bboxes:
            overlaps = np.array([calculate_iou(b["rect"], gtbox["rect"]) for gtbox in gt_bboxes])
            bbox_idx_max = np.argmax(overlaps)
            if overlaps[bbox_idx_max] > pos_iou:
                b["class"] = gt_bboxes[bbox_idx_max]["class"]
                rp_candidates[imgkey].append(b)
                count_class[b["class"]] += 1
            elif overlaps[bbox_idx_max] < neg_iou:
                # background candidate
                b["class"] = BACKGROUND_LABEL
                bg_cands.append((imgkey, b))

        for b in gt_bboxes:
            count_class[b["class"]] += 1

    max_count = max([count_class[c] for c in count_class])
    bg_cands = sorted(bg_cands, key=lambda t: t[1]["obj"], reverse=True)
    # skip top 1% to avoid false negative
    bg_lidx = int(0.01 * len(bg_cands))
    bg_ridx = min(len(bg_cands), int(bg_lidx+max_count*1.5))
    for i in range(bg_lidx, bg_ridx):
        k, b = bg_cands[i]
        rp_candidates[k].append(b)
    print("added #background: {}".format(bg_ridx-bg_lidx))

    num_gt = 0
    num_rp = 0
    num_img = 0
    with open(outfile, 'w') as fout:
        for imgkey in gt:
            num_img += 1
            gt_bboxes = gt[imgkey]

            num_rp += len(rp_candidates[imgkey])
            num_gt += len(gt_bboxes)

            fout.write('\t'.join([imgkey, json.dumps(gt_bboxes + rp_candidates[imgkey])]))
            fout.write('\n')
    print("load #img: {}, #gt: {}, #proposal: {}".format(num_img, num_gt, num_rp))
    outlineidx = outfile.rsplit('.', 1)[0] + ".lineidx"
    if os.path.isfile(outlineidx):
        os.remove(outlineidx)
