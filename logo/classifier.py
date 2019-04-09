import base64
import collections
import copy
import datetime
import json
import logging
import math
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
from qd.qd_common import calculate_iou, write_to_yaml_file, load_from_yaml_file, init_logging, worth_create, ensure_directory, int_rect, is_valid_rect
from evaluation.eval_utils import DetectionFile
from logo import constants
from qd.tsv_io import TSVDataset, TSVFile, tsv_reader, tsv_writer
from qd import tsv_io
try:
    from scripts.yolotrain import yolo_predict
    from evaluation import dataproc
except ImportError:
    pass

class CropTaggingWrapper(object):
    def __init__(self, det_expid, tag_expid, tag_snap_id="snapshot"):
        self.det_expid = det_expid
        self._data_folder = "data/brand_output/"
        self._rootpath = "data/brand_output/{}/classifier/{}".format(det_expid, tag_expid)
        self.labelmap = os.path.join(self._rootpath, "labelmap.txt")
        self.tag_model_path = os.path.join(self._rootpath, "{}/model_best.pth.tar".format(tag_snap_id))
        self.log_file = os.path.join(self._rootpath, "eval/prediction_log.txt")
        ensure_directory(self._rootpath)
        ensure_directory(os.path.dirname(self.tag_model_path))
        ensure_directory(os.path.dirname(self.log_file))

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
        # align the order of imgkeys
        dataproc.align_detection(dataset_name, split, outfile)
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
        prototype_dataset = "logo40can2"
        prototype_split = "train"
        proto_tsv = TSVDataset(prototype_dataset)
        data_yaml = self._write_data_yaml(prototype_dataset, prototype_split, "test",
                labelfile=proto_tsv.get_data(prototype_split, 'label', version=-1))
        gt_fea_file = self.extract_feature(data_yaml)

        # compare similarity
        outfile = os.path.join(self._rootpath,
                "{}.{}.fea.predict.region.{}.tsv".format(dataset_name, split, region_source))
        if worth_create(fea_file, outfile) or worth_create(gt_fea_file, outfile):
            acc_str = compare_similarity(gt_fea_file, fea_file, outfile, top_k_acc=top_k_acc)
            if acc_str:
                with open(self.log_file, 'a+') as fp:
                    fp.write("\nTime: {}\t Method: {}\n" \
                            "Test dataset: {}({})\t RegionProposal: {}\t Prototype database: {}({})\n".format(
                            datetime.datetime.now(), "predict_on_unknown_class",
                            dataset_name, split, rp_file, prototype_dataset, prototype_split))
                    fp.write(acc_str)
                    fp.write('\n')
        # align the order of imgkeys
        dataproc.align_detection(dataset_name, split, outfile)
        return outfile

    def compare_pairs(self, dataset_name, split):
        """
        Internal metric for featurizer, calculate ROC for positive and negative pairs
        """
        # TODO: configure constant
        rp_file = "/raid/data/logo40_pair/test.label.tsv"
        data_yaml = self._write_data_yaml(dataset_name, split, "test", rp_file)
        fea_file = self.extract_feature(data_yaml)

        outfile = os.path.join(os.path.dirname(self.log_file), "{}.{}.pair.roc.png".format(dataset_name, split))
        if worth_create(fea_file, outfile):
            compare_pair_features(fea_file, outfile)
        return outfile

    def det_predict(self, dataset_name, split):
        pred_file, _ = yolo_predict(full_expid=self.det_expid, test_data=dataset_name, test_split=split)
        return pred_file

    def eval_classification(self, dataset_name, split, rp_file, topk=(1, 5)):
        num_bboxes_per_img = 0
        num_classes_per_img = 0

        data_yaml = self._write_data_yaml(dataset_name, split, "test", rp_file)
        tag_file = self.tag_predict(data_yaml, max_k=max(max(topk), 10), force_rewrite=False)
        all_classes = [p[0] for p in tsv_reader(self.labelmap)]
        class2idx = {p: i for i, p in enumerate(all_classes)}
        num_classes = len(all_classes)

        all_pred = []    # num_samples * num_classes
        all_target = []  # num_samples * num_classes, multi-label
        is_multi_label = False
        all_imgkey = set()
        for cols in tsv_reader(tag_file):
            imgkey = cols[0]
            all_imgkey.add(imgkey)
            cur_pred = [0]*num_classes
            cur_target = [0]*num_classes
            gt_box = json.loads(cols[1])
            # multi label or single label
            if "classes" in gt_box:
                gt_classes = gt_box["classes"]
                is_multi_label = True
            else:
                gt_classes = [gt_box["class"]]
            num_bboxes_per_img += 1
            num_classes_per_img += len(gt_classes)

            for c in gt_classes:
                cur_target[class2idx[c]] = 1.0

            for it in cols[-1].split(';'):
                c, conf = it.split(':')
                cur_pred[class2idx[c]] = float(conf)
            all_pred.append(cur_pred)
            all_target.append(cur_target)

        num_imgs = len(all_imgkey)
        num_bboxes_per_img /= float(num_imgs)
        num_classes_per_img /= float(num_imgs)

        if is_multi_label:
            # multi label
            acc = accuracy.MultiLabelAccuracy()
            acc.calc(torch.tensor(all_pred), torch.tensor(all_target))
            acc_str = acc.result_str()
        else:
            # single label
            acc = accuracy.SingleLabelAccuracy(topk)
            for pred, target in zip(all_pred, all_target):
                target_indices = [i for i, t in enumerate(target) if t==1]
                for t in target_indices:
                    acc.calc(torch.tensor([pred]), torch.tensor([[t]]))
            acc_str = acc.result_str()

        log_str = "\nTime: {}\t Method: {}\n" \
                "Test dataset: {}({})\t RegionProposal: {}\t" \
                "is_multi_label: {}, #bboxs per img: {}, #classes per img: {}\n".format(
                datetime.datetime.now(), "classification",
                dataset_name, split, rp_file, is_multi_label, num_bboxes_per_img, num_classes_per_img)
        log_str += acc_str
        logging.info(log_str)
        with open(self.log_file, 'a+') as fp:
            fp.write(log_str)
            fp.write('\n')


    def tag_predict(self, datayaml, force_rewrite=False, max_k=1):
        """ Tagging on given images and regions
        output: TSV file of image_key, json bbox(rect, obj), tag:conf list separated by ;
        """
        outpath = "{}.tagging.tsv".format(datayaml.rsplit('.', 1)[0])
        if not force_rewrite and not worth_create(datayaml, outpath):
            logging.info("skip tagging, already exists: {}".format(outpath))
            return outpath
        pred.main([
            datayaml,
            "--model", self.tag_model_path,
            "--output", outpath,
            "--labelmap", self.labelmap,
            "--topk", str(max_k),
            "--workers", str(self.num_workers)
        ])
        return outpath

    def extract_feature(self, datayaml, force_rewrite=False):
        """ Features on given images and regions
        output: TSV file of image_key, json bbox(rect, obj), b64_string of np.float32 array
        """
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

    def whole_image_region(self, dataset_name, split, is_multi_label=False):
        # set the proposed region as the whole image
        rp_file = os.path.join(self._data_folder, "{}.{}.whole_img_region.tsv".format(dataset_name, split))
        dataset = TSVDataset(dataset_name)
        if worth_create(dataset.get_data(split), rp_file) \
                or worth_create(dataset.get_data(split, 'label', version=-1), rp_file):
            label_iter = dataset.iter_data(split, 'label', version=-1)
            def gen_labels():
                for imgkey, hw in dataset.iter_data(split, "hw"):
                    h, w = hw.split(' ')
                    k, coded_rects = label_iter.next()
                    assert(k == imgkey)
                    bboxes = json.loads(coded_rects)
                    classes = set(b["class"] for b in bboxes)
                    if is_multi_label:
                        label = [{"rect": [0, 0, int(w), int(h)], "classes": list(classes)}]
                    else:
                        # single label per rect, rect value can be same
                        label = []
                        for c in classes:
                            label.append({"rect": [0, 0, int(w), int(h)], "class": c})
                    yield imgkey, json.dumps(label, separators=(',', ':'))
            tsv_writer(gen_labels(), rp_file)
        return os.path.realpath(rp_file)

    def _write_data_yaml(self, dataset_name, split, session, labelfile=None):
        dataset = TSVDataset(dataset_name)
        data_yaml_info = [dataset_name, split]
        if labelfile:
            # check key orders, the image key must match in image file and label file
            ordered_keys = []
            for key, _, _ in dataset.iter_data(split):
                ordered_keys.append(key)
            label_tsv = tsv_io.TSVFile(labelfile)
            keys = [label_tsv.seek_first_column(i) for i in range(len(label_tsv))]
            if len(ordered_keys)!=len(keys) or any([i!=j for i, j in zip(ordered_keys, keys)]):
                key_to_idx = {key: i for i, key in enumerate(keys)}
                def gen_rows():
                    for key in ordered_keys:
                        if key in key_to_idx:
                            idx = key_to_idx[key]
                            yield label_tsv.seek(idx)
                        else:
                            yield [key, "[]"]
                new_labelfile = labelfile+".ordered"
                tsv_writer(gen_rows(), new_labelfile)
                labelfile = new_labelfile

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
            if label == constants.BACKGROUND_LABEL:
                is_bg = True
                break
        if is_bg:
            continue

        # use classification label as label, classification score * obj as conf score
        bbox["class"] = label_conf_pair[0][0]
        bbox["conf"] = float(label_conf_pair[0][1]) * bbox["obj"]

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
        if constants.BBOX_POS_PAIR_IDS in bbox:
            for pid in bbox[constants.BBOX_POS_PAIR_IDS]:
                if pid in pos_pair_lineidx_dict:
                    assert(len(pos_pair_lineidx_dict[pid]) == 1)
                    pos_pair_lineidx_dict[pid].append(i)
                else:
                    pos_pair_lineidx_dict[pid] = [i]

        # negative pairs
        if constants.BBOX_NEG_PAIR_IDS in bbox:
            for pid in bbox[constants.BBOX_NEG_PAIR_IDS]:
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


def prepare_training_data(det_expid, gt_dataset_name, outdir, gt_split="train",
            version=0, pos_iou=0.5, neg_iou=(0.1, 0.3), enlarge_bbox=2.5):
    """
    Merge ground truth bbox with region proposal bbox
    region proposal is annotated with the corresponding class if IoU>pos_iou,
    annotated as __background if max(IoU)<neg_iou
    """
    # generate region proposal
    detpred_file, _ = yolo_predict(full_expid=det_expid, test_data=gt_dataset_name, test_split=gt_split)

    # merge region proposal and ground truth
    outfile = os.path.join(outdir, "region_proposal/{}.{}.gt_rp.{}_{}_{}.tsv".format(
            gt_dataset_name, gt_split, pos_iou, neg_iou[0], neg_iou[1]))
    gt_dataset = TSVDataset(gt_dataset_name)

    def get_hw(key):
        parts = gt_dataset.seek_by_key(key, gt_split, t="hw")
        assert(len(parts) == 2)
        assert(parts[0] == key)
        nums = parts[1].split(' ')
        assert(len(nums) == 2)
        return int(nums[0]), int(nums[1])

    rp = DetectionFile(detpred_file)
    rp_candidates = collections.defaultdict(list)  # imgkey: list of bboxes
    class2count = collections.defaultdict(int)  # class: count
    bg_cands = []  # tuple of imgkey, bbox
    num_gt = 0
    num_total_regions = 0

    for imgkey, coded_rects in gt_dataset.iter_data(gt_split, t='label', version=version):
        # HACK
        if imgkey == "http://www.mimifroufrou.com/scentedsalamander/images/Le-Male-2009-Billboard-B.jpg":
            continue

        im_h, im_w = get_hw(imgkey)
        gt_bboxes = json.loads(coded_rects)
        num_gt += len(gt_bboxes)
        for idx in range(len(gt_bboxes)):
            cur_bbox = copy.deepcopy(gt_bboxes[idx])
            cur_bbox["rect"] = int_rect(cur_bbox["rect"], enlarge_factor=1.0, im_h=im_h, im_w=im_w)
            enlarged_rect = int_rect(cur_bbox["rect"], enlarge_factor=enlarge_bbox, im_h=im_h, im_w=im_w)
            overlaps = [calculate_iou(enlarged_rect, gtbox["rect"]) for i, gtbox in enumerate(gt_bboxes) if i!=idx]
            # enlarge bbox only if it does not overlap other boxes
            if len(overlaps) > 0 and max(overlaps) < neg_iou[0]:
                cur_bbox["rect"] = enlarged_rect
            if not is_valid_rect(cur_bbox["rect"]):
                print("invalid rect")
                continue
            rp_candidates[imgkey].append(cur_bbox)
            num_total_regions += 1

        rp_bboxes = rp[imgkey]
        for b in rp_bboxes:
            new_rect = int_rect(b["rect"], enlarge_factor=enlarge_bbox, im_h=im_h, im_w=im_w)
            b["rect"] = new_rect
            if not is_valid_rect(b["rect"]):
                continue
            overlaps = [calculate_iou(new_rect, gtbox["rect"]) for gtbox in gt_bboxes]
            sorted_overlaps = sorted([(i, v) for i, v in enumerate(overlaps)], key=lambda t: t[1], reverse=True)
            if len(overlaps) == 0:
                continue
            max_iou_idx, max_iou = sorted_overlaps[0]

            if max_iou >= neg_iou[0] and max_iou <= neg_iou[1]:
                # background candidate
                b["class"] = constants.BACKGROUND_LABEL
                bg_cands.append((imgkey, b))
            elif max_iou > pos_iou:
                if len(sorted_overlaps)>1 and sorted_overlaps[1][1]>pos_iou:
                    # skip if the region covers >1 instances
                    continue
                b["class"] = gt_bboxes[sorted_overlaps[0][0]]["class"]
                rp_candidates[imgkey].append(b)
                num_total_regions += 1

    bg_cands = sorted(bg_cands, key=lambda t: t[1]["obj"], reverse=True)
    # skip top 1% to avoid false negative
    bg_lidx = int(0.01 * len(bg_cands))
    bg_ridx = min(len(bg_cands), int(bg_lidx+num_total_regions*2))
    for i in range(bg_lidx, bg_ridx):
        k, b = bg_cands[i]
        rp_candidates[k].append(b)
    print("added #background: {}, #gt: {}, #proposal: {}".format(bg_ridx-bg_lidx, num_gt, num_total_regions-num_gt))

    def gen_labels():
        for imgkey, coded_rects in gt_dataset.iter_data(gt_split, t='label', version=version):
            yield imgkey, json.dumps(rp_candidates[imgkey], separators=(',', ':'))

    tsv_writer(gen_labels(), outfile)
    return outfile
