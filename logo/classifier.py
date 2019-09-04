import base64
import collections
import copy
import datetime
from deprecated import deprecated
import json
import logging
import math
import numpy as np
import os
import os.path as op
import random
import torch

from sklearn.metrics import roc_curve, auc
import matplotlib
# use a non-interactive backend to generate images without having a window appear
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from logo import constants

from qd_classifier.scripts import extract, pred
from qd_classifier.utils import accuracy
from qd.qd_common import calculate_iou, write_to_yaml_file, load_from_yaml_file, init_logging, ensure_directory, int_rect, is_valid_rect, worth_create
from qd.qd_common import ensure_copy_file, write_to_file
from qd.process_tsv import populate_dataset_details
from qd.tsv_io import TSVDataset, TSVFile, tsv_reader, tsv_writer, reorder_tsv_keys, rm_tsv
from qd import tsv_io
from qd_classifier.utils.prep_data import ensure_populate_dataset_crop_index

class CropTaggingWrapper(object):
    def __init__(self, det_expid, tag_expid, tag_snap_id="snapshot", tag_model_id="model_best.pth.tar", labelmap=None):
        self.det_expid = det_expid
        self._data_folder = "./brand_output/tmp"
        self.eval_dir = "./brand_output/{}/{}/eval".format(tag_expid, tag_snap_id)
        self.labelmap = labelmap
        self.tag_model_path = os.path.join(os.path.dirname(self.eval_dir), tag_model_id)
        assert(os.path.isfile(self.tag_model_path))
        self.log_file = os.path.join(self.eval_dir, "prediction_log.txt")
        ensure_directory(self.eval_dir)
        ensure_directory(os.path.dirname(self.tag_model_path))

        self.num_workers = 4

    def predict_on_known_class(self, dataset_name, split, version=-1,
                region_source=constants.PRED_REGION, conf_from=constants.CONF_OBJ_TAG,
                enlarge_bbox=1.0, eval_topk_acc=None):
        """
        Two stage methods by combining detector with classifier
        Returns file of imgkey, list of bboxes
        """
        # get region proposal
        rp_file = self.get_region_proposal(dataset_name, split,
                region_source=region_source, version=version)

        # get tagging results
        # data_yaml = self._write_data_yaml(dataset_name, split, "test", rp_file, enlarge_bbox=enlarge_bbox)
        max_k = eval_topk_acc if eval_topk_acc else 1
        tag_file = self.tag_predict(dataset_name, split, version, max_k=max_k, enlarge_bbox=enlarge_bbox)
        # tag_file = self.tag_predict_deprecated(data_yaml, max_k=max_k, enlarge_bbox=enlarge_bbox, force_rewrite=True)

        # combine
        outfile = tag_file + ".parsed"
        topk_acc = parse_tagging_predict(tag_file, outfile, conf_from=conf_from,
                eval_accuracy=(eval_topk_acc is not None))

        return outfile, topk_acc, tag_file

    def predict_on_unknown_class(self, dataset_name, split,
                region_source=constants.PRED_REGION, version=-1):
        """
        Predicts on unknown classes by comparing similarity with reference database
        Returns file of imgkey, list of bboxes
        """
        # get region proposal
        rp_file = self.get_region_proposal(dataset_name, split,
                region_source=region_source, version=version)
        if region_source == constants.PRED_REGION:
            top_k_acc = None
        elif region_source == constants.GT_REGION:
            top_k_acc = (1,5)

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
        outfile = os.path.join(self.eval_dir,
                "{}.{}.fea.predict.region.{}.tsv".format(dataset_name, split, region_source))
        outfile_tmp = outfile + '.tmp'
        if worth_create(fea_file, outfile_tmp) or worth_create(gt_fea_file, outfile_tmp):
            acc_str = compare_similarity(gt_fea_file, fea_file, outfile_tmp, top_k_acc=top_k_acc)
            if acc_str:
                with open(self.log_file, 'a+') as fp:
                    fp.write("\nTime: {}\t Method: {}\n" \
                            "Test dataset: {}({})\t RegionProposal: {}\t Prototype database: {}({})\n".format(
                            datetime.datetime.now(), "predict_on_unknown_class",
                            dataset_name, split, rp_file, prototype_dataset, prototype_split))
                    fp.write(acc_str)
                    fp.write('\n')
        # align the order of imgkeys
        ordered_keys = TSVDataset(dataset_name).load_keys(split)
        reorder_tsv_keys(outfile_tmp, ordered_keys, outfile)
        rm_tsv(outfile_tmp)
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
        from qd.yolotrain import yolo_predict
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

    def tag_predict(self, data, split, version, max_k=1, enlarge_bbox=1.0):
        """ Tagging on given images and regions
        output: TSV file of image_key, json bbox(rect, obj), tag:conf list separated by ;
        """
        ensure_populate_dataset_crop_index(data, split, version)
        outpath = op.join(self.eval_dir, '.'.join([data, split, str(version), str(enlarge_bbox)]) + ".tagging.tsv")
        args = [
            '.'.join([data, split, str(version)]),
            "--model", self.tag_model_path,
            "--output", outpath,
            "--topk", str(max_k),
            "--num_workers", str(self.num_workers),
            '--enlarge_bbox', str(enlarge_bbox)
        ]
        pred.main(args)
        return outpath

    def tag_predict_deprecated(self, datayaml, force_rewrite=False, max_k=1, enlarge_bbox=1.0):
        """ Tagging on given images and regions
        output: TSV file of image_key, json bbox(rect, obj), tag:conf list separated by ;
        """
        outpath = "{}.tagging.tsv".format(datayaml.rsplit('.', 1)[0])
        if not force_rewrite and not worth_create(datayaml, outpath) and not worth_create(self.tag_model_path, outpath):
            logging.info("skip tagging, already exists: {}".format(outpath))
            return outpath
        args = [
            datayaml,
            "--model", self.tag_model_path,
            "--output", outpath,
            "--topk", str(max_k),
            "--workers", str(self.num_workers),
            '--enlarge_bbox', str(enlarge_bbox)
        ]
        if self.labelmap:
            args.extend(["--labelmap", self.labelmap,])
        pred.main(args)
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

    def get_region_proposal(self, dataset_name, split, region_source=constants.PRED_REGION, version=-1):
        if region_source == constants.PRED_REGION:
            rp_file = self.det_predict(dataset_name, split)
        elif region_source == constants.GT_REGION:
            d = TSVDataset(dataset_name)
            rp_file = d.get_data(split, t='label', version=version)
        else:
            raise ValueError("Invalid region source: {}".format(region_source))
        return rp_file

    def _write_data_yaml(self, dataset_name, split, session,
            labelfile=None, enlarge_bbox=1.0):
        dataset = TSVDataset(dataset_name)
        data_yaml_info = [dataset_name, split, str(enlarge_bbox)]
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
        data_yaml = os.path.join(self.eval_dir, ".".join(data_yaml_info))
        if os.path.isfile(data_yaml):
            data_config = load_from_yaml_file(data_yaml)
        else:
            data_config = {}

        new_cfg = {
            "tsv": os.path.realpath(dataset.get_data(split)),
        }
        if self.labelmap:
            new_cfg["labelmap"] = self.labelmap
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


def parse_tagging_predict(infile, outfile, nms_type=None, bg_skip=1,
            conf_from=constants.CONF_OBJ_TAG, eval_accuracy=False):
    """ Convert two-stage prediction result to regular format
    Args:
        infile: imgkey, bbox, classification results (term:conf;)
        outfile: imgkey, bbox list
        nms_type: None (no nms), cls_dep, cls_indep
        bg_skip: int, if background is in the first [n] prediction, it is treated as bg
        eval_accuracy: calculate top k accuracy basing on bbox from infile,
                where k is the number of tags provided. Default is no evaluation.
    """
    pred_dict = collections.defaultdict(list) # key: a list of bbox
    correct_counts = None
    num_samples = 0
    for cols in tsv_reader(infile):
        assert(len(cols) == 3)
        num_samples += 1
        key = cols[0]
        bbox = json.loads(cols[1])
        label_conf_pairs = [p.rsplit(':', 1) for p in cols[2].split(';')]

        # decide if the prediction is background
        is_bg = False
        for i, (label, conf) in enumerate(label_conf_pairs):
            if i >= bg_skip:
                break
            if label == constants.BACKGROUND_LABEL:
                is_bg = True
                break

        if eval_accuracy:
            if correct_counts is None:
                correct_counts = [0] * len(label_conf_pairs)
            gt_label = bbox["class"]
            for i, pair in enumerate(label_conf_pairs):
                if pair[0] == gt_label:
                    correct_counts[i] += 1
                    break

        # use top1 classification prediction as label
        bbox["class"] = label_conf_pairs[0][0]
        # use classification score * obj as conf score
        if conf_from == constants.CONF_OBJ_TAG:
            if 'obj' in bbox:
                obj = bbox['obj']
            elif 'conf' in bbox:
                obj = bbox['conf']
            else:
                obj = 1.0
            bbox["conf"] = float(label_conf_pairs[0][1]) * obj
        elif conf_from == constants.CONF_TAG:
            bbox["conf"] = float(label_conf_pairs[0][1])
        elif conf_from == constants.CONF_OBJ:
            bbox["conf"] = bbox["obj"]
        else:
            raise ValueError("invalid confidence type: {}".format(conf_from))

        if not is_bg:
            pred_dict[key].append(bbox)
        else:
            if key not in pred_dict:
                pred_dict[key] = []
    if nms_type:
        for key in pred_dict:
            pred_dict[key] = nms_wrapper(pred_dict[key], nms_type=nms_type)

    tsv_writer([[k, json.dumps(pred_dict[k])] for k in pred_dict], outfile)

    if eval_accuracy:
        if len(correct_counts) > 1:
            for i in range(1, len(correct_counts)):
                correct_counts[i] += correct_counts[i-1]
        return [float(num_correct)/num_samples for num_correct in correct_counts]
    else:
        return None


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

def ensure_populate_dataset_crop_index(data, split, version, num_min_samples=0):
    from qd.qd_common import int_rect

    dataset = TSVDataset(data)
    outfile = dataset.get_data(split, t='crop.index', version=version)
    # if op.isfile(outfile):
    #     return

    hw_iter = dataset.iter_data(split, t='hw')
    label_iter = dataset.iter_data(split, t='label', version=version)
    label_to_indices = collections.defaultdict(list)

    for img_idx, (parts1, parts2) in enumerate(zip(hw_iter, label_iter)):
        assert parts1[0] == parts2[0]
        im_h, im_w = parts1[1].split(' ')
        for bbox_idx, bbox in enumerate(json.loads(parts2[1])):
            rect = int_rect(bbox["rect"], im_h=int(im_h), im_w=int(im_w))
            left, top, right, bot = rect
            if right - left >= 3 and bot - top >= 3:
                label_to_indices[bbox["class"]].append([img_idx, bbox_idx])
            else:
                logging.info("invalid bbox at data:{} split:{} version:{} image:{} bbox:{}"
                    .format(data, split, version, parts2[0], json.dumps(bbox)))
    all_indices = []
    for label in label_to_indices:
        cur_indices = label_to_indices[label]
        num_copies = 1
        if len(cur_indices) < num_min_samples:
            num_copies = int(math.ceil(num_min_samples / len(cur_indices)))
        for _ in range(num_copies):
            all_indices.extend(cur_indices)

    random.seed(6)
    random.shuffle(all_indices)
    tsv_writer(all_indices, outfile)

def build_balanced_crop_index(all_src_data_info,
            out_data, out_split="train", num_min_samples=0):
    out_dataset = TSVDataset(out_data)

    out_labelmap = []
    out_labelmap_set = set()
    out_imgs = []
    out_labels = []
    out_hws = []
    out_shuffle = []
    for src_idx, (src_data, src_split, src_version) in enumerate(all_src_data_info):
        populate_dataset_details(src_data)
        src_dataset = TSVDataset(src_data)

        cur_labels = [p[0] for p in src_dataset.iter_data(src_split, t='labelmap', version=src_version)]
        for label in cur_labels:
            if label in out_labelmap_set:
                continue
            out_labelmap.append(label)
            out_labelmap_set.add(label)
        out_imgs.append(src_dataset.get_data(src_split))
        out_hws.append(src_dataset.get_data(src_split, 'hw'))
        out_labels.append(src_dataset.get_data(src_split, 'label', version=src_version))
        for label_idx in range(len(TSVFile(src_dataset.get_data(src_split)))):
            out_shuffle.append([src_idx, label_idx])

    tsv_writer([[p] for p in out_labelmap], out_dataset.get_data(out_split, t='labelmap'))
    tsv_writer([[p] for p in out_imgs], out_dataset.get_data(out_split + 'X'))
    tsv_writer([[p] for p in out_hws], out_dataset.get_data(out_split + 'X', 'hw'))
    tsv_writer([[p] for p in out_labels], out_dataset.get_data(out_split + 'X', 'label'))
    tsv_writer(out_shuffle, out_dataset.get_shuffle_file(out_split))

    ensure_populate_dataset_crop_index(out_data, out_split, 0, num_min_samples=num_min_samples)
