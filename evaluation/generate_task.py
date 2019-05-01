import base64
import collections
import copy
import cv2
from deprecated import deprecated
import json
import logging
import math
import numpy as np
import os
from tqdm import tqdm
import uuid

import evaluation.utils
from evaluation.utils import read_from_file, write_to_file, escape_json_obj, calculate_bbox_area
from qd.process_tsv import get_img_url2
from qd import deteval
from qd.tsv_io import TSVFile, TSVDataset, tsv_writer
from qd import process_tsv

OPT_POS = 1
OPT_NEG = 2

# required fields in task file
CLASS_DISPLAY_NAME_KEY = "objects_to_find"
URL_KEY = "image_url"
BBOXES_KEY = "bboxes"               # json list of dict

# optional fields in task file
QUESTION_TYPE_KEY = "question_type" # default is VerifyBox
HP_KEY = "expected_output"
INFO_KEY = "image_info"             # used to store self-defined info, e.g., id of bbox_task
UUID_KEY = "uuid"                   # auto-generated for each question

class HoneyPotGenerator(object):
    """ Generate honey pot from ground truth or honey pot data
    """
    def __init__(self, data, data_type, hp_neg_prob=0.5,
                 class_candidates=None):
        """
        Params:
            data: list of dict, each dict is bbox label
            data_type: indicate if the label is in gt or hp format
            hp_neg_prob: float, the prob that the honey pot is negative
            class_candidates: iterable, from which negative samples are chosen
        """
        self._data = data
        if data_type not in ["gt", "hp"]:
            raise Exception("invalid data type: {}".format(data_type))
        self._data_type = data_type
        self._count = len(data)
        self._cur_idx = np.random.randint(0, self._count)
        if self._count == 0:
            raise Exception("data cannot be empty")
        self.hp_neg_prob = hp_neg_prob

        if not class_candidates:
            class_candidates = set()
        else:
            class_candidates = set(class_candidates)
        for d in self._data:
            class_candidates.add(d[CLASS_DISPLAY_NAME_KEY])
        self.class_candidates = list(class_candidates)

    def next(self):
        self._cur_idx = (self._cur_idx + 1) % self._count
        if self._data_type == "hp":
            # return it directly if data is alreay in honey pot format
            return self._data[self._cur_idx]
        else:
            # convert gt label to honey pot format
            hp = copy.deepcopy(self._data[self._cur_idx])
            if self.hp_neg_prob > 0 and np.random.rand() < self.hp_neg_prob:
                false_class = hp[CLASS_DISPLAY_NAME_KEY]
                while evaluation.utils.is_same_class(false_class, hp[CLASS_DISPLAY_NAME_KEY]):
                    false_class = np.random.choice(self.class_candidates, size=1)[0]
                hp[CLASS_DISPLAY_NAME_KEY] = false_class
                hp[HP_KEY] = OPT_NEG
            else:
                hp[HP_KEY] = OPT_POS
            return hp


def pack_task_with_honey_pot(task_data, hp_data, hp_type, num_tasks_per_hit,
                             num_hp_per_hit, hp_neg_prob=0.5, multiple_tasks_per_hit=True):
    ''' Generate hits composed of real tasks and honey pot
    '''
    output_content = []
    num_total_task = len(task_data)
    if num_hp_per_hit > 0:
        class_candidates = set(t[CLASS_DISPLAY_NAME_KEY] for t in task_data)
        hp_gen = HoneyPotGenerator(hp_data, hp_type, hp_neg_prob=hp_neg_prob, class_candidates=class_candidates)
    else:
        hp_gen = None

    if multiple_tasks_per_hit:
        # packed UI: each hit shows multiple tasks
        for start in np.arange(0, num_total_task, num_tasks_per_hit):
            end = min(start + num_tasks_per_hit, num_total_task)
            line = task_data[start: end]
            for _ in range(num_hp_per_hit):
                line.append(hp_gen.next())
            if num_hp_per_hit > 0:
                np.random.shuffle(line)
            output_content.append(line)
    else:
        # each hit shows only one task
        ratio = int(num_tasks_per_hit / num_hp_per_hit) if num_hp_per_hit>0 else num_tasks_per_hit
        for start in np.arange(0, num_total_task, ratio):
            end = min(start + ratio, num_total_task)
            output_content.extend(task_data[start: end])
            if hp_gen:
                output_content.append(hp_gen.next())
    return output_content


def write_task_file(data, filepath):
    """ Add header and write tasks to file for uploading
    """
    with open(filepath + ".tmp", 'w') as fout:
        fout.write('input_content')
        fout.write('\n')
        for line in data:
            fout.write(escape_json_obj(line))
            fout.write('\n')
    if os.path.isfile(filepath):
        os.remove(filepath)
    os.rename(filepath + ".tmp", filepath)


def generate_task_files(task_type, label_file, hp_file, outbase,
                        hp_type, num_tasks_per_hit=10, num_hp_per_hit=2,
                        num_hits_per_file=2000):
    """
    Params:
        label_file: tsv file containing labels to be verified,
            columns: image_info, json list of bboxes, image_url
        hp_file: txt file containing honey pot labels, used to
            detect bad workers, each line is a question(json dict)
        outfile: one column of "input_content", used to submit
            UHRS/MTurk tasks
    """

    if task_type == "VerifyImage":
        box_per_img="one"
    elif task_type == "VerifyBox":
        box_per_img="one"
    elif task_type == "VerifyCover":
        box_per_img="class"
    elif task_type == "DrawBox":
        box_per_img="class"
    else:
        raise Exception("invalid task type: {}".format(task_type))

    return _generate_task_files_helper(task_type, label_file, hp_file, outbase, hp_type,
                                description_file=None, num_tasks_per_hit=num_tasks_per_hit,
                                num_hp_per_hit=num_hp_per_hit, hp_neg_prob=0.5,
                                box_per_img=box_per_img, num_hits_per_file=num_hits_per_file)

@deprecated(reason="no longer generate honey pot on the fly to avoid class collision, e.g., cat vs animal")
def generate_box_honeypot(dataset_name, imgfiles, labelfiles, outfile=None, easy_area_thres=0.2):
    """ Load ground truth data from specified dataset, split, version.
        The bbox is considered easy for human to verify if:
        bbox_area / image_area > easy_area_thres
    """
    hp_data = []
    num_total_bboxes = 0
    num_easy_bboxes = 0
    for imgfile, labelfile in zip(imgfiles, labelfiles):
        imgtsv = TSVFile(imgfile)
        labeltsv = TSVFile(labelfile)
        assert(imgtsv.num_rows() == labeltsv.num_rows())

        for i in range(imgtsv.num_rows()):
            key, _, coded_image = imgtsv.seek(i)
            lkey, coded_rects = labeltsv.seek(i)
            assert(key == lkey)
            img = base64.b64decode(coded_image)
            npimg = np.fromstring(img, dtype=np.uint8)
            source = cv2.imdecode(npimg, 1)
            h, w, c = source.shape
            bboxes = json.loads(coded_rects)
            num_total_bboxes += len(bboxes)
            image_url = get_img_url2(key)
            for bbox in bboxes:
                bbox["rect"] = evaluation.utils.truncate_rect(bbox["rect"], h, w)
                if not evaluation.utils.is_valid_bbox(bbox):
                    continue
                bbox_area = calculate_bbox_area(bbox)
                if float(bbox_area) / (h*w) > easy_area_thres:
                    num_easy_bboxes += 1
                    term = bbox["class"]
                    hp_data.append(
                        {INFO_KEY: dataset_name, URL_KEY: image_url,
                            CLASS_DISPLAY_NAME_KEY: term,
                            BBOXES_KEY: [bbox]})

    print("#total bboxes: {:d}\t#easy bboxes:{:d}".format(num_total_bboxes,
                                                          num_easy_bboxes))
    if outfile:
        with open(outfile, 'w') as fout:
            for d in hp_data:
                fout.write(json.dumps(d))
                fout.write('\n')
    return hp_data

def generate_verify_box_hp(gt_dataset_name, split, version, outfile=None,
        min_bbox_area_thres=0.08, neg_iou_thres=0.2, replace_label=None):
    all_hp_data = []
    process_tsv.populate_dataset_details(gt_dataset_name, check_image_details=True)
    dataset = TSVDataset(gt_dataset_name)

    def get_url(key):
        parts = dataset.seek_by_key(key, split, t="key.url")
        assert(parts[0] == key)
        return parts[1]
    def get_hw(key):
        parts = dataset.seek_by_key(key, split, t="hw")
        assert(parts[0] == key)
        h, w = parts[1].split(' ')
        return int(h), int(w)

    all_labels = [l[0] for l in dataset.iter_data(split, 'labelmap', version=version)]

    for key, coded_rects in tqdm(dataset.iter_data(split, 'label', version=version)):
        gt_rects = json.loads(coded_rects)
        if len(gt_rects) == 0:
            continue
        cur_all_labels = [b["class"] for b in gt_rects]
        im_h, im_w = get_hw(key)
        url = get_url(key)
        num_correct = len(gt_rects)
        num_wrong_loc = int(np.random.uniform(0.2, 2) * num_correct) + 1
        num_wrong_label = int(np.random.uniform(0.5, 5) * num_correct) + 1
        neg_rects = []
        neg_rects.extend(gen_bbox_wrong_label(num_wrong_label, gt_rects, all_labels))
        neg_rects.extend(gen_bbox_wrong_loc(num_wrong_loc, gt_rects, im_h, im_w,
                    neg_iou_thres=neg_iou_thres, min_bbox_area_thres=min_bbox_area_thres,
                    all_labels=cur_all_labels))

        cur_hp_list = [bbox_to_hp(b, url, info=gt_dataset_name,
                        is_neg=False, display_name=replace_label)
                        for b in gt_rects]
        cur_hp_list.extend([bbox_to_hp(b, url, info=gt_dataset_name,
                        is_neg=True, display_name=replace_label)
                        for b in neg_rects])
        np.random.shuffle(cur_hp_list)
        all_hp_data.extend(cur_hp_list)

    if outfile:
        tsv_writer([[json.dumps(d)] for d in all_hp_data], outfile)
    return all_hp_data

def gen_bbox_wrong_label(num_target, gt_rects, all_labels):
    all_res = []
    num_gt = len(gt_rects)
    for i in range(num_target):
        cur_bbox = copy.deepcopy(gt_rects[i % num_gt])
        true_class = cur_bbox["class"]
        false_class = true_class
        while evaluation.utils.is_same_class(false_class, true_class):
            false_class = np.random.choice(all_labels, size=1)[0]
        cur_bbox["class"] = false_class
        cur_bbox["from"] = "hp_wrong_label"
        all_res.append(cur_bbox)
    return all_res

def gen_bbox_wrong_loc(num_target, gt_rects, im_h, im_w,
        neg_iou_thres, min_bbox_area_thres, all_labels):
    all_res = []
    all_labels_set = set(all_labels)
    while len(all_res) < num_target:
        cur_box = gen_random_box(im_h, im_w, area_thres=min_bbox_area_thres)
        overlap_labels = set()
        for gt_box in gt_rects:
            if deteval.IoU(gt_box["rect"], cur_box) > neg_iou_thres:
                overlap_labels.add(gt_box["class"])
        non_overlap_labels = list(all_labels_set - overlap_labels)
        if len(non_overlap_labels) > 0:
            all_res.append({"class": np.random.choice(non_overlap_labels),
                    "rect": cur_box, "from": "hp_wrong_loc"})
    return all_res

def gen_random_box(im_h, im_w, area_thres=0.08, min_pixel=5):
    if im_h <= min_pixel or im_w <= min_pixel or im_h*im_w*area_thres <= min_pixel*min_pixel:
        return None
    while True:
        x = np.random.randint(0, im_w - min_pixel)
        y = np.random.randint(0, im_h - min_pixel)
        max_h = im_h - y
        min_w = max(min_pixel, math.ceil(area_thres * im_w * im_h / max_h))
        max_w = im_w - x
        if min_w >= max_w:
            continue
        w = np.random.randint(min_w, max_w + 1)
        min_h = max(min_pixel, math.ceil(area_thres * im_w * im_h / w))
        h = np.random.randint(min_h, max_h + 1)
        return [x, y, x+w, y+h]

def bbox_to_hp(bbox, image_url, info=None, is_neg=False, display_name=None):
    assert('class' in bbox and "rect" in bbox)
    if not display_name:
        display_name = bbox["class"]
    hp = {URL_KEY: image_url, CLASS_DISPLAY_NAME_KEY: display_name, BBOXES_KEY: [bbox]}
    if info:
        hp[INFO_KEY] = info
    exp_ans = OPT_NEG if is_neg else OPT_POS
    hp[HP_KEY] = exp_ans
    return hp

def generate_tag_honeypot(dataset_name, split, outfile=None):
    dataset = TSVDataset(dataset_name)
    hp_data = []
    all_cls = set(dataset.load_labelmap())
    for imgkey, coded_rects, coded_img in dataset.iter_data(split):
        image_url = get_img_url2(imgkey)
        pos_tags = set(b["class"] for b in json.loads(coded_rects))
        neg_tags = all_cls - pos_tags
        neg_tags = np.random.choice(list(neg_tags), size=min(len(pos_tags), len(neg_tags)))
        for tag in pos_tags:
            hp_data.append({INFO_KEY: dataset_name, URL_KEY: image_url,
                            CLASS_DISPLAY_NAME_KEY: tag, HP_KEY: OPT_POS,
                            QUESTION_TYPE_KEY: "VerifyImage"})
        for tag in neg_tags:
            hp_data.append({INFO_KEY: dataset_name, URL_KEY: image_url,
                            CLASS_DISPLAY_NAME_KEY: tag, HP_KEY: OPT_NEG,
                            QUESTION_TYPE_KEY: "VerifyImage"})
    if outfile:
        tsv_writer([[json.dumps(d)] for d in hp_data], outfile)
    return hp_data


def generate_draw_box_honeypot(dataset_name, split, outfile=None, version=0,
                               max_boxes=5, easy_area_thres=0.05):
    dataset = TSVDataset(dataset_name)
    hp_data = []
    img_hw_dict = {}
    for cols in dataset.iter_data(split, 'hw'):
        imgkey = cols[0]
        h, w = cols[1].split(' ')
        img_hw_dict[imgkey] = (int(h), int(w))
    num_total_candidates = 0
    for imgkey, coded_rects in dataset.iter_data(split, 'label', version=version):
        image_url = get_img_url2(imgkey)
        im_h, im_w = img_hw_dict[imgkey]
        class2bboxes = collections.defaultdict(list)
        for b in json.loads(coded_rects):
            class2bboxes[b["class"]].append(b)
        num_total_candidates += len(class2bboxes)
        for term in class2bboxes:
            bboxes = class2bboxes[term]
            # too many boxes on one image
            if len(bboxes) > max_boxes:
                continue
            is_hard = False
            for b in bboxes:
                if is_hard:
                    break
                b["rect"] = evaluation.utils.truncate_rect(b["rect"], im_h, im_w)
                if not evaluation.utils.is_valid_bbox(b):
                    is_hard = True
                else:
                    b_area = evaluation.utils.calculate_bbox_area(b)
                    if b_area / float(im_h*im_w) <= easy_area_thres:
                        is_hard = True
            if not is_hard:
                hp_data.append({CLASS_DISPLAY_NAME_KEY: term, HP_KEY: bboxes, URL_KEY: image_url})
    logging.info("#total candidates: {}\t #easy: {}".format(num_total_candidates, len(hp_data)))
    if outfile:
        tsv_writer([[json.dumps(d)] for d in hp_data], outfile)
    return hp_data


def _generate_task_files_helper(task_type, label_file, hp_file, outbase, hp_type,
                                description_file, num_tasks_per_hit,
                                num_hp_per_hit, hp_neg_prob, box_per_img, num_hits_per_file):
    """
    Params:
        task_type: choose from VerifyBox, VerifyImage
        label_file: tsv file containing labels to be verified,
            columns: image_info, json list of bboxes, image_url
        hp_file: txt file containing honey pot labels, used to
            detect bad workers, each line is a question(json dict)
        outfile: one column of "input_content", used to submit
            UHRS/MTurk tasks
        description_file: tsv file describing the terms, used to
            change display name. columns: term, description
        box_per_img: choose from one, class, all to 1) show one box per image
            for verification; 2) show boxes of the same class per image; 3) show
            all boxes on the image
    """
    # load terms and descriptions
    term_description_map = None
    if description_file:
        term_description_map = {}
        for parts in read_from_file(description_file, sep='\t'):
            if parts[0] in term_description_map:
                raise Exception("duplicate term: {}, description: {}; {}"
                                .format(parts[0], parts[1],
                                        term_description_map[parts[0]]))
            term_description_map[parts[0]] = parts[1]

    # load task labels to verify
    logging.info("load task labels from {}".format(label_file))
    term_count = collections.defaultdict(int)
    task_data = []
    for parts in read_from_file(label_file, sep='\t', check_num_cols=3):
        bbox_list = json.loads(parts[1])
        image_url = parts[2]
        term_bbox_map = collections.defaultdict(list)
        for bbox in bbox_list:
            term = bbox['class']
            if term_description_map and term in term_description_map:
                term = term_description_map[term]
            term_count[term] += 1
            term_bbox_map[term].append(bbox)
        if box_per_img == "one":
            for term in term_bbox_map:
                for bbox in term_bbox_map[term]:
                    task_data.append({UUID_KEY: str(uuid.uuid4()), INFO_KEY: parts[0],
                              QUESTION_TYPE_KEY: task_type, URL_KEY: image_url,
                              CLASS_DISPLAY_NAME_KEY: term, BBOXES_KEY: [bbox]})
        elif box_per_img == "class":
            for term in term_bbox_map:
                task_data.append({UUID_KEY: str(uuid.uuid4()), INFO_KEY: parts[0],
                              QUESTION_TYPE_KEY: task_type, URL_KEY: image_url,
                              CLASS_DISPLAY_NAME_KEY: term, BBOXES_KEY: term_bbox_map[term]})
        elif box_per_img == "all":
            all_terms = ' ;'.join([t for t in term_bbox_map])
            task_data.append({UUID_KEY: str(uuid.uuid4()), INFO_KEY: parts[0],
                              QUESTION_TYPE_KEY: task_type, URL_KEY: image_url,
                              CLASS_DISPLAY_NAME_KEY: all_terms, BBOXES_KEY:bbox_list})
        else:
            raise ValueError("Invalid box_per_img type: {}. Choose from: one, class, all".format(box_per_img))

    # load gt data to create honey pot
    if hp_file:
        logging.info("load honey pot from {}".format(hp_file))
        hp_data = [json.loads(l[0])
                   for l in read_from_file(hp_file, check_num_cols=1)]
    else:
        logging.info("no honey pot")
        hp_data = []

    # merge task and honey pot data
    output_data = pack_task_with_honey_pot(task_data, hp_data, hp_type,
                                           num_tasks_per_hit,
                                           num_hp_per_hit, hp_neg_prob,
                                           multiple_tasks_per_hit=(task_type!="DrawBox"))
    logging.info("writing #task: {}, #HP: {}"
                 .format(len(task_data),
                         len(task_data)*num_hp_per_hit/num_tasks_per_hit))

    # task file must not exceed 10MB
    file_idx = 0
    outfiles = []
    for start_idx in np.arange(0, len(output_data), num_hits_per_file):
        end_idx = min(start_idx + num_hits_per_file, len(output_data))
        outfile = outbase + "_part{}.tsv".format(file_idx)
        write_task_file(output_data[start_idx: end_idx], outfile)
        outfiles.append(outfile)
        file_idx += 1
    logging.info("write to {} files: {}"
                 .format(len(outfiles), ", ".join(outfiles)))
    return outfiles
