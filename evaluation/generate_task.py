import base64
import collections
import copy
import cv2
import json
import logging
import numpy as np
import os
import uuid

from utils import read_from_file, write_to_file, escape_json_obj, calculate_bbox_area
import _init_paths
from process_tsv import get_img_url2
from tsv_io import TSVFile


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
        self._cur_idx = -1
        self._count = len(data)
        if self._count == 0:
            raise Exception("data cannot be empty")
        self.hp_neg_prob = hp_neg_prob
        self.class_candidates = set(class_candidates)
        if not self.class_candidates:
            self.class_candidates = set(d["class"] for d in self._data)
        self.OPT_POS = 1
        self.OPT_NEG = 2

    def next(self):
        self._cur_idx = (self._cur_idx + 1) % self._count
        if self._data_type == "hp":
            # return it directly if data is alreay in honey pot format
            return self._data[self._cur_idx]
        else:
            # convert gt label to honey pot format
            hp = copy.deepcopy(self._data[self._cur_idx])
            if self.hp_neg_prob > 0 and np.random.rand() < self.hp_neg_prob:
                false_class = np.random.choice(
                    list(self.class_candidates - set(hp["objects_to_find"])),
                    size=1)
                hp["objects_to_find"] = false_class[0]
                hp["expected_output"] = self.OPT_NEG
            else:
                hp["expected_output"] = self.OPT_POS
            return hp


def pack_task_with_honey_pot(task_data, hp_data, hp_type, num_tasks_per_hit,
                             num_hp_per_hit, hp_neg_prob=0.5):
    ''' Generate hits composed of real tasks and honey pot
    '''
    output_content = []
    num_total_task = len(task_data)
    num_total_hp = len(hp_data)
    class_candidates = set(d["objects_to_find"] for d in task_data)
    hp_gen = HoneyPotGenerator(hp_data, hp_type, hp_neg_prob=hp_neg_prob,
                               class_candidates=class_candidates)
    for start in np.arange(0, num_total_task, num_tasks_per_hit):
        end = min(start + num_tasks_per_hit, num_total_task)
        line = task_data[start: end]
        for _ in range(num_hp_per_hit):
            line.append(hp_gen.next())
        np.random.shuffle(line)
        output_content.append(line)
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


def generate_verify_box_task(label_file, gt_file, outbase,
                             description_file=None, num_tasks_per_hit=10,
                             num_hp_per_hit=2, hp_neg_prob=0.5):
    """
    Params:
        label_file: tsv file containing labels to be verified,
            columns: image_info, json list of bboxes, image_url
        gt_file: txt file containing ground truth labels, used to
            create honey pot, each line is a json dict
            {"image_info": ..., "image_url": ...,
            "objects_to_find": ..., "bboxes":...}
        outfile: one column of "input_content", used to submit
            UHRS/MTurk tasks
        description_file: tsv file describing the terms, used to
            change display name. columns: term, description
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
        dataset = parts[0]
        bbox_list = json.loads(parts[1])
        image_url = parts[2]
        for bbox in bbox_list:
            term = bbox['class']
            if term_description_map and term in term_description_map:
                term = term_description_map[term]
            if (dataset.startswith("brand") or dataset.startswith("logo")) and not term.endswith("logo"):
                term = bbox['class'] + " logo"
            term_count[term] += 1
            task_data.append({"uuid": str(uuid.uuid4()), "image_info": dataset,
                              "image_url": image_url,
                              "objects_to_find": term, "bboxes": [bbox]})

    # load gt data to create honey pot
    logging.info("load honey pot from {}".format(gt_file))
    gt_data = [json.loads(l[0])
               for l in read_from_file(gt_file, check_num_cols=1)]

    # merge task and honey pot data
    output_data = pack_task_with_honey_pot(task_data, gt_data, "gt",
                                           num_tasks_per_hit,
                                           num_hp_per_hit, hp_neg_prob)
    logging.info("writing #task: {}, #HP: {}"
                 .format(len(task_data),
                         len(task_data)*num_hp_per_hit/num_tasks_per_hit))

    num_hits_per_file = 2000  # task file must not exceed 10MB
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


def generate_honeypot(dataset_name, imgfiles, labelfiles, outfile=None, easy_area_thres=0.2):
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
                bbox["rect"] = [np.clip(bbox["rect"][0], 0, w), np.clip(bbox["rect"][1], 0, h),
                                np.clip(bbox["rect"][2], 0, w), np.clip(bbox["rect"][3], 0, h)]
                bbox_area = calculate_bbox_area(bbox)
                if float(bbox_area) / (h*w) > easy_area_thres:
                    num_easy_bboxes += 1
                    term = bbox["class"]
                    if dataset_name.startswith("brand") or dataset_name.startswith("logo"):
                        term = bbox["class"] + " logo"
                    hp_data.append(
                        {"image_info": dataset_name, "image_url": image_url,
                            "objects_to_find": term,
                            "bboxes": [bbox]})

    print("#total bboxes: {:d}\t#easy bboxes:{:d}".format(num_total_bboxes,
                                                          num_easy_bboxes))
    if outfile:
        with open(outfile, 'w') as fout:
            for d in hp_data:
                fout.write(json.dumps(d))
                fout.write('\n')
    return hp_data
