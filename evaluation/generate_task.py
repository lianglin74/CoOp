import collections
import copy
import json
import logging
import numpy as np
import os
import uuid

from utils import read_from_file, write_to_file, escape_json_obj


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
                             description_file=None, num_tasks_per_hit=8,
                             num_hp_per_hit=2, hp_neg_prob=0.5):
    """
    Params:
        label_file: tsv file containing labels to be verified,
            columns: image_key, json list of bboxes, image_url
        gt_file: txt file containing ground truth labels, used to
            create honey pot, each line is a json dict
            {"image_key": ..., "image_url": ...,
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
        key = parts[0]
        bbox_list = json.loads(parts[1])
        image_url = parts[2]
        for bbox in bbox_list:
            term = bbox['class']
            if term_description_map and term in term_description_map:
                term = term_description_map[term]
            term_count[term] += 1
            task_data.append({"uuid": str(uuid.uuid4()), "image_key": key,
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
