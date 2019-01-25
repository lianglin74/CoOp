from __future__ import absolute_import
from __future__ import division

import json
import numpy as np
import os
import shutil

import _init_paths
from scripts import deteval

def list_files_in_dir(dirpath):
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath)
            if os.path.isfile(os.path.join(dirpath, f))]


def ensure_dir_empty(dirpath, remove_ok=False):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        if len(os.listdir(dirpath)) > 0:
            if not remove_ok:
                raise Exception("Directory not empty: {}".format(dirpath))
            shutil.rmtree(dirpath)
            os.mkdir(dirpath)


def read_from_file(filepath, sep='\t', check_num_cols=None):
    with open(filepath, 'r') as fin:
        for line in fin:
            cols = line.strip().split(sep)
            if check_num_cols and len(cols) != check_num_cols:
                raise Exception("expect {} columns, but get {}"
                                .format(check_num_cols, str(cols)))
            yield cols


def write_to_file(data, filepath, sep='\t'):
    with open(filepath+".tmp", 'w') as fout:
        for cols in data:
            fout.write(sep.join([str(c) for c in cols]))
            fout.write('\n')
    if os.path.isfile(filepath):
        os.remove(filepath)
    os.rename(filepath + ".tmp", filepath)


def escape_json_obj(json_obj):
    # deprecated: make json_string CSV compatiable
    # return json.dumps(json_obj).replace(',', '#').replace('"', '$')
    return json.dumps(json_obj, separators=(',',':'))


def load_escaped_json(json_str):
    # return json.loads(json_str.replace('#', ',').replace('$', '"'))
    return json.loads(json_str)


def search_bbox_in_list(new_bbox, existing_list, iou_threshold):
    """Gets the first index of a matching bbox in the list,
        if not exist, return -1
    """
    for idx, cur_bbox in enumerate(existing_list):
        if new_bbox["class"].lower() == cur_bbox["class"].lower():
            if iou_threshold == 0:
                return idx
            else:
                if calculate_iou(new_bbox, cur_bbox) > iou_threshold:
                    return idx
    return -1


def get_max_iou_idx(new_bbox, bbox_list):
    """Gets the indices of the bboxes in bbox_list with max IoU
    """
    max_iou = 0
    max_indices = []
    for idx, cur_bbox in enumerate(bbox_list):
        if new_bbox["class"].lower() == cur_bbox["class"].lower():
            cur_iou = calculate_iou(new_bbox, cur_bbox)
            if cur_iou > max_iou:
                max_iou = cur_iou
                max_indices = [idx]
            elif cur_iou == max_iou:
                max_indices.append(idx)
    return max_indices, max_iou


def get_bbox_matching_map(bbox_list1, bbox_list2, iou_threshold, allow_multiple_to_one=False):
    """
    Returns a dict that maps indices of bbox_list1 to those of bbox_list2,
    s.t. IoU(bbox_list1[i], bbox_list2[map[i]]) > iou_threshold
    bbox_list1, bbox_list2: a list of bboxes
    allow_multiple_to_one: if False, force a one-to-one mapping, i.e., if i!=j, map[i]!=map[j]
    """
    match_map = {}

    if iou_threshold == 0:
        # class only
        for i, p in enumerate(bbox_list1):
            for j, g in enumerate(bbox_list2):
                if p["class"].lower() == g["class"].lower():
                    match_map[i] = j
                    break
    else:
        # class + rect
        matched = [False] * len(bbox_list2)
        for i, p in enumerate(bbox_list1):
            max_iou = 0
            max_iou_idx = None
            for j, g in enumerate(bbox_list2):
                if p["class"].lower() == g["class"].lower():
                    if allow_multiple_to_one or not matched[j]:
                        iou = deteval.IoU(p["rect"], g["rect"])
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_idx = j
            if max_iou > iou_threshold:
                matched[max_iou_idx] = True
                match_map[i] = max_iou_idx
    return match_map


def calculate_iou(bbox1, bbox2):
    if not is_valid_bbox(bbox1) or not is_valid_bbox(bbox2):
        raise Exception("invalid bbox")
    return deteval.IoU(bbox1["rect"], bbox2["rect"])


def is_valid_bbox(bbox):
    if "class" in bbox and "rect" in bbox and len(bbox["rect"]) == 4:
        if bbox["rect"][0] < bbox["rect"][2] and \
                bbox["rect"][1] < bbox["rect"][3]:
            return True
    return False


def calculate_bbox_area(bbox):
    return (bbox['rect'][2] - bbox['rect'][0] + 1) * \
            (bbox['rect'][3] - bbox['rect'][1] + 1)


def truncate_rect(rect, im_h, im_w):
    return [np.clip(rect[0], 0, im_w), np.clip(rect[1], 0, im_h), np.clip(rect[2], 0, im_w), np.clip(rect[3], 0, im_h)]
