from __future__ import absolute_import
from __future__ import division

import json
import numpy as np
import os
import shutil


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
    return json.dumps(json_obj).replace(',', '#').replace('"', '$')


def load_escaped_json(json_str):
    return json.loads(json_str.replace('#', ',').replace('$', '"'))


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


def get_bbox_matching_map(list1, list2, iou_threshold):
    """
    Returns the matching map from list1 to list2. List is a list of bboxes
    i.e., list1[i] matches list2[res[i]] if res[i] is not None
    """
    res = [None] * len(list1)

    if iou_threshold == 0:
        # class only
        for idx1, bbox1 in enumerate(list1):
            for idx2, bbox2 in enumerate(list2):
                if bbox1["class"] == bbox2["class"]:
                    res[idx1] = idx2
    else:
        # class + rect
        visited = set()  # visited idx in list2
        for idx1, bbox1 in enumerate(list1):
            indices2, max_iou = get_max_iou_idx(bbox1, list2)
            if max_iou < iou_threshold:
                continue
            for idx2 in indices2:
                if idx2 not in visited:
                    res[idx1] = idx2
                    visited.add(idx2)
    return res


def calculate_iou(bbox1, bbox2):
    if not is_valid_bbox(bbox1) or not is_valid_bbox(bbox2):
        raise Exception("invalid bbox")
    # intersection part
    intersection_left = max(bbox1['rect'][0], bbox2['rect'][0])
    intersection_top = max(bbox1['rect'][1], bbox2['rect'][1])
    intersection_right = min(bbox1['rect'][2], bbox2['rect'][2])
    intersection_bottom = min(bbox1['rect'][3], bbox2['rect'][3])

    if intersection_right <= intersection_left or \
            intersection_bottom <= intersection_top:
        return 0.0
    w = max(0.0, intersection_right-intersection_left+1)
    h = max(0.0, intersection_bottom-intersection_top+1)
    intersection_area = w * h
    bb1_area = calculate_bbox_area(bbox1)
    bb2_area = calculate_bbox_area(bbox2)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert(iou > 0 and iou <= 1)
    return iou


def is_valid_bbox(bbox):
    if "class" in bbox and "rect" in bbox and len(bbox["rect"]) == 4:
        if bbox["rect"][0] < bbox["rect"][2] and \
                bbox["rect"][1] < bbox["rect"][3]:
            return True
    return False


def calculate_bbox_area(bbox):
    return (bbox['rect'][2] - bbox['rect'][0] + 1) * \
            (bbox['rect'][3] - bbox['rect'][1] + 1)
