from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def read_from_file(filepath, sep='\t', check_num_cols=None):
    with open(filepath, 'r') as fin:
        for line in fin:
            cols = line.rstrip('\n').split(sep)
            if check_num_cols and len(cols)!=check_num_cols:
                raise Exception("expect {} columns, but get {}".format(check_num_cols, str(cols)))
            yield cols

def write_to_file(data, filepath, sep='\t'):
    with open(filepath, 'w') as fout:
        for cols in data:
            fout.write(sep.join([str(c) for c in cols]))
            fout.write('\n')

def search_bbox_in_list(new_bbox, existing_list, iou_threshold):
    """Gets the first index of a matching bbox in the list, if not exist, return -1
    """
    for idx, cur_bbox in enumerate(existing_list):
        if new_bbox["class"].lower() == cur_bbox["class"].lower() \
                and calculate_iou(new_bbox, cur_bbox) > iou_threshold:
            return idx
    return -1

def calculate_iou(bbox1, bbox2):
    if not is_valid_bbox(bbox1) or not is_valid_bbox(bbox2):
        raise Exception("invalid bbox")
    # intersection part
    intersection_left = max(bbox1['rect'][0], bbox2['rect'][0])
    intersection_top = max(bbox1['rect'][1], bbox2['rect'][1])
    intersection_right = min(bbox1['rect'][2], bbox2['rect'][2])
    intersection_bottom = min(bbox1['rect'][3], bbox2['rect'][3])

    if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
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
        if bbox["rect"][0] < bbox["rect"][2] and bbox["rect"][1] < bbox["rect"][3]:
            return True
    return False

def calculate_bbox_area(bbox):
    return (bbox['rect'][2] - bbox['rect'][0] + 1) * (bbox['rect'][3] - bbox['rect'][1] + 1)