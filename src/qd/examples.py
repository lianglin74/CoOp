from qd.qd_common import init_logging
from future.utils import viewitems
from qd.tsv_io import TSVDataset
from qd.qd_common import list_to_dict
import logging
import simplejson as json
from qd.qd_common import json_dump
from qd.db import is_positive_uhrs_verified, is_negative_uhrs_verified
from qd.tsv_io import is_verified_rect
from qd.qd_common import calculate_iou

# you can all all the functions prefixed with test_ without any input
# parameter
def create_parameters(**kwargs):
    kwargs.update({'data': 'voc20',
                   'max_iter': 10000,
                   'effective_batch_size': 64})
    return kwargs

def create_pipeline(kwargs):
    from qd.qd_pytorch import YoloV2PtPipeline
    return YoloV2PtPipeline(**kwargs)

def find_best_matched_rect(target, rects, check_class=True):
    target_class_lower = target['class'].lower()
    if check_class:
        same_class_rects = [r for r in rects if r['class'].lower() == target_class_lower]
    else:
        same_class_rects = rects
    rect_ious = [(r, calculate_iou(r['rect'], target['rect']))
        for r in same_class_rects]
    if len(rect_ious) == 0:
        return None, -1
    return max(rect_ious, key=lambda r: r[-1])

def uhrs_verify_db_merge_to_tsv(collection_name='uhrs_logo_verification'):
    from qd.db import create_bbverification_db
    c = create_bbverification_db(collection_name=collection_name)
    data_split_to_key_rects, all_id = c.get_completed_uhrs_result()
    merge_uhrs_result_to_dataset(data_split_to_key_rects)
    c.set_status_as_merged(all_id)

def uhrs_merge_one(uhrs_rect, target_rects):
    info = {'num_added': 0,
            'num_removed': 0,
            'verified_confirmed': 0,
            'verified_removed': 0,
            'non_verified_confirmed': 0,
            'non_verified_removed': 0}
    same_rect, iou = find_best_matched_rect(uhrs_rect, target_rects)
    if iou < 0.8:
        if is_positive_uhrs_verified(uhrs_rect):
            target_rects.append(uhrs_rect)
            info['num_added'] = 1
        return info

    if is_verified_rect(same_rect):
        if is_positive_uhrs_verified(uhrs_rect):
            info['verified_confirmed'] = 1
        elif is_negative_uhrs_verified(uhrs_rect):
            info['verified_removed'] = 1
            target_rects.remove(same_rect)
    else:
        if is_positive_uhrs_verified(uhrs_rect):
            info['non_verified_confirmed'] = 1
        elif is_negative_uhrs_verified(uhrs_rect):
            info['non_verified_removed'] = 1
            target_rects.remove(same_rect)

    same_rect['uhrs'] = {}
    for t, v in viewitems(uhrs_rect['uhrs']):
        same_rect['uhrs'][t] = v

    return info

def merge_uhrs_result_to_dataset(data_split_to_key_rects):
    for (data, split), uhrs_key_rects in viewitems(data_split_to_key_rects):
        logging.info((data, split))
        dataset = TSVDataset(data)
        uhrs_key_to_rects = list_to_dict(uhrs_key_rects, 0)
        logging.info('number of image will be affected: {}'.format(len(uhrs_key_rects)))
        info = {}
        def gen_rows():
            for key, str_rects in dataset.iter_data(split, 'label', -1,
                    progress=True):
                rects = json.loads(str_rects)
                if key in uhrs_key_to_rects:
                    uhrs_rects = uhrs_key_to_rects[key]
                    del uhrs_key_to_rects[key]
                else:
                    uhrs_rects = []
                for uhrs_rect in uhrs_rects:
                    sub_info = uhrs_merge_one(uhrs_rect, rects)
                    for k, v in viewitems(sub_info):
                        info[k] = v + info.get(k, 0)
                yield key, json_dump(rects)
            assert len(uhrs_key_to_rects) == 0
        def generate_info():
            for k, v in viewitems(info):
                yield k, v
            for key, rects in viewitems(uhrs_key_to_rects):
                yield key, json_dump(rects)
        dataset.update_data(gen_rows(), split, 'label',
                generate_info=generate_info())


def test_model_pipeline(**kwargs):
    '''
    run the script by

    mpirun -npernode 4 \
            python script_with_this_function_called.py
    '''
    init_logging()
    kwargs.update(create_parameters(**kwargs))
    pip = create_pipeline(kwargs)

    if kwargs.get('monitor_train_only'):
        pip.monitor_train()
    else:
        pip.ensure_train()
        pip.ensure_predict()
        pip.ensure_evaluate()

def test_uhrs_verify_db_merge_to_tsv():
    uhrs_verify_db_merge_to_tsv()

def test_create_new_image_tsv_if_exif_rotated():
    from qd.process_tsv import create_new_image_tsv_if_exif_rotated
    create_new_image_tsv_if_exif_rotated('voc20', 'train')

