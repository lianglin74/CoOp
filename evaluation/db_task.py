import argparse
import collections
try:
    # Python 2/3 compatibility
    import izip as zip
except ImportError:
    pass
from future.utils import viewitems
import json
import logging
import os
import os.path as op
from pprint import pformat
import sys

from qd.qd_common import calculate_iou, json_dump
from qd.tsv_io import TSVDataset, tsv_reader,tsv_writer
from qd.db import BoundingBoxVerificationDB
from qd.process_tsv import upload_image_to_blob

CFG_IOU = "iou"
CFG_REQUIRED_FIELDS = "required"
TASK_CONFIGS = {
    "uhrs_tag_verification": {CFG_IOU: 0, CFG_REQUIRED_FIELDS: ["class"]},
    "uhrs_bounding_box_verification": {CFG_IOU: 0.5, CFG_REQUIRED_FIELDS: ["class", "rect"]},
    "uhrs_logo_verification": {CFG_IOU: 0.5, CFG_REQUIRED_FIELDS: ["class", "rect"]},
    "uhrs_text_verification": {CFG_IOU: 0, CFG_REQUIRED_FIELDS: ["class"]},
}

def submit_to_verify(gt_dataset_name, gt_split, pred_file, collection_name,
            conf_thres=0, gt_version=-1, is_urgent=False):
    """ Submits UHRS verification task to qd database
    Args:
        gt_dataset_name: name of the ground truth dataset
        gt_split: split of the ground truth dataset
        pred_file: TSV file, columns are image_key, json list of dict.
                For detection results, dict must contain "class", "rect".
                For tag reuslts, dict must contain "class".
        collection_name: choose from
                uhrs_tag_verification -> for tagging
                uhrs_bounding_box_verification -> for general OD
                uhrs_logo_verification -> for logo detection
        conf_thres: confidence threshold of prediction results
    """
    cfg = TASK_CONFIGS[collection_name]
    priority_tier = BoundingBoxVerificationDB.urgent_priority_tier if is_urgent else 2
    gt_dataset = TSVDataset(gt_dataset_name)

    key2pred = {}
    for parts in tsv_reader(pred_file):
        if len(parts) == 2 and parts[0] and parts[1]:
            key2pred[parts[0]] = json.loads(parts[1])
        else:
            logging.info('invalid pred: {}'.format(str(parts)))

    num_total_bboxes = 0
    num_verify = 0
    num_in_gt = 0
    all_bb_tasks = []

    gt_iter = gt_dataset.iter_data(gt_split, t='label', version=gt_version)
    url_iter = gt_dataset.iter_data(gt_split, t='key.url', version=gt_version)

    for gt_parts, url_parts in zip(gt_iter, url_iter):
        assert(gt_parts[0] == url_parts[0])
        key = gt_parts[0]
        if key not in key2pred:
            continue
        gt_bboxes = json.loads(gt_parts[1])
        url = url_parts[1]
        for bbox in key2pred[key]:
            if not all([k in bbox for k in cfg[CFG_REQUIRED_FIELDS]]):
                raise ValueError("missing fields in bbox: {}. Must include: {}"
                            .format(str(bbox), str(cfg[CFG_REQUIRED_FIELDS])))
            num_total_bboxes += 1
            # NOTE: if exists a gt bbox, s.t., "class" is equal in case-insensitive mode,
            # AND "rect" IoU > thres, then the bbox will be treated correct, and will NOT
            # be verified
            is_in_gt = False
            for gt_bbox in gt_bboxes:
                if bbox["class"].lower() == gt_bbox["class"].lower() and \
                            (cfg[CFG_IOU] == 0 or calculate_iou(bbox["rect"], gt_bbox["rect"]) > cfg[CFG_IOU]):
                    is_in_gt = True
                    break
            if is_in_gt:
                num_in_gt += 1
                continue

            if "conf" not in bbox or conf_thres==0 or bbox["conf"] >= conf_thres:
                num_verify += 1
                if "rect" not in bbox:
                    bbox["rect"] = "DUMMY"  # for tag results
                all_bb_tasks.append({
                    "url": url,
                    "split": gt_split,
                    "key": key,
                    "data": gt_dataset_name,
                    'priority_tier': priority_tier,
                    'priority': 1,
                    "rect": bbox
                })

    logging.info("#total bboxes: {}, #in gt: {}, #submit: {}"
            .format(num_total_bboxes, num_in_gt, num_verify))
    cur_db = BoundingBoxVerificationDB(db_name = 'qd', collection_name = collection_name)
    cur_db.request_by_insert(all_bb_tasks)

def download_merge_correct_to_gt(collection_name, gt_dataset_name=None, gt_split=None):
    cfg = TASK_CONFIGS[collection_name]
    tag_only = (collection_name == "uhrs_tag_verification")
    # query completed results
    cur_db = BoundingBoxVerificationDB(db_name = 'qd', collection_name = collection_name)
    extra_match = {}
    if gt_dataset_name:
        extra_match['data'] = gt_dataset_name
    if gt_split:
        extra_match['split'] = gt_split
    data_split_to_key_rects, all_id = cur_db.get_completed_uhrs_result(extra_match=extra_match)

    all_data_split = list(data_split_to_key_rects.keys())
    # check rect format
    for (data, split), uhrs_key_rects in viewitems(data_split_to_key_rects):
        for _, rect in uhrs_key_rects:
            assert(all([k in rect for k in cfg[CFG_REQUIRED_FIELDS]]))
            if tag_only and "rect" in rect:
                del rect["rect"]

    # merge verified correct to ground truth
    from qd import process_tsv
    process_tsv.merge_uhrs_result_to_dataset(data_split_to_key_rects, tag_only=tag_only)
    all_gt_labels = []
    for data, split in all_data_split:
        dataset = TSVDataset(data)
        all_gt_labels.append(dataset.get_data(split, 'label', -1))
    return all_gt_labels


def is_uhrs_consensus_correct(uhrs_res):
    """ From UHRS feedback, "1" -> Yes, "2" -> No, "3" -> Can't judge
    Consensus correct means more people select "1" than "2"
    """
    if "1" in uhrs_res:
        if "2" not in uhrs_res and uhrs_res["1"] > 0:
            return True
        elif "2" in uhrs_res and uhrs_res["1"] > uhrs_res["2"]:
            return True
    return False


def get_uhrs_detailed_judgment():
    collection_name = 'uhrs_text_verification'
    # query completed results
    cur_db = BoundingBoxVerificationDB(db_name = 'qd', collection_name = collection_name)
    data_split_to_key_rects, all_id = cur_db.get_completed_uhrs_result()
    for data, split in data_split_to_key_rects.keys():
        key2rects = collections.defaultdict(list)
        for key, rect in data_split_to_key_rects[(data, split)]:
            key2rects[key].append(rect)

        outfile = op.join('data', data, 'uhrs',
                '{}.uhrs.counts.tsv'.format(split))
        out = []
        for key, rects in key2rects.items():
            out.append([key, json_dump(rects)])
        tsv_writer(out, outfile)


def ensure_build_dataset_to_verify(dataset_name, split, image_tsv=None,
        gt_label_tsv=None):
    img_fpath = 'data/uhrs_verify_tag_imagenet/Jian1570A2_base64.balance_min50.tsv'
    gt_fpath = 'data/uhrs_verify_tag_imagenet/Jian1570A2_base64.label.balance_min50.tsv'
    dataset_name = 'uhrs_verify_tag_imagenet'
    split = 'train'

    def gen_rows():
        img_iter = tsv_reader(img_fpath)
        gt_iter = tsv_reader(gt_fpath)
        for img_parts, gt_parts in zip(img_iter, gt_iter):
            assert(img_parts[0] == gt_parts[0])
            gt_labels = gt_parts[1].split(',')
            gt_bboxes = []
            for l in gt_labels:
                l = l.strip()
                assert l != ''
                gt_bboxes.append({'class': l})
            yield img_parts[0], json_dump(gt_bboxes), img_parts[2]

    outfile = op.join('data', dataset_name, '{}.tsv'.format(split))
    if op.isfile(outfile):
        logging.info('already exists: {}'.format(outfile))
        return
    tsv_writer(gen_rows(), outfile)


def populate_dataset_to_evaluate(dataset_name, split, data_root):
    data_dir = op.join(data_root, dataset_name)
    assert op.isdir(data_dir), 'directory: {} does not exist'.format(data_dir)

    # check image url
    image_file = op.join(data_dir, '{}.tsv'.format(split))
    assert op.isfile(image_file), 'image tsv: {} does not exist'.format(image_file)
    url_file = op.join(data_dir, '{}.key.url.tsv'.format(split))
    if not op.isfile(url_file):
        upload_image_to_blob(dataset_name, split)

    # check label file
    label_file = op.join(data_dir, '{}.label.tsv'.format(split))
    if not op.isfile(label_file):
        def gen_rows():
            for parts in tsv_reader(image_file):
                yield parts[0], "[]"
        tsv_writer(gen_rows(), label_file)


def check_predict_format(data_type, predict_file, dataset_name, split, data_root):
    assert op.isfile(predict_file), 'predict file: {} does not exist'.format(predict_file)
    data_dir = op.join(data_root, dataset_name)
    label_file = op.join(data_dir, '{}.label.tsv'.format(split))
    image_keys = set(p[0] for p in tsv_reader(label_file))

    def is_label_valid(labels, required_fields):
        if not isinstance(labels, list):
            return False
        for rect in labels:
            if not isinstance(rect, dict):
                return False
            for f in required_fields:
                if f not in rect:
                    return False
        return True

    if data_type in ['tagging', 'text']:
        required_fields = ['class']
    else:
        required_fields = ['class', 'rect']
    for parts in tsv_reader(predict_file):
        key = parts[0]
        labels = json.loads(parts[1])
        assert key in image_keys, '{} not found in dataset {}'.format(key,
                dataset_name)
        if not is_label_valid(labels, required_fields):
            raise ValueError('Format error in predict file: {}'.format(parts[1]))


def main():
    logging.info(pformat(sys.argv))

    parser = argparse.ArgumentParser(description='Human evaluation for tagging and detection')
    parser.add_argument('--task', required=True, type=str,
                        help='Choose from submit, download')
    parser.add_argument('--type', required=True, type=str,
                        help='Choose from tagging, detection, logo')
    parser.add_argument('--dataset', required=True, type=str,
                        help='The name of evaluation dataset. See README.md for'
                             'a list of available datasets')
    parser.add_argument('--split', default='test', type=str,
                        help='The split of evaluation dataset, default is test')
    parser.add_argument('--predict_file', default='', type=str,
                        help='Required for submitting task, the prediction'
                             'results not in ground truth will be submitted to verify')
    parser.add_argument('--conf_thres', default=0, type=float,
                        help='Confidence threshold of prediction results')

    args = parser.parse_args()
    if args.type == 'tagging':
        collection_name = 'uhrs_tag_verification'
    elif args.type == 'detection':
        collection_name = 'uhrs_bounding_box_verification'
    elif args.type == 'logo':
        collection_name = 'uhrs_logo_verification'
    elif args.type == 'text':
        collection_name = 'uhrs_text_verification'
    else:
        raise ValueError('unknown data type: {}'.format(args.type))

    data_root = './data'  # data root is hard coded due to TSVDataset requirements
    gt_dataset_name = args.dataset
    gt_split = args.split
    populate_dataset_to_evaluate(gt_dataset_name, gt_split, data_root)

    if args.task == 'submit':
        predict_file = args.predict_file
        check_predict_format(args.type, predict_file, gt_dataset_name, gt_split, data_root)
        submit_to_verify(gt_dataset_name, gt_split, predict_file, collection_name,
                conf_thres=args.conf_thres, gt_version=-1, is_urgent=True)

        logging.info('*****************************************')
        logging.info('Successfully submitted tasks. Please check the progress at:')
        logging.info('verify tag -> https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitappid=35851')
        logging.info('verify bbox -> https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitAppId=35716&taskOption=0&project=-1&taskGroupId=-1')
        logging.info('*****************************************')

    elif args.task == 'download':
        logging.info('*********IMPORTANT NOTE******************')
        logging.info('Make sure that all the tasks you submitted have finished before downloading')
        logging.info('verify tag -> https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitappid=35851')
        logging.info('verify bbox -> https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitAppId=35716&taskOption=0&project=-1&taskGroupId=-1')

        updated_files = download_merge_correct_to_gt(collection_name, gt_dataset_name=gt_dataset_name, gt_split=gt_split)

        logging.info('*****************************************')
        logging.info('Successfully download tasks.')
        logging.info('The ground truth labels are updated at: {}'.format('; '.join(updated_files)))
        logging.info('*****************************************')
    else:
        raise ValueError('unknown task: {}'.format(args.task))


if __name__ == "__main__":
    from qd.qd_common import init_logging
    init_logging()
    main()
    #get_uhrs_detailed_judgment()

