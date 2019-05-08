from qd.tsv_io import tsv_reader
from qd.qd_common import write_to_file
from qd.qd_common import init_logging
from qd.qd_common import worth_create
from qd.process_tsv import load_key_rects
import logging
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os.path as op
import copy


def tsv_rect_to_json_rect(rect, image_id, label_to_id=None, extra=1):
    ann = copy.deepcopy(rect)

    r = rect['rect']
    ann['bbox'] = [r[0], r[1], r[2] - r[0] + extra, r[3] - r[1] + extra]
    del ann['rect']

    ann['category_id'] = label_to_id[rect['class']] if label_to_id else rect['class']
    del ann['class']

    ann['image_id'] = image_id

    if 'conf'  in rect:
        # if it is gt, it has no conf
        ann['score'] = rect['conf']
        del rect['conf']

    return ann

def convert_to_cocoformat(predict_tsv, predict_json, label_to_id=None,
        key_to_id=None):
    rows = tsv_reader(predict_tsv)
    annotations = []
    from tqdm import tqdm
    logging.info(predict_tsv)
    for row in tqdm(rows):
        image_id = row[0]
        if key_to_id:
            image_id = key_to_id[image_id]
        rects = json.loads(row[1])
        for rect in rects:
            ann = tsv_rect_to_json_rect(rect, image_id, label_to_id)
            annotations.append(ann)

    write_to_file(json.dumps(annotations), predict_json)

def convert_gt_to_cocoformat(gt_iter, gt_json):
    key_rects = load_key_rects(gt_iter)
    result = {}
    result['images'] = [{'id': key} for key, _ in key_rects]
    result['categories'] = [{'id': k, 'name': k} for k in
            set([r['class'] for key, rects in key_rects for r in rects])]
    result['annotations'] = [tsv_rect_to_json_rect(rect, key) for key, rects in
        key_rects for rect in rects]
    for i, r in enumerate(result['annotations']):
        r['id'] = i
        if 'iscrowd' not in r:
            r['iscrowd'] = 0
        if 'area' not in r:
            r['area'] = r['bbox'][2] * r['bbox'][3]
    write_to_file(json.dumps(result), gt_json)

def coco_eval_tsv(predict_tsv, gt_tsv):
    gt_json = gt_tsv + '.cocoformat.json'
    predict_json = predict_tsv + '.cocoformat.json'
    if worth_create(gt_tsv, gt_json):
        convert_gt_to_cocoformat(tsv_reader(gt_tsv), gt_json)
    if worth_create(predict_tsv, predict_json):
        convert_to_cocoformat(predict_tsv, predict_json)

    return coco_eval_json(predict_json, gt_json)

def coco_eval_json(predict_json, gt_json):
    cocoGt=COCO(gt_json)
    cocoDt=cocoGt.loadRes(predict_json)

    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    keys = ['all-all', '0.5-all', '0.75-all',
            'all-small', 'all-medium', 'all-large',
            'AR-all-1', 'AR-all-10', 'AR-all',
            'AR-small', 'AR-medium', 'AR-large']
    # s is the type of np.float64. convert it to native type
    return {key: float(s) for key, s in zip(keys, cocoEval.stats)}

def coco_eval_simple_tsv_json(predict_tsv, gt_json, force=False):
    predict_json = predict_tsv + '.cocoformat.json'
    # gt file

    if worth_create(predict_tsv, predict_json) or force:
        convert_to_cocoformat(predict_tsv, predict_json)
    else:
        logging.info('ignore to create the json file')

    return coco_eval_json(predict_json, gt_json)

def coco_eval_tsv_json(predict_tsv, gt_json):
    '''
    we need to figure out the label_to_id and key_to_id from gt_json
    '''
    predict_json = predict_tsv + '.cocoformat_align.json'
    # gt file

    if worth_create(predict_tsv, predict_json):
        logging.info('create json file: {}'.format(predict_json))
        with open(gt_json, 'r') as fp:
            gt = json.load(fp)
        label_to_id = {cat_info['name']: cat_info['id'] for cat_info in gt['categories']}
        key_to_id = {str(x['id']): x['id'] for x in gt['images']}
        convert_to_cocoformat(predict_tsv, predict_json, label_to_id,
                key_to_id)
    else:
        logging.info('ignore to create the json file')

    return coco_eval_json(predict_json, gt_json)

def test_coco_eval():
    predict_tsv = \
        './output/coco2017_darknet19_448_B_noreorg_extraconv2/snapshot/model_iter_236574.caffemodel.coco2017.test.maintainRatio.predict'
    gt_tsv = './data/coco2017Full/test.label.tsv'
    gt_json = op.expanduser(
            '~/data/raw_data/raw_coco/annotations/instances_val2017.json')
    coco_eval_tsv_json(predict_tsv, gt_json)
    coco_eval_tsv(predict_tsv, gt_tsv)

if __name__ == '__main__':
    init_logging()
    test_coco_eval()


