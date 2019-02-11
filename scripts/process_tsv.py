try:
    from itertools import izip as zip
except:
    # if it comes here, it is python3
    pass
from process_image import show_images
import re
import shutil
import glob
import yaml
import matplotlib.pyplot as plt
from tsv_io import tsv_reader, tsv_writer
from pprint import pformat
import os
import os.path as op
import sys
import json
from pprint import pprint
import multiprocessing as mp
import random
import numpy as np
from qd_common import ensure_directory
from qd_common import default_data_path, load_net
import time
from multiprocessing import Queue
from shutil import copytree
from shutil import rmtree
from qd_common import img_from_base64
import cv2
import base64
from taxonomy import load_label_parent, labels2noffsets
from taxonomy import LabelToSynset, synset_to_noffset
from taxonomy import noffset_to_synset
from taxonomy import get_nick_name
from taxonomy import load_all_tax, merge_all_tax
from taxonomy import child_parent_print_tree2
from taxonomy import create_markdown_url
#from taxonomy import populate_noffset
from taxonomy import populate_cum_images
from taxonomy import gen_term_list
from taxonomy import gen_noffset
from taxonomy import populate_url_for_offset
from taxonomy import disambibuity_noffsets
from taxonomy import Taxonomy
from qd_common import read_to_buffer, load_list_file
from qd_common import write_to_yaml_file, load_from_yaml_file
from qd_common import encoded_from_img
from tsv_io import extract_label
from tsv_io import create_inverted_tsv
from tsv_io import create_inverted_list

from process_image import draw_bb, show_image, save_image
from qd_common import write_to_file
from tsv_io import TSVDataset, TSVFile
from qd_common import init_logging
import logging
from qd_common import is_cluster
from tsv_io import tsv_shuffle_reader
import argparse
from tsv_io import get_meta_file
import imghdr
from qd_common import calculate_iou
from qd_common import yolo_old_to_new
from qd_common import generate_lineidx
from qd_common import list_to_dict, list_to_dict_unique
from qd_common import dict_to_list
from qd_common import parse_test_data
from qd_common import worth_create
from tsv_io import load_labels
import unicodedata
from taxonomy import is_noffset
import copy
from tsv_io import create_inverted_list2
from qd_common import calculate_image_ap
from tqdm import tqdm
from pymongo import MongoClient
import pymongo
from datetime import datetime
import inspect
from qd_common import hash_sha1

def get_data_distribution(data, split, version):
    # return a dictionary to represent the distribution of the data
    # only composite dataset requires the following two information
    result = {}

    populate_dataset_details(data)
    source_to_num_image = {}
    source_to_num_bb = {}
    dataset = TSVDataset(data)
    tsvsource_to_imageratio = {src[5:]: 
            (float(numdestimages) / float(numsrcimages))
            for src, numdestimages, numsrcimages in 
            dataset.iter_data(split + 'X', 'tsvsource.numdestimages.numsrcimages')}
    result['tsvsource_to_imageratio'] = tsvsource_to_imageratio

    tsvsource_to_boxratio = {src[5:]: 
            (float(numdest) / float(numsrc))
            for src, numdest, numsrc in 
            dataset.iter_data(split + 'X', 'tsvsource.numdestbbs.numsrcbbs')}
    result['tsvsource_to_boxratio'] = tsvsource_to_boxratio

    tsvsource_to_num_image = {source[5:]: int(num_image) for source, num_image in dataset.iter_data(split + 'X', 'numImagesPerSource')}
    result['tsvsource_to_num_image'] = tsvsource_to_num_image

    tsvsource_to_num_category = {source[5:]: int(num_image) for source, num_image
            in dataset.iter_data(split + 'X', 'numCategoriesPerSource')}
    result['tsvsource_to_num_category'] = tsvsource_to_num_category

    label_to_count = {l: int(c) for l, c in dataset.iter_data(split, 'inverted.label.count', version=version)}
    unique_labels = set([l if not l.startswith('-') else l[1:] for l in label_to_count])
    import math
    all_label_pos_negs = [[l, [label_to_count.get(l, 0), -label_to_count.get('-' + l, 0)]] for l in unique_labels]
    all_label_pos_negs = [[l, [math.log10(label_to_count.get(l, 0) + 1), 
        -math.log10(label_to_count.get('-' + l, 0) + 1)]] for l in unique_labels]
    all_label_pos_negs = sorted(all_label_pos_negs, key=lambda x: x[1][0])
    result['all_label_pos_negs'] = all_label_pos_negs

    result['num_labels_with_neg'] = len([l for l, (pos, neg) in all_label_pos_negs if neg != 0])
    result['num_labels_with_pos'] = len([l for l, (pos, neg) in all_label_pos_negs if pos != 0])

    return result

def convert_pred_to_dataset_label(full_expid, predict_file, 
        th_file, min_value):
    pred_file = op.join('output',
            full_expid,
            'snapshot',
            predict_file)

    from qd_common import parse_test_data
    data, split = parse_test_data(predict_file)

    from process_tsv import load_key_rects

    dataset = TSVDataset(data)
    populate_dataset_details(data)
    latest_version = dataset.get_latest_version(split, 'label')
    gt_key_rects = load_key_rects(dataset.iter_data(split, 'label',
        latest_version))
    pred_key_rects = load_key_rects(tsv_reader(pred_file))
    pred_key_to_rects = {key: rects for key, rects in pred_key_rects}
    if th_file:
        th_file = op.join('output', full_expid, 'snapshot', th_file)
        per_cat_th = {l: max(float(th), min_value) for l, th, _ in tsv_reader(th_file)}
    else:
        per_cat_th = {}
    def gen_rows():
        for key, rects in gt_key_rects:
            pred_rects = pred_key_to_rects.get(key, [])
            pred_rects = [r for r in pred_rects
                    if r['conf'] >= per_cat_th.get(r['class'], 0)]
            yield key, json.dumps(pred_rects)
    info = [('full_expid', full_expid),
            ('predict_file', pred_file),
            ('th_file', th_file),
            ('min_value', min_value)]
    dataset.update_data(gen_rows(), split, 'label',
            generate_info=info)

    populate_dataset_details(data)


def ensure_inject_expid(full_expid):
    from qd_common import get_all_predict_files
    from qd_common import load_solver
    solver_prototxt = op.join('output', full_expid,
        'solver.prototxt')
    if not op.isfile(solver_prototxt):
        logging.info('ignore since {} does not exist'.format(solver_prototxt))
        return
    solver_param = load_solver(solver_prototxt)
    model_param = '{0}_iter_{1}.caffemodel'.format(
            solver_param.snapshot_prefix, solver_param.max_iter)
    all_predict = glob.glob(model_param + '*.predict')
    for p in all_predict:
        ensure_inject_expid_pred(full_expid, op.basename(p))

def ensure_inject_dataset(data):
    for split in ['train', 'trainval', 'test']:
        ensure_upload_image_to_blob(data, split)
        ensure_inject_image(data, split)
        ensure_inject_gt(data, split)

def ensure_inject_decorate(func):
    def func_wrapper(*args, **kwargs):
        client = get_mongodb_client()
        db = client['qd']
        task = db['task']
        # .func_name is ok for python2, but not for python3. .__name__ is ok
        # for both
        func_name = func.__name__
        if func_name.startswith('ensure_'):
            func_name = func_name[len('ensure_'): ]
        argnames, ins_varargs, ins_kwargs, ins_defaults = inspect.getargspec(func)
        # we have not supported if other paramseters is not None
        assert ins_varargs is None and ins_kwargs is None and ins_defaults is None
        query = {'task_type': func_name}
        for n, v in zip(argnames[: len(args)], args):
            assert n not in query
            query[n] = v
        for k in kwargs:
            assert k not in query
            query[k] = kwargs[k]
        
        # create the unique index to make the update_one atomic
        logging.info('make sure the index is created: {}'.format(
            ', '.join(query.keys())))
        task.create_index([(k, pymongo.ASCENDING) for k in query])

        while True:
            result = task.update_one(filter=query,
                                    update={'$setOnInsert': query},
                                    upsert=True)
            if result.matched_count == 0:
                task.update_one(filter=query,
                        update={'$set': {'status': 'started',
                                         'create_time': datetime.now()}})
                func(*args, **kwargs)
                task.update_many(filter=query,
                        update={'$set': {'status': 'done'}})
                return True
            else:
                assert result.matched_count == 1
                existing_entries = list(task.find(query))
                if any(e for e in existing_entries if e['status'] == 'done'):
                    logging.info('ignore to inject since it is done: \n{}'.format(query))
                    break
                else:
                    logging.info('waiting to finish \n{}'.format(
                        pformat(existing_entries)))
                    time.sleep(10)
                # return False if not done
    return func_wrapper

def ensure_composite_key_url(data, split):
    dataset = TSVDataset(data)
    if not dataset.has(split):
        logging.info('data tsv does not exist')
        return
    if dataset.has(split, 'key.url'):
        logging.info('it exists')
        return
    data_to_dataset = {}
    uploaded_key_url = set()
    def gen_rows():
        for key, _ in dataset.iter_data(split, 'label'):
            c_data, c_split, c_key = parse_combine(key)
            if c_data in data_to_dataset:
                c_dataset = data_to_dataset[c_data]
            else:
                c_dataset = TSVDataset(c_data)
                data_to_dataset[c_data] = c_dataset
            if (c_data, c_split) not in uploaded_key_url:
                ensure_upload_image_to_blob(c_data, c_split)
                uploaded_key_url.add((c_data, c_split))
            _, url = c_dataset.seek_by_key(c_key, c_split, 'key.url')
            yield key, url
    dataset.write_data(gen_rows(), split, 'key.url')

def upload_image_to_blob(data, split):
    from cloud_storage import CloudStorage
    from StringIO import StringIO
    s = CloudStorage()
    dataset = TSVDataset(data)
    if not dataset.has(split):
        logging.info('{} - {} does not exist'.format(data, split))
        return
    logging.info('{} - {}'.format(data, split))
    if not op.isfile(dataset.get_data(split)) and \
            op.isfile(dataset.get_data(split + 'X')):
        ensure_composite_key_url(data, split)
        logging.info('ignore to upload images for composite dataset')
        return
    def gen_rows():
        for key, _, str_im in tqdm(dataset.iter_data(split)):
            key = parse_combine(key)[-1]
            clean_name = _map_img_key_to_name2(key)
            url = s.upload_stream(StringIO(base64.b64decode(str_im)),
                    clean_name)
            yield key, url
    dataset.write_data(gen_rows(), split, 'key.url')

@ensure_inject_decorate
def ensure_upload_image_to_blob(data, split):
    upload_image_to_blob(data, split)

@ensure_inject_decorate
def ensure_inject_image(data, split):
    from pymongo.errors import BulkWriteError

    dataset = TSVDataset(data)
    if not dataset.has(split):
        return

    client = MongoClient()
    db = client['qd']
    images = db['image']
    images.delete_many(filter={'data': data, 'split': split})
    images.create_index([
        ('data', pymongo.ASCENDING),
        ('split', pymongo.ASCENDING),
        ('key', pymongo.ASCENDING),
        ],
        unique=True)
    from StringIO import StringIO
    from process_tsv import ImageTypeParser
    from cloud_storage import CloudStorage
    all_data = []

    logging.info('injecting {} - {}'.format(data, split))
    injected = set()
    key_to_hw = None
    if dataset.has(split, 'hw'):
        key_to_hw = {key: [int(x) for x in hw.split(' ')] for key, hw in
                dataset.iter_data(split, 'hw')}
    assert dataset.has(split, 'key.url')
    key_to_url = {key: url for key, url in dataset.iter_data(split, 'key.url')}
    logging.info('injecting image for {} - {}'.format(data, split))
    for i, (key, _) in tqdm(enumerate(dataset.iter_data(split,
        'label'))):
        if key in injected:
            continue
        else:
            injected.add(key)
        url = key_to_url[key]
        doc = {'data': data,
                'split': split,
                'key': key,
                'idx_in_split': i,
                'url': url,
                'create_time': datetime.now(),
                }
        if key_to_hw:
            doc['height'], doc['width'] = key_to_hw[key]
        all_data.append(doc)
        if len(all_data) > 1000:
            images.insert_many(all_data)
            all_data = []
    if len(all_data) > 0:
        images.insert_many(all_data)
        all_data = []

def get_mongodb_client():
    return MongoClient()

@ensure_inject_decorate
def ensure_update_pred_with_correctness(data, split,
        full_expid, predict_file):
    ensure_inject_gt(data, split)
    ensure_inject_pred(full_expid, predict_file, data, split)

    update_pred_with_correctness(data, split,
        full_expid,
        predict_file)


def update_pred_with_correctness(test_data, test_split,
        full_expid,
        predict_file):
    client = MongoClient()
    db = client['qd']
    pred_collection = db['predict_result']
    _gt = db['ground_truth']
    _pred = db['predict_result']

    # get the latest version of the gt
    pipeline = [{'$match': {'data': test_data,
                            'split': test_split,
                            'version': {'$lte': 0}}},
                {'$group': {'_id': {'data': '$data',
                                    'split': '$split',
                                    'key': '$key',
                                    'action_target_id': '$action_target_id'},
                            'contribution': {'$sum': '$contribution'},
                            'class': {'$first': '$class'},
                            'rect': {'$first': '$rect'}}},
                {'$match': {'contribution': {'$gte': 1}}}, # if it is 0, it means we removed the box
                {'$addFields': {'data': '$_id.data',
                                'split': '$_id.split',
                                'key': '$_id.key'}},
                ]
    def get_query_pipeline(data, split, key, class_name):
        return [{'$match': {'$expr': {'$and': [{'$eq': ['$data', '$${}'.format(data)]},
                                               {'$eq': ['$split', '$${}'.format(split)]},
                                               {'$eq': ['$key', '$${}'.format(key)]},
                                               {'$eq': ['$class', '$${}'.format(class_name)]}]}}},
                 {'$group': {'_id': '$action_target_id',
                             'contribution': {'$sum': '$contribution'},
                             'create_time': {'$max': '$create_time'},
                             'rect': {'$first': '$rect'}}},
                 {'$match': {'contribution': {'$gte': 1}}},
                ]

    # get the target prediction bounding boxes
    pipeline = [{'$match': {'full_expid': full_expid,
                            'pred_file': predict_file}},
                {'$group': {'_id': {'data': '$data',
                                   'split': '$split',
                                   'key': '$key',
                                   'class': '$class'},
                           'pred_box': {'$push': {'rect': '$rect',
                                                  'conf': '$conf',
                                                  'pred_box_id': '$_id'}}}},
                {'$lookup': {'from': 'ground_truth',
                             'let': {'target_data': '$_id.data',
                                     'target_split': '$_id.split',
                                     'target_key': '$_id.key',
                                     'target_class': '$_id.class'},
                             'pipeline': get_query_pipeline('target_data',
                                                            'target_split',
                                                            'target_key',
                                                            'target_class'),
                             'as': 'gt_box'}},
            ]
    all_correct_box, all_wrong_box = [], []
    for row in tqdm(_pred.aggregate(pipeline, allowDiskUse=True)):
        curr_pred = row['pred_box']
        curr_gt = row['gt_box']
        curr_pred = sorted(curr_pred, key=lambda x: -x['conf'])
        for p in curr_pred:
            matched = [g for g in curr_gt if
                    not g.get('used', False) and
                    ('rect' not in p or
                        not p['rect'] or
                        'rect' not in g or
                        not g['rect'] or
                        calculate_iou(g['rect'], p['rect']) > 0.3)]
            if len(matched) == 0:
                all_wrong_box.append(p['pred_box_id'])
            else:
                all_correct_box.append(p['pred_box_id'])
                matched[0]['used'] = True
        if len(all_correct_box) > 1000:
            _pred.update_many({'_id': {'$in': all_correct_box}},
                    {'$set': {'correct': 1}})
            all_correct_box = []
        if len(all_wrong_box) > 1000:
            _pred.update_many({'_id': {'$in': all_wrong_box}},
                    {'$set': {'correct': 0}})
            all_wrong_box = []
    if len(all_correct_box) > 0:
        _pred.update_many({'_id': {'$in': all_correct_box}},
                {'$set': {'correct': 1}})
        all_correct_box = []
    if len(all_wrong_box) > 0:
        _pred.update_many({'_id': {'$in': all_wrong_box}},
                {'$set': {'correct': 0}})
        all_wrong_box = []

def ensure_inject_expid_pred(full_expid, predict_file):
    try:
        data, split = parse_test_data(predict_file)
    except:
        logging.info('ignore to inject {} - {}'.format(full_expid, predict_file))
        return
    ensure_upload_image_to_blob(data, split)
    ensure_inject_image(data, split)
    ensure_inject_gt(data, split)
    ensure_inject_pred(full_expid,
            predict_file,
            data,
            split)
    ensure_update_pred_with_correctness(data, split,
        full_expid, predict_file)

@ensure_inject_decorate
def ensure_inject_pred(full_expid, pred_file, test_data, test_split):
    client = MongoClient()
    db = client['qd']
    pred_collection = db['predict_result']
    logging.info('cleaning {} - {}'.format(full_expid, pred_file))
    pred_collection.delete_many({'full_expid': full_expid, 'pred_file': pred_file})
    pred_file = op.join('output', full_expid, 'snapshot', pred_file)
    all_rect = []
    for key, label_str in tqdm(tsv_reader(pred_file)):
        rects = json.loads(label_str)
        rects = [r for r in rects if r['conf'] > 0.05]
        for i, r in enumerate(rects):
            r['full_expid'] = full_expid
            r['pred_file'] = op.basename(pred_file)
            r['data'] = test_data
            r['split'] = test_split
            r['key'] = key
        all_rect.extend(rects)
        if len(all_rect) > 10000:
            pred_collection.insert_many(all_rect)
            all_rect = []
    if len(all_rect) > 0:
        pred_collection.insert_many(all_rect)

def ensure_inject_gt(data, split):
    client = MongoClient()
    db = client['qd']
    task = db['task']
    if split is None:
        for s in ['train', 'trainval', 'test']:
            ensure_inject_gt(data, s)
    else:
        dataset = TSVDataset(data)
        version = 0
        set_previous_key_rects = False
        while dataset.has(split, 'label', version=version):
            query = {'task_type': 'inject_gt',
                                    'data': data,
                                    'split': split,
                                    'version': version}
            result = task.update_one(filter=query,
                                    update={'$setOnInsert': query},
                                    upsert=True)
            if result.matched_count == 0:
                logging.info('start to inserting {}-{}-{}'.format(data,
                    split, version))
                # no process is working on inserting current version
                task.update_one(filter=query,
                        update={'$set': {'status': 'started',
                                         'create_time': datetime.now()}})
                if not set_previous_key_rects:
                    if version == 0:
                        previous_key_to_rects = {}
                    else:
                        key_rects = load_key_rects(dataset.iter_data(split, 'label',
                            version=version-1))
                        previous_key_to_rects = {key: rects for key, rects in key_rects}
                    set_previous_key_rects = True
                inject_gt_version(data, split, version, previous_key_to_rects)
                task.update_many(filter={'task_type': 'inject_gt',
                    'data': data,
                    'split': split,
                    'version': version}, update={'$set': {'status': 'done'}})
            else:
                # it is done or it is started by anohter process. let's skip
                # this version
                assert result.matched_count == 1
                while True:
                    existing_entries = list(task.find(query))
                    if any(e for e in existing_entries if e['status'] == 'done'):
                        logging.info('ignore to inject since it is done: \n{}'.format(query))
                        break
                    elif len(existing_entries) == 0:
                        logging.info('we will do it')
                        version = version - 1
                        break
                    else:
                        logging.info('waiting to finish \n{}'.format(
                            pformat(existing_entries)))
                        time.sleep(10)
                set_previous_key_rects = False
            version = version + 1

def inject_gt_version(data, split, version, previous_key_to_rects):
    client = MongoClient()
    db = client['qd']
    gt = db['ground_truth']
    dataset = TSVDataset(data)
    gt.delete_many({'data': data, 'split': split, 'version': version})
    dataset = TSVDataset(data)

    if not dataset.has(split, 'label', version):
        return False
    all_rect = []
    logging.info('{}-{}-{}'.format(data, split, version))
    for i, (key, label_str) in tqdm(enumerate(dataset.iter_data(
        split, 'label', version=version))):
        rects = json.loads(label_str)
        for j, r in enumerate(rects):
            r['data'] = data
            r['split'] = split
            r['key'] = key
            # for tagging dataset, tehre is no rect
            r['action_target_id'] = hash_sha1([data, split, key, r['class'],
                r.get('rect', [])])
            r['version'] = version
        if version == 0:
            assert key not in previous_key_to_rects
            for r in rects:
                r['action_type'] = 'add'
                r['contribution'] = 1
                r['create_time'] = datetime.now()
            all_rect.extend(rects)
        else:
            previous_rects = previous_key_to_rects[key]
            # use strict_rect_in_rects rather than rect_in_rects: if the higher
            # version contains more properies, we want to have the properies in
            # teh database also.
            previous_not_in_current = copy.deepcopy([r for r in previous_rects if
                    not strict_rect_in_rects(r, rects)])
            current_not_in_previous = copy.deepcopy([r for r in rects if 
                    not strict_rect_in_rects(r, previous_rects)])
            for r in previous_not_in_current:
                r['data'] = data
                r['split'] = split
                r['key'] = key
                r['action_target_id'] = hash_sha1([data, split, key,
                    r['class'], r.get('rect', [])])
                r['version'] = version
                r['action_type'] = 'remove'
                r['contribution'] = -1
                r['create_time'] = datetime.now()
                r['version'] = version
            all_rect.extend(previous_not_in_current)
            for r in current_not_in_previous:
                r['data'] = data
                r['split'] = split
                r['key'] = key
                r['action_target_id'] = hash_sha1([data, split, key,
                    r['class'], r.get('rect', [])])
                r['version'] = version
                r['action_type'] = 'add'
                r['contribution'] = 1
                r['create_time'] = datetime.now()
                r['version'] = version
            all_rect.extend(current_not_in_previous)
        previous_key_to_rects[key] = rects
        if len(all_rect) > 10000:
            db_insert_many(gt, all_rect)
            all_rect = []

    if len(all_rect) > 0:
        logging.info('inserting {} - {} - {}'.format(data, split, version))
        db_insert_many(gt, all_rect)
        all_rect = []
    return True

def db_insert_many(collection, all_info):
    for a in all_info:
        if '_id' in a:
            del a['_id']
    collection.insert_many(all_info)

class VisualizationDatabaseByMongoDB():
    def __init__(self):
        self._client = MongoClient()
        self._db = self._client['qd']
        self._pred = self._db['predict_result']
        self._gt = self._db['ground_truth']

    def _get_positive_start(self, start_id, max_item):
        if start_id < 0:
            rank = pymongo.DESCENDING
            start_id = min(0, start_id + max_item)
            start_id = abs(start_id)
        else:
            rank = pymongo.ASCENDING
        return rank, start_id

    def query_pipeline(self, pipeline, collection, db_name):
        image_info = list(self._client[db_name][collection].aggregate(
            pipeline, allowDiskUse=True))
        logging.info(len(image_info))
        if len(image_info) > 0:
            logging.info(pformat(image_info[0]))
        image_info = [{'key': x['key'],
            'url': x.get('url', ''),
            'gt': x.get('gt', []),
            'pred': x.get('pred', [])} for x in image_info]
        return image_info

    def insert(self, dic, collection, db_name):
        return self._client[db_name][collection].insert(dic)

    def query_by_id(self, _id, collection, db_name):
        from bson.objectid import ObjectId
        return list(self._client[db_name][collection].find({'_id':
            ObjectId(_id)}))[0]

    def query_predict_recall(self, full_expid, pred_file, class_name, threshold, start_id, max_item):
        rank, start_id = self._get_positive_start(start_id, max_item)
        # from the predict file
        row = self._pred.find_one({'full_expid': full_expid, 'pred_file':
            pred_file})
        if row is None:
            logging.info('no prediction data in db')
            return []
        data = row['data']
        split = row['split']
        logging.info(data)
        logging.info(split)
        pipeline = [{'$match': {'data': data, 'split': split, 'class': class_name}},
                    {'$group': {'_id': {'key': '$key', 'target_action_id': '$target_action_id'},
                                'contribution': {'$sum': '$contribution'},
                                'num_target_label': {'$sum': 1}}},
                    {'$match': {'contribution': {'$gte': 1}}},
                    {'$group': {'_id': {'key': '$_id.key'},
                                'num_contribution': {'$sum': 1}}},
                    ## add the number of correct predicted
                    {'$lookup': {
                        'from': 'predict_result',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'full_expid': full_expid,
                                                 'pred_file': pred_file,
                                                 'class': class_name,
                                                 'conf': {'$gte': threshold}}},
                                     {'$group': {'_id': {'key': '$key'},
                                                 'correct': {'$sum': '$correct'}}},
                                      {'$match': {'$expr': {'$and': [{'$eq': ['$_id.key', '$$key']}]}}},
                                      ],
                        'as': 'recall',
                        }},
                    {'$unwind': {'path': '$recall', 'preserveNullAndEmptyArrays': True}},
                    {'$addFields': {'recall': {'$divide': ['$recall.correct', '$num_target_label']}}},
                    {'$sort': {'recall': rank}},
                    {'$skip': start_id},
                    {'$limit': max_item},
                    # add the field of url
                    {'$lookup': {
                        'from': 'image',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', data]},
                                                                    {'$eq': ['$split', split]},
                                                                    {'$eq': ['$key', '$$key']},],}}},
                                     {'$project': {'url': True, '_id': 0}}],
                        'as': 'url',
                        }},
                    {'$addFields': {'url': {'$arrayElemAt': ['$url', 0]}}},
                    {'$addFields': {'url': '$url.url'}},
                    ## add the field of pred
                    {'$lookup': {
                        'from': 'predict_result',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$full_expid', full_expid]},
                                                                    {'$eq': ['$pred_file', pred_file]},
                                                                    {'$eq': ['$key', '$$key']},
                                                                    {'$gte': ['$conf', threshold]}]}}},
                                     {'$project': {'conf': True, 'rect': True, 'class': True, '_id': 0}}],
                        'as': 'pred',
                        }
                    },
                    ## add the gt field
                    {'$lookup': {
                        'from': 'ground_truth',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', data]},
                                                                    {'$eq': ['$split', split]},
                                                                    {'$eq': ['$key', '$$key']},],}}},
                                    {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                                'contribution': {'$sum': '$contribution'},
                                                'rect': {'$first': '$rect'},
                                                'class': {'$first': '$class'},
                                                'conf': {'$first': '$conf'}}, },
                                    {'$match': {'contribution': {'$gte': 1}}},
                                    ],
                        'as': 'gt',
                        }}
                    ]
        image_info = list(self._gt.aggregate(pipeline))
        logging.info(len(image_info))
        image_info = [{'key': x['_id']['key'],
            'url': x['url'],
            'gt': x['gt'],
            'pred': x['pred']} for x in image_info]
        return image_info

    def query_predict_precision(self, full_expid, pred_file, class_name, threshold, start_id, max_item):
        rank, start_id = self._get_positive_start(start_id, max_item)
        filter_pred = {'full_expid': full_expid,
                            'pred_file': pred_file,
                            'conf': {'$gte': threshold}}
        if class_name != 'None' and class_name:
            filter_pred['class'] = class_name
        pipeline = [
                {'$match': filter_pred},
                {'$group': {'_id': {'data': '$data',
                                    'split': '$split',
                                    'key': '$key'},
                            'correct': {'$sum': '$correct'},
                            'total': {'$sum': 1}}},
                {'$addFields': {'precision': {'$divide': ['$correct', '$total']}}},
                {'$sort': {'precision': rank}},
                {'$skip': start_id},
                {'$limit': max_item},
                # add the field of url
                {'$lookup': {
                    'from': 'image',
                    'let': {'data': '$_id.data',
                            'split': '$_id.split',
                            'key': '$_id.key'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', '$$data']},
                                                                {'$eq': ['$split', '$$split']},
                                                                {'$eq': ['$key', '$$key']},],}}},
                                 {'$project': {'url': True, '_id': 0}}],
                    'as': 'url',
                    }},
                {'$unwind': '$url'},
                {'$addFields': {'url': '$url.url'}},
                ## add the field of pred
                {'$lookup': {
                    'from': 'predict_result',
                    'let': {'data': '$_id.data',
                            'split': '$_id.split',
                            'key': '$_id.key'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$full_expid', full_expid]},
                                                                {'$eq': ['$pred_file', pred_file]},
                                                                {'$eq': ['$key', '$$key']},
                                                                {'$gte': ['$conf', threshold]}]}}},
                                 {'$project': {'conf': True, 'rect': True, 'class': True, '_id': 0}}],
                    'as': 'pred',
                    }
                },
                ## add the gt field
                {'$lookup': {
                    'from': 'ground_truth',
                    'let': {'data': '$_id.data',
                            'split': '$_id.split',
                            'key': '$_id.key'},
                    'pipeline': [{'$match': {'$expr': {'$and': [{'$eq': ['$data', '$$data']},
                                                                {'$eq': ['$split', '$$split']},
                                                                {'$eq': ['$key', '$$key']},],}}},
                                {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                            'contribution': {'$sum': '$contribution'},
                                            'rect': {'$first': '$rect'},
                                            'class': {'$first': '$class'},
                                            'conf': {'$first': '$conf'}}, },
                                {'$match': {'contribution': {'$gte': 1}}},
                                ],
                    'as': 'gt',
                    }}
                ]

        logging.info(pformat(pipeline))
        image_info = list(self._pred.aggregate(pipeline))
        image_info = [{'key': x['_id']['key'],
            'url': x['url'],
            'gt': x['gt'],
            'pred': x['pred']} for x in image_info]
        return image_info

    def query_ground_truth(self, data, split, version, label, start_id, max_item):
        pipeline = self.get_ground_truth_pipeline(data, split, version, label, start_id, max_item)
        image_info = list(self._gt.aggregate(pipeline['pipeline']))
        logging.info(len(image_info))
        image_info = [{'key': x['key'], 
            'url': x['url'], 
            'gt': x['gt']} for x in image_info]
        return pipeline, image_info

    def get_ground_truth_pipeline(self, data, split, version, label, start_id, max_item):
        rank, start_id = self._get_positive_start(start_id, max_item)

        match_pairs = {'data': data}
        gt_match = []
        gt_match.append({'$eq': ['$data', data]})
        url_match = []
        url_match.append({'$eq': ['$data', data]})
        if split is not None:
            match_pairs['split'] = split
            gt_match.append({'$eq': ['$split', split]})
            url_match.append({'$eq': ['$split', split]})
        if label is not None:
            match_pairs['class'] = label
        if version is not None:
            match_pairs['version'] = {'$lte': version}
            gt_match.append({'$lte': ['$version', version]})
        gt_match.append({'$eq': ['$key', '$$key']})
        url_match.append({'$eq': ['$key', '$$key']})
        pipeline = [{'$match': match_pairs},
                    {'$group': {'_id': {'key': '$key'},
                                'contribution': {'$sum': '$contribution'}}},
                    {'$match': {'contribution': {'$gte': 1}}},
                    {'$sort': {'_id.key': rank}},
                    {'$skip': start_id},
                    {'$limit': max_item},
                    # add gt boxes
                    {'$lookup': {
                        'from': 'ground_truth',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': gt_match}}},
                                     {'$group': {'_id': {'action_target_id': '$action_target_id'},
                                                 'contribution': {'$sum': '$contribution'},
                                                 'rect': {'$first': '$rect'},
                                                 'class': {'$first': '$class'},
                                                 'conf': {'$first': '$conf'}}, },
                                     {'$match': {'contribution': {'$gte': 1}}},
                                    ],
                        'as': 'gt',
                        }},
                    # add url
                    {'$lookup': {
                        'from': 'image',
                        'let': {'key': '$_id.key'},
                        'pipeline': [{'$match': {'$expr': {'$and': url_match}}},
                                     {'$project': {'url': True, '_id': 0}}],
                        'as': 'url',
                        }},
                    {'$unwind': '$url'},
                    {'$addFields': {'url': '$url.url',
                                    'key': '$_id.key'}},
                    ]

        return {'pipeline': pipeline,
                'database': 'qd',
                'collection': 'ground_truth'}

def get_class_count(data, splits):
    dataset = TSVDataset(data)
    result = {}
    for split in splits:
        result[split] = {row[0]: int(row[1])
                for row in dataset.iter_data(
                    split, 'inverted.label.count', -1)}
    return result

def update_yolo_test_proto(input_test, test_data, map_file, output_test):
    dataset = TSVDataset(test_data)
    if op.isfile(dataset.get_noffsets_file()):
        test_noffsets = dataset.load_noffsets()
    else:
        test_noffsets = labels2noffsets(dataset.load_labelmap())
    test_map_id = []
    net = load_net(input_test)
    for l in net.layer:
        if l.type == 'RegionOutput':
            tree_file = l.region_output_param.tree
            r = load_label_parent(tree_file)
            noffset_idx, noffset_parentidx, noffsets = r
            for noffset in test_noffsets:
                test_map_id.append(noffset_idx[noffset])
            write_to_file('\n'.join(map(str, test_map_id)), map_file)
            l.region_output_param.map = map_file
            l.region_output_param.thresh = 0.005
    assert len(test_noffsets) == len(test_map_id)
    write_to_file(str(net), output_test)

def get_colored_name(node, colors):
    name = node.name
    if hasattr(node, 'sub_group'):
        sub_group = node.sub_group
    else:
        sub_group = -1
    idx = sub_group + 1
    while idx >= len(colors):
        colors.append('rgb({},{},{})'.format(
            int(random.random() * 255),
            int(random.random() * 255),
            int(random.random() * 255)))
    return "<span style='color:{}'>{}</span>".format(colors[idx],
            name)

def get_vis_url(data, l, split=None):
    if split is None:
        return '/detection/view_image?data={}&label={}&start_id=0'.format(
                data, l)
    else:
        return '/detection/view_image?data={}&split={}&label={}&start_id=0'.format(
                data, split, l)


def get_readable_label(l):
    if is_noffset(l):
        return '{}({})'.format(l, get_nick_name(noffset_to_synset(l)))
    else:
        return l

def get_node_key(node, k, default_value=-1):
    r = default_value
    if hasattr(node, k):
        r = node.__getattribute__(k)
    return r

def get_node_info(data, node, colors, all_data):
    ss = []
    if hasattr(node, 'ap'):
        ss.append('ap({0:.2f})'.format(node.ap))
    if hasattr(node, 'images_with_bb') and \
            hasattr(node, 'images_no_bb'):
        keys = ['with_bb', 'no_bb']
        for k in keys:
            ss.append("{}({},<span><a href='{}' target='_blank'>{}</a></span>,<span><a href='{}' target='_blank'>{}</a></span>)".format(
                k,
                get_node_key(node, 'images_' + k, -1),
                get_vis_url(data + '_' + k, node.name, split='train'),
                get_node_key(node, k + '_train', -1),
                get_vis_url(data + '_' + k, node.name, split='test'),
                get_node_key(node, k + '_test', -1)))
    ignore_keys = ['support', 'dist', 'name', 'url',
            'with_bb_train',
            'with_bb_test',
            'no_bb_train',
            'no_bb_test',
            'ap',
            'images_with_bb',
            'images_no_bb',
            'noffset',
            'sub_group',
            'cum_images_with_bb',
            'cum_images_no_bb']
    for key in node.features:
        if any(x == key for x in ignore_keys):
            continue
        if key.endswith('_readable') or \
                key.endswith('_toTrain') or \
                key.endswith('_toTest') or \
                key.endswith('_total'):
            continue
        value = node.__getattribute__(key)
        if key in all_data and value != None:
            total = get_node_key(node, '{}_total'.format(key))
            toTrain = get_node_key(node, '{}_toTrain'.format(key))
            toTest = get_node_key(node, '{}_toTest'.format(key))
            extra_info = '({},{},{})'.format(total, toTrain, toTest)
            labels = value.split(',')
            value = ','.join(["<a href='{}' target='_blank'>{}</a>".format(get_vis_url(key, l),
                get_readable_label(l)) for l in labels])
            value = '{}{}'.format(value, extra_info)
        ss.append('{}[{}]'.format(key, value))
    if len(ss) > 0:
        return "{}: {}".format(
                get_colored_name(node, colors),
                '; '.join(ss))
    else:
        return get_colored_name(node, colors)

def gen_html_tree_view(data, full_expid=None,
        predict_file=None):
    colors=['rgb(0,0,0)',
                'rgb(255,0,0)',
                'rgb(0,0,255)']
    dataset = TSVDataset(data)
    all_data = os.listdir('./data')
    file_name = op.join(dataset._data_root, 'root_enriched.yaml')
    logging.info('loading {}'.format(file_name))
    with open(file_name, 'r') as fp:
        config_tax = yaml.load(fp)
    tax = Taxonomy(config_tax)
    if full_expid is not None and \
            predict_file is not None:
        map_file = op.join('output',full_expid, 'snapshot',
                op.splitext(predict_file)[0] + '.report.class_ap.json')
        if op.isfile(map_file):
            class_ap = json.loads(read_to_buffer(map_file))
            class_ap = class_ap['overall']['0.3']['class_ap']
            for node in tax.root.iter_search_nodes():
                node.add_feature('ap',
                        class_ap.get(node.name, -1))

    def gen_html_tree_view_rec(root):
        '''
        include itself
        '''
        if len(root.children) == 0:
            s = u"<li data-jstree='{{\"icon\":\"glyphicon glyphicon-leaf\"}}'><span>{}</span></li>".format(
                    get_node_info(data, root, colors, all_data))
            return s
        else:
            result = []
            # we cannot remove span tag here
            result.append("<li data-jstree='{{\"opened\":true}}' ><span>{}</span>".format(
                get_node_info(data, root, colors, all_data)))
            result.append('<ul>')
            all_child = sorted(root.children, key=lambda c: c.sub_group)
            for c in all_child:
                r = gen_html_tree_view_rec(c)
                result.append(r)
            result.append('</ul>')
            result.append('</li>')
            return '\n'.join(result)
    s = gen_html_tree_view_rec(tax.root)
    return s


def gt_predict_images(predicts, gts, test_data, target_images, label, start_id, threshold,
        label_to_idx, image_aps, test_data_split='test'):
    test_dataset = TSVDataset(test_data)
    test_tsv = TSVFile(test_dataset.get_data(test_data_split))
    for i in xrange(start_id, len(target_images)):
        key = target_images[i]
        logging.info('key = {}, ap = {}'.format(key, image_aps[i][1]))
        idx = label_to_idx[key]
        row = test_tsv.seek(idx)
        origin = img_from_base64(row[2])
        im_gt = np.copy(origin)
        draw_bb(im_gt, [g['rect'] for g in gts[key]],
                [g['class'] for g in gts[key]])
        im_gt_target = np.copy(origin)
        gts_target = [g for g in gts[key] if g['class'] == label]
        draw_bb(im_gt_target, [g['rect'] for g in gts_target],
                [g['class'] for g in gts_target])
        im_pred = np.copy(origin)
        rects = [p for p in predicts[key] if p['conf'] > threshold]
        draw_bb(im_pred, [r['rect'] for r in rects],
                [r['class'] for r in rects],
                [r['conf'] for r in rects])
        im_pred_target = np.copy(origin)
        rects = [p for p in rects if p['class'] == label]
        draw_bb(im_pred_target, [r['rect'] for r in rects],
                [r['class'] for r in rects],
                [r['conf'] for r in rects])
        yield key, origin, im_gt_target, im_pred_target, im_gt, im_pred, image_aps[i][1]

def get_confusion_matrix_by_predict_file_label(full_expid,
        predict_file, label, threshold):
    '''
    get confusion matrix for specific label
    '''
    test_data, test_data_split = parse_test_data(predict_file)

    # load the gt
    logging.info('loading {} - {}'.format(test_data, test_data_split))
    test_dataset = TSVDataset(test_data)
    rows = test_dataset.iter_data(test_data_split, 'label', -1)
    gts = {}
    label_to_idx = {}
    keys_with_label = []
    for i, row in enumerate(rows):
        rects = json.loads(row[1])
        gts[row[0]] = rects
        label_to_idx[row[0]] = i
        if any(r['class'] == label for r in rects):
            keys_with_label.append(row[0])

    predict_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    if not full_expid.startswith('Tax'):
        logging.info('loading {}'.format(predict_file))
        predicts, _ = load_labels(predict_full_path)
    else:
        # create the inverted index
        pred_label_file = '{}.labelmap.tsv'.format(predict_full_path)
        pred_inverted_file = '{}.inverted.tsv'.format(predict_full_path)
        pred_key_file = '{}.key.tsv'.format(predict_full_path)
        if not op.isfile(pred_label_file) or \
                not op.isfile(pred_inverted_file) or \
                not op.isfile(pred_key_file):
            pred_inverted = {}
            pred_keys = []
            rows = tsv_reader(predict_full_path)
            logging.info('loading data and creating index')
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(predict_full_path))
            logging.info('done loading data and creating index')
            pred_labels = inverted.keys()
            tsv_writer(([l] for l in pred_labels), pred_label_file)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_file)
            tsv_writer([[k] for k in pred_keys], pred_key_file)
        # find out data whose prediction has the target label
        all_labels = load_list_file(pred_label_file)
        if label in all_labels:
            row = TSVFile(pred_inverted_file).seek(all_labels.index(label))
            assert row[0] == label
            idx = map(int, row[1].split(' '))
        else:
            idx = []
        # find out the index from the ground truth
        key_to_predidx = {k: i for i, k in enumerate(load_list_file(pred_key_file))}
        idx.extend([key_to_predidx[k] for k in keys_with_label])
        idx = set(idx)
        tsv = TSVFile(predict_full_path)
        predicts = {}
        logging.info('loading')
        for i in idx:
            row = tsv.seek(i)
            assert len(row) == 2
            predicts[row[0]] = json.loads(row[1])
        logging.info('done')
        # load data from the inverted index
    logging.info('done loading {}'.format(predict_file))

    return {'predicts': predicts,
            'gts': gts,
            'label_to_idx': label_to_idx}

def get_confusion_matrix_by_predict_file(full_expid,
        predict_file, threshold):
    '''
    get confusion matrix for all classes
    '''

    test_data, test_data_split = parse_test_data(predict_file)

    # load the gt
    logging.info('loading {} - {}'.format(test_data, test_data_split))
    test_dataset = TSVDataset(test_data)
    rows = test_dataset.iter_data(test_data_split, 'label', -1)
    gts = {}
    label_to_idx = {}
    for i, row in enumerate(rows):
        gts[row[0]] = json.loads(row[1])
        label_to_idx[row[0]] = i

    logging.info('loading {}'.format(predict_file))
    predicts, _ = load_labels(op.join('output', full_expid, 'snapshot', predict_file))
    logging.info('done loading {}'.format(predict_file))

    # calculate the confusion matrix
    confusion_pred_gt = {}
    confusion_gt_pred = {}
    update_confusion_matrix(predicts, gts, threshold,
            confusion_pred_gt,
            confusion_gt_pred)

    return {'predicts': predicts,
            'gts': gts,
            'confusion_pred_gt': confusion_pred_gt,
            'confusion_gt_pred': confusion_gt_pred,
            'label_to_idx': label_to_idx}

def inc_one_dic_dic(dic, c1, c2):
    if c1 not in dic:
        dic[c1] = {}
    if c2 not in dic[c1]:
        dic[c1][c2] = 0
    dic[c1][c2] = dic[c1][c2] + 1

def update_confusion_matrix(predicts, gts, threshold,
            confusion_pred_gt,
            confusion_gt_pred):
    for key in predicts:
        curr_pred = [p for p in predicts[key] if p['conf'] > threshold]
        if key not in gts:
            continue
        curr_gt = gts[key]
        if len(curr_pred) == 0 and len(curr_gt) > 0:
            for g in curr_gt:
                inc_one_dic_dic(confusion_gt_pred, g['class'], 'None')
            continue
        elif len(curr_pred) > 0 and len(curr_gt) == 0:
            for p in curr_pred:
                inc_one_dic_dic(confusion_pred_gt, p['class'], 'None')
            continue
        elif len(curr_pred) == 0 and len(curr_gt) == 0:
            continue
        ious = np.zeros((len(curr_pred), len(curr_gt)))
        for i, p in enumerate(curr_pred):
            for j, g in enumerate(curr_gt):
                iou = calculate_iou(p['rect'], g['rect'])
                ious[i, j] = iou
        gt_idx = np.argmax(ious, axis=1)
        for i, p in enumerate(curr_pred):
            j = gt_idx[i]
            predict_class = p['class']
            gt_class = curr_gt[j]['class']
            if ious[i, j] > 0.3:
                inc_one_dic_dic(confusion_pred_gt,
                        predict_class, gt_class)
            else:
                inc_one_dic_dic(confusion_pred_gt,
                        predict_class, 'None')
        pred_idx = np.argmax(ious, axis=0)
        for j, g in enumerate(curr_gt):
            i = pred_idx[j]
            predict_class = curr_pred[i]['class']
            gt_class = g['class']
            if ious[i, j] > 0.3:
                inc_one_dic_dic(confusion_gt_pred,
                        gt_class, predict_class)
            else:
                inc_one_dic_dic(confusion_gt_pred,
                        gt_class, 'None')


def normalize_to_str(s):
    if sys.version_info.major == 3:
        return s
    else:
        if type(s) is str:
            s = s.decode('unicode_escape')
        return unicodedata.normalize('NFKD', s).encode('ascii','ignore')

def normalize_str_in_rects(data, out_data):
    '''
    normalize all the unicode to string
    '''
    dataset = TSVDataset(data)
    dest_dataset = TSVDataset(out_data)
    splits = ['train', 'test', 'trainval']
    for split in splits:
        if not op.isfile(dataset.get_data(split)):
            continue
        def gen_rows():
            for key, label_str, im_str in tsv_reader(dataset.get_data(split)):
                rects = json.loads(label_str)
                for rect in rects:
                    s = rect['class']
                    rect['class'] = normalize_to_str(s)
                    if s != rect['class']:
                        logging.info(u'{}->{}'.format(s, rect['class']))
                label_str = json.dumps(rects)
                yield key, label_str, im_str
        tsv_writer(gen_rows(), dest_dataset.get_data(split))

def tsv_details(row_hw, row_label, num_rows):
    label_count = {}
    sizes = []
    logging.info('tsv details...')
    for r_hw, r_label in tqdm(zip(row_hw, row_label), total=num_rows):
        if r_label[1] == 'd':
            # this is the deleted label
            rects = []
        else:
            rects = json.loads(r_label[1])
        assert r_hw[0] == r_label[0]
        height, width = map(int, r_hw[1].split(' '))
        if type(rects) is list:
            # this is the detection dataset
            # convert it to str. if it is unicode, in yaml, there will be some
            # special tags, which is annoying
            curr_labels = set(normalize_to_str(rect['class']) for rect in rects)
            for rect in rects:
                if "rect" not in rect:
                    # it is image-level annotation
                    continue
                r = rect['rect']
                if all(x == 0 for x in r):
                    # it is image-level annotation
                    continue
                # this should be a valid bounding box
                cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                rw, rh = r[2] - r[0], r[3] - r[1]
                #assert cx >= 0 and cx < width \
                        #and cy >= 0 and cy < height
                if rw < 1 or rh < 1:
                    logging.warn('rw or rh too small: {} - {}'.format(r_label[0],
                        ','.join(map(str, r))))
        else:
            # this is classification dataset
            assert type(rects) is int
            curr_labels = [rects]
        for c in curr_labels:
            if c in label_count:
                label_count[c] = label_count[c] + 1
            else:
                label_count[c] = 1
        sizes.append((height, width))
    size_counts = [s[0] * s[1] for s in sizes]
    min_size = sizes[np.argmin(size_counts)]
    max_size = sizes[np.argmax(size_counts)]
    min_size = map(float, min_size)
    max_size = map(float, max_size)
    mean_size = (np.mean([s[0] for s in sizes]),
            np.mean([s[1] for s in sizes]))
    mean_size = map(float, mean_size)

    return {'label_count': label_count,
            'min_image_size': min_size,
            'max_image_size': max_size,
            'mean_image_size': mean_size}


def detect_duplicate_key(tsv, duplicate_tsv):
    rows = tsv_reader(tsv)
    key_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        if key in key_to_idx:
            key_to_idx[key].append(i)
        else:
            key_to_idx[key] = [i]
    found_error = False
    def gen_rows():
        for key in key_to_idx:
            idxs = key_to_idx[key]
            if len(idxs) > 1:
                logging.info('duplicate key: {}: {}'.format(key, ', '.join(map(str,
                    idxs))))
                yield key, ','.join(map(str, idxs))
    tsv_writer(gen_rows(), duplicate_tsv)
    return TSVFile(duplicate_tsv).num_rows()

def populate_all_dataset_details():
    all_data = os.listdir('data/')
    for data in all_data:
        try:
            populate_dataset_details(data)
        except:
            continue

def update_labelmap(rows, num_rows, labelmap):
    '''
    labelmap is a hashset and can be added
    '''
    logging.info('updating labelmap')
    for row in tqdm(rows, total=num_rows):
        assert len(row) == 2
        try:
            labelmap.update(set([rect['class'] for rect in
                json.loads(row[1])]))
        except:
            labelmap.add(row[1])
    logging.info('done')

def derive_composite_meta_data(data, split, t):
    data_to_dataset = {}
    dataset = TSVDataset(data)
    def gen_rows():
        assert dataset.has(split, 'label')
        for key, _ in tqdm(dataset.iter_data(split, 'label')):
            c_data, c_split, c_key = parse_combine(key)
            if c_data in data_to_dataset:
                c_dataset = data_to_dataset[c_data]
            else:
                c_dataset = TSVDataset(c_data)
                data_to_dataset[c_data] = c_dataset
            assert c_dataset.has(c_split, t)
            _, t_data = c_dataset.seek_by_key(c_key, c_split, t)
            yield key, t_data
    assert not dataset.has(split, t)
    dataset.write_data(gen_rows(), split, t)

def populate_dataset_details(data, check_image_details=False):
    logging.info(data)
    dataset = TSVDataset(data)

    splits = ['trainval', 'train', 'test']

    # populate the height and with
    for split in splits:
        if dataset.has(split) and not dataset.has(split, 'hw') \
                and check_image_details:
            if op.isfile(dataset.get_data(split + 'X')):
                derive_composite_meta_data(data, split, 'hw')
            else:
                multi_thread = True
                if not multi_thread:
                    logging.info('generating hw')
                    rows = dataset.iter_data(split, progress=True)
                    dataset.write_data(((row[0], ' '.join(map(str,
                        img_from_base64(row[-1]).shape[:2]))) for 
                        row in rows), split, 'hw')
                else:
                    from pathos.multiprocessing import ProcessingPool as Pool
                    num_worker = 128
                    num_tasks = num_worker * 3
                    num_images = dataset.num_rows(split)
                    num_image_per_worker = (num_images + num_tasks - 1) // num_tasks 
                    assert num_image_per_worker > 0
                    all_idx = []
                    for i in range(num_tasks):
                        curr_idx_start = i * num_image_per_worker
                        if curr_idx_start >= num_images:
                            break
                        curr_idx_end = curr_idx_start + num_image_per_worker
                        curr_idx_end = min(curr_idx_end, num_images)
                        if curr_idx_end > curr_idx_start:
                            all_idx.append(range(curr_idx_start, curr_idx_end))
                    logging.info('creating pool')
                    m = Pool(num_worker)
                    def get_hw(filter_idx):
                        dataset = TSVDataset(data)
                        rows = dataset.iter_data(split, progress=True,
                                filter_idx=filter_idx)
                        return [(row[0], ' '.join(map(str,
                            img_from_base64(row[-1]).shape[:2]))) 
                            for row in rows]
                    all_result = m.map(get_hw, all_idx)
                    x = []
                    for r in all_result:
                        x.extend(r)
                    dataset.write_data(x, split, 'hw')

    # for each data tsv, generate the label tsv and the inverted file
    for split in splits:
        full_tsv = dataset.get_data(split)
        label_tsv = dataset.get_data(split, 'label')
        if not op.isfile(label_tsv) and op.isfile(full_tsv):
            extract_label(full_tsv, label_tsv)

    for split in splits:
        tsv_file = dataset.get_data(split)
        out_file = get_meta_file(tsv_file)
        if not op.isfile(out_file) and \
                dataset.has(split, 'hw') and \
                dataset.has(split):
            row_hw = dataset.iter_data(split, 'hw')
            row_label = dataset.iter_data(split, 'label')
            num_rows = dataset.num_rows(split)
            if check_image_details:
                details = tsv_details(row_hw, row_label, num_rows)
                write_to_yaml_file(details, out_file)

    labelmap = []
    # generate the label map if there is no
    if not op.isfile(dataset.get_labelmap_file()) and \
            not op.islink(dataset.get_labelmap_file()):
        logging.info('no labelmap, generating...')
        labelmap = []
        for split in splits:
            label_tsv = dataset.get_data(split, 'label', version=0)
            if not op.isfile(label_tsv):
                continue
            for row in tqdm(tsv_reader(label_tsv)):
                try:
                    labelmap.extend(set([rect['class'] for rect in
                        json.loads(row[1])]))
                except:
                    labelmap.append(row[1])
        if len(labelmap) == 0:
            logging.warning('there are no labels!')
        labelmap = sorted(list(set(labelmap)))
        logging.info('find {} labels'.format(len(labelmap)))
        need_update = False
        if op.isfile(dataset.get_labelmap_file()):
            origin_labelmap = dataset.load_labelmap()
            if len(origin_labelmap) == len(labelmap):
                for o in origin_labelmap:
                    if o not in labelmap:
                        need_update = True
                        break
            else:
                need_update = True
        else:
            need_update = True
        if need_update:
            logging.info('updating {}'.format(dataset.get_labelmap_file()))
            write_to_file('\n'.join(labelmap), dataset.get_labelmap_file())

    if not op.isfile(dataset.get_pos_labelmap_file()):
        ls = dataset.load_labelmap()
        write_to_file('\n'.join([l for l in ls if not l.startswith('-')]),
                dataset.get_pos_labelmap_file())

    # generate the rows with duplicate keys
    for split in splits:
        label_tsv = dataset.get_data(split, 'label')
        duplicate_tsv = dataset.get_data(split, 'key_duplicate')
        if op.isfile(label_tsv) and not op.isfile(duplicate_tsv):
            num_duplicate = detect_duplicate_key(label_tsv, duplicate_tsv)

    # generate lineidx if it is not generated
    for split in splits:
        lineidx = dataset.get_lineidx(split)
        full_tsv = dataset.get_data(split)
        if not op.isfile(lineidx) and op.isfile(full_tsv):
            logging.info('no lineidx for {}. generating...'.format(split))
            generate_lineidx(full_tsv, lineidx)

    # for each data tsv, generate the inverted file, and the labelmap
    for split in splits:
        v = 0
        while True:
            if not dataset.has(split, 'label', v):
                break
            if not dataset.has(split, 'labelmap', v) or \
                dataset.last_update_time(split, 'labelmap', v) < dataset.last_update_time(split, 'label', v):
                curr_labelmap = set()
                update_labelmap(dataset.iter_data(split, 'label', v),
                        dataset.num_rows(split),
                        curr_labelmap)
                curr_labelmap = sorted(list(curr_labelmap))
                dataset.write_data([[l] for l in curr_labelmap], split, 'labelmap', v)
            else:
                curr_labelmap = None
            if not dataset.has(split, 'inverted.label', v) or \
                    not dataset.has(split, 'inverted.label.with_bb', v) or \
                    not dataset.has(split, 'inverted.label.no_bb', v) or \
                    dataset.last_update_time(split, 'inverted.label', v) < dataset.last_update_time(split, 'label', v):
                if curr_labelmap is None:
                    curr_labelmap = []
                    for row in dataset.iter_data(split, 'labelmap', v):
                        assert len(row) == 1
                        curr_labelmap.append(row[0])
                def gen_inverted_rows(inv):
                    for label in inv:
                        assert label in curr_labelmap
                    for label in curr_labelmap:
                        i = inv[label] if label in inv else []
                        yield label, ' '.join(map(str, i))
                inverted, inverted_with_bb, inverted_no_bb = create_inverted_list(
                        dataset.iter_data(split, 'label', v))
                dataset.write_data(gen_inverted_rows(inverted),
                        split, 'inverted.label', v)
                dataset.write_data(gen_inverted_rows(inverted_with_bb),
                        split, 'inverted.label.with_bb', v)
                dataset.write_data(gen_inverted_rows(inverted_no_bb),
                        split, 'inverted.label.no_bb', v)
            v = v + 1
    # check if the number of rows from the label tsv are equal to the number of rows in image tsv
    for split in splits:
        v = 0
        num_rows = None
        while True:
            if not dataset.has(split, 'label', v):
                break
            if dataset.has(split, 'check_count', v):
                v = v + 1
                continue
            if num_rows is None:
                num_rows = dataset.num_rows(split)
            num_rows_in_label = dataset.num_rows(split, 'label', v)
            # we can remove the assert
            assert num_rows == num_rows_in_label
            if num_rows != num_rows_in_label:
                dataset.write_data([['num_rows', num_rows], ['num_rows_in_label', num_rows_in_label]],
                        split, 'check_count', v)
            else:
                dataset.write_data([], split, 'check_count', v)
            v = v + 1

    if not op.isfile(dataset.get_noffsets_file()):
        logging.info('no noffset file. generating...')
        labelmap = dataset.load_labelmap()
        mapper = LabelToSynset(True)
        ambigous = []
        ss = [mapper.convert(l) for l in labelmap]
        for l, (success, s) in zip(labelmap, ss):
            if not success and len(s) > 1:
                d = create_info_for_ambigous_noffset(l, [synset_to_noffset(s1)
                    for s1 in s])
                ambigous.append(d)
        if len(ambigous) > 0:
            logging.info('ambigous term which has no exact noffset: {}'.format(
                dataset.name))
            write_to_yaml_file(ambigous, dataset.get_noffsets_file() +
                    '.ambigous.yaml')
        noffsets = []
        for success, s in ss:
            if not success:
                noffsets.append('')
            else:
                noffsets.append(','.join([synset_to_noffset(o) for o in s]))
        write_to_file('\n'.join(noffsets), dataset.get_noffsets_file())

    if not op.isfile(dataset.get_labelmap_of_noffset_file()):
        noffsets = dataset.load_noffsets()
        all_line = []
        for noffset in noffsets:
            if len(noffset) == 0:
                all_line.append('unkown')
            else:
                ss = [noffset_to_synset(n) for n in noffset.split(',')]
                all_line.append(','.join([get_nick_name(s) for s in ss]))
        write_to_file('\n'.join(all_line),
                dataset.get_labelmap_of_noffset_file())

    # generate the label -> count tsv
    for split in splits:
        v = 0
        while True:
            label_idx_file = dataset.get_data(split, 'inverted.label', v)
            label_count_file = dataset.get_data(split, 'inverted.label.count', v)
            if op.isfile(label_idx_file):
                if not op.isfile(label_count_file):
                    label_idx = dataset.load_inverted_label_as_list(split, v)
                    tsv_writer(((l, str(len(i))) for l, i in label_idx), label_count_file)
            else:
                break
            v = v + 1

    if op.isfile(dataset.get_data('trainX')):
        # composite dataset.
        create_index_composite_dataset(dataset)

    populate_num_images_composite(dataset)

    add_node_to_ancestors(dataset)

    populate_all_label_counts(dataset)

def populate_all_label_counts(dataset):
    for split in ['train', 'trainval', 'test']:
        v = 0
        while True:
            if dataset.has(split, 'inverted.label.count', v):
                if not dataset.has(split, 'inverted.label.count.total', v):
                    count = sum([int(count) for _, count in dataset.iter_data(split,
                        'inverted.label.count', v)])
                    dataset.write_data([(str(count),)],
                            split,
                            'inverted.label.count.total', v)
                v = v + 1
            else:
                break

def add_node_to_ancestors(dataset):
    tree_file = op.join(dataset._data_root, 'root.yaml')
    out_file = op.join(dataset._data_root, 'treenode_to_ancestors.tsv')
    if op.isfile(tree_file) and worth_create(tree_file, out_file):
        tax = Taxonomy(load_from_yaml_file(tree_file))
        tax.update()
        tsv_writer([[name, ','.join(tax.name_to_ancestors_list[name])] for name in
            tax.name_to_ancestors_list], out_file)

def populate_num_images_composite(dataset):
    data = dataset.name
    data_with_bb = dataset.name + '_with_bb'
    data_no_bb = dataset.name + '_no_bb'
    datas = [data_with_bb, data_no_bb]
    datasets = [TSVDataset(d) for d in datas]
    suffixes = ['with_bb', 'no_bb']
    splits = ['train', 'test']
    dest_tree_file = op.join(dataset._data_root, 'root_enriched.yaml')
    src_tree_file = op.join(dataset._data_root, 'root.yaml')
    if op.isfile(dest_tree_file) or not op.isfile(src_tree_file):
        return
    tax = Taxonomy(load_from_yaml_file(src_tree_file))
    for split in splits:
        for suffix, d in zip(suffixes, datasets):
            if not d.has(split, 'inverted.label.count'):
                continue
            label_to_count_with_bb = {label: int(count) for label, count in d.iter_data(split,
                    'inverted.label.count')}
            for label in label_to_count_with_bb:
                count = label_to_count_with_bb[label]
                if not label.startswith('-'):
                    nodes = tax.root.search_nodes(name=label)
                else:
                    nodes = tax.root.search_nodes(name=label[1:])
                assert len(nodes) == 1, label
                node = nodes[0]
                if not label.startswith('-'):
                    node.add_feature('{}_{}'.format(suffix, split), count)
                else:
                    node.add_feature('{}_{}_neg'.format(suffix, split), count)
    write_to_yaml_file(tax.dump(), dest_tree_file)

def create_index_composite_dataset(dataset):
    split = 'train'
    splitx = split + 'X'
    fname_numImagesPerSource = dataset.get_data(splitx, 
            'numImagesPerSource')
    if op.isfile(fname_numImagesPerSource):
        return
    # how many images are contributed from each data source
    trainX_file = dataset.get_data(splitx)
    source_tsvs = load_list_file(trainX_file)

    shuffle_file = dataset.get_shuffle_file(split)
    if not op.isfile(shuffle_file):
        return
    rows = tsv_reader(shuffle_file)
    all_idxSource_idxRow = []
    for idx_source, idx_row in rows:
        all_idxSource_idxRow.append((int(idx_source), int(idx_row)))
    # note, the data may be duplicated.
    all_idxSource_idxRow = list(set(all_idxSource_idxRow))

    idxSource_to_idxRows = list_to_dict(all_idxSource_idxRow, 0)
    num_images_per_datasource = [None] * len(source_tsvs)
    num_srcimages_per_datasource = [TSVFile(s).num_rows() for s in source_tsvs]
    for idxSource in idxSource_to_idxRows:
        assert num_images_per_datasource[idxSource] is None
        num_images_per_datasource[idxSource] = len(idxSource_to_idxRows[idxSource])
    tsv_writer([(name, str(num)) for name, num in zip(source_tsvs,
        num_images_per_datasource)], fname_numImagesPerSource)
    dataset.write_data([(source_tsvs[i], num_images_per_datasource[i], num_srcimages_per_datasource[i]) 
        for i in range(len(source_tsvs))], splitx,
        'tsvsource.numdestimages.numsrcimages')

    # for each data source, how many labels are contributed and how many are
    # not
    source_dataset_names = [op.basename(op.dirname(t)) for t in source_tsvs]
    source_tsv_label_files = load_list_file(dataset.get_data(splitx,
        'origin.label'))
    source_tsv_labels = [TSVFile(t) for t in source_tsv_label_files]
    trainX_label_file = dataset.get_data(splitx, 'label')
    all_dest_label_file = load_list_file(trainX_label_file)
    dest_labels = [TSVFile(f) for f in all_dest_label_file]
    all_idxSource_sourceLabel_destLabel = []
    logging.info('each datasource and each idx row')
    idxSource_to_numRect = {}
    all_idxSource_numSourceRects_numDestRects = []
    for idx_source, idx_row in tqdm(all_idxSource_idxRow):
        source_rects = json.loads(source_tsv_labels[idx_source].seek(idx_row)[-1])
        dest_rects = json.loads(dest_labels[idx_source].seek(idx_row)[-1])
        if idx_source not in idxSource_to_numRect:
            idxSource_to_numRect[idx_source] = 0
        idxSource_to_numRect[idx_source] = idxSource_to_numRect[idx_source] + \
            len(dest_rects)
        all_idxSource_numSourceRects_numDestRects.append((
            idx_source, len(source_rects), len(dest_rects)))
        for r in dest_rects:
            all_idxSource_sourceLabel_destLabel.append((idx_source,
                r.get('class_from', r['class']), r['class']))
    idxSource_to_numSourceRects_numDestRects = list_to_dict(
            all_idxSource_numSourceRects_numDestRects, 0)
    idxSource_to_numSourceRects_numDestRect = {idxSource: [sum(x1 for x1, x2 in idxSource_to_numSourceRects_numDestRects[idxSource]), 
             sum(x2 for x1, x2 in idxSource_to_numSourceRects_numDestRects[idxSource])] 
        for idxSource in idxSource_to_numSourceRects_numDestRects}

    dataset.write_data([(source_tsvs[idxSource],
        idxSource_to_numSourceRects_numDestRect[idxSource][1],
        idxSource_to_numSourceRects_numDestRect[idxSource][0])
        for idxSource in range(len(source_tsvs))],
        splitx, 'tsvsource.numdestbbs.numsrcbbs')

    sourcetsv_to_num_rect = {source_tsvs[idx_source]: idxSource_to_numRect[idx_source] 
            for idx_source in idxSource_to_numRect}
    dataset.write_data([(s, sourcetsv_to_num_rect[s]) for s in source_tsvs],
        splitx, 'numRectsPerSource')
    idxSource_to_sourceLabel_destLabels = list_to_dict(
            all_idxSource_sourceLabel_destLabel, 0)
    source_numSourceLabels = [(s, 0) for s in source_tsvs]
    source_includedSourceLabels = [(s, []) for s in source_tsvs]
    for idxSource in idxSource_to_sourceLabel_destLabels:
        sourceLabel_destLabels = idxSource_to_sourceLabel_destLabels[idxSource]
        sourceLabel_to_destLabels = list_to_dict(sourceLabel_destLabels, 0)
        source_numSourceLabels[idxSource] = (source_tsvs[idxSource],
                len(sourceLabel_to_destLabels))
        source_includedSourceLabels[idxSource][1].extend(
                sourceLabel_to_destLabels.keys())
    dataset.write_data([(n, str(i)) for (n, i) in source_numSourceLabels], 
            splitx, 'numCategoriesPerSource')

    # save teh list of included labels
    sourceDataset_to_includedSourceLabels = {}
    for source, sourceLabels in source_includedSourceLabels:
        source_dataset_name = op.basename(op.dirname(source))
        if source_dataset_name not in sourceDataset_to_includedSourceLabels:
            sourceDataset_to_includedSourceLabels[source_dataset_name] = []
        sourceDataset_to_includedSourceLabels[source_dataset_name].extend(sourceLabels)
    for source_dataset_name in sourceDataset_to_includedSourceLabels:
        sourceDataset_to_includedSourceLabels[source_dataset_name] = \
                set(sourceDataset_to_includedSourceLabels[source_dataset_name])

    tsv_writer([(n, ','.join(sourceDataset_to_includedSourceLabels[n])) for n in
        sourceDataset_to_includedSourceLabels], op.join(dataset._data_root, 
            'trainX.includeCategoriesPerSourceDataset.tsv'))

    tsv_writer([(n, get_nick_name(noffset_to_synset(s)) if is_noffset(s) else
        s) for n in sourceDataset_to_includedSourceLabels
          for s in sourceDataset_to_includedSourceLabels[n]], 
          op.join(dataset._data_root, 'trainX.includeCategoriesPerSourceDatasetReadable.tsv'))

    #sourceDataset_to_excludeSourceLabels = {}
    ## find out the excluded label list
    #for source_dataset_name in sourceDataset_to_includedSourceLabels:
        #source_dataset = TSVDataset(source_dataset_name)
        #full_label_names = set(source_dataset.load_labelmap())
        #included_labels = sourceDataset_to_includedSourceLabels[source_dataset_name]
        #for l in included_labels:
            ## if l is not in the full_label_names, it will throw exceptions
            #full_label_names.remove(l)
        #sourceDataset_to_excludeSourceLabels[source_dataset_name] = full_label_names

    #tsv_writer([(n, ','.join(v)) for (n, v) in
        #sourceDataset_to_excludeSourceLabels.iteritems()], op.join(dataset._data_root,
            #'trainX.excludeCategoriesPerSourceDataset.tsv'))

    #tsv_writer([(n, get_nick_name(noffset_to_synset(s)) if is_noffset(s) else
        #s) for (n, v) in sourceDataset_to_excludeSourceLabels.iteritems() for s in v],
        #op.join(dataset._data_root, 'trainX.excludeCategoriesPerSourceDatasetReadable.tsv'))

class TSVTransformer(object):
    def __init__(self):
        self._total_rows = 0
        self._row_processor = None

    def ReadProcess(self, source_tsv, row_processor):
        self._row_processor = row_processor
        self._total_rows = 0

        rows = tsv_reader(source_tsv)
        x = [self._over_row_processor(row) for row in rows]

        logging.info('total rows = {}; total_processed = {}'.format(self._total_rows,
                len(x)))

    def Process(self, source_tsv, dst_tsv, row_processor):
        '''
        row_processor: a function whose input should be a list of tsv cols and
                      whose return is also a list of tsv colums (will be saved into dst_tsv)
        '''
        self._row_processor = row_processor
        self._total_rows = 0

        rows = tsv_reader(source_tsv)
        result = (self._over_row_processor(row) for row in rows)
        tsv_writer(result, dst_tsv)

        logging.info('total rows = {}'.format(self._total_rows))

    def _over_row_processor(self, row):
        out = self._row_processor(row)
        self._total_rows = self._total_rows + 1
        if self._total_rows % 500 == 0:
            logging.info('processed = {}'.format(self._total_rows))
        return out

def randomize_tsv_file(tsv_file):
    prefix = os.path.splitext(tsv_file)[0]
    shuffle_file = prefix + '.shuffle'
    if os.path.exists(shuffle_file):
        return shuffle_file
    idx_file = prefix + '.lineidx'
    with open(idx_file, 'r') as fp:
        num = len([line for line in fp.readlines() if len(line.strip()) > 0])
    np.random.seed(777)
    nums = np.random.permutation(num)
    result = '\n'.join(map(str, nums))
    with open(shuffle_file, 'w') as fp:
        fp.write(result)
    return shuffle_file

def gen_tsv_from_labeling(input_folder, output_folder):
    fs = glob.glob(op.join(input_folder, '*'))
    labels = set()
    def gen_rows():
        for f in fs:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            yaml_file = op.splitext(f)[0] + '.yaml'
            if not op.isfile(yaml_file):
                logging.info('{} not exist'.format(yaml_file))
            with open(yaml_file, 'r') as fp:
                bb_labels = yaml.loads(fp.read())
            for bb_label in bb_labels:
                labels.add(bb_label['class'])
            with open(f, 'r') as fp:
                encoded_im = base64.b64encode(fp.read())
            yield op.basename(f), json.dumps(bb_labels), encoded_im

    tsv_writer(gen_rows(), op.join(output_folder, 'train.tsv'))
    write_to_file('\n'.join(labels), op.join(output_folder, 'labelmap.txt'))

def try_json_parse(s):
    try:
        return json.loads(s)
    except ValueError:
        return s

def visualize_predict_no_draw(full_expid, predict_file, label, start_id,
        threshold):
    test_data, test_data_split = parse_test_data(predict_file)
    pred_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    pred_key_path = '{}.key.tsv'.format(pred_full_path)
    pred_label_path = '{}.labelmap.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_inverted_path = '{}.inverted.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_sorted_cache_path = '{}.key_idxGT_idxPred_ap.th{}.{}.tsv'.format(
            pred_full_path, threshold, label)

    test_dataset = TSVDataset(test_data)
    if not op.isfile(pred_sorted_cache_path):
        if not op.isfile(pred_key_path) or \
                not op.isfile(pred_label_path) or \
                not op.isfile(pred_inverted_path):
            logging.info('loading {}'.format(pred_full_path))
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(pred_full_path), threshold)
            pred_labels = inverted.keys()
            logging.info('writing {}'.format(pred_key_path))
            tsv_writer([[k] for k in pred_keys], pred_key_path)
            tsv_writer(([l] for l in pred_labels), pred_label_path)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_path)
        keys_from_pred = []
        labelmap = load_list_file(pred_label_path)
        pred_keys = load_list_file(pred_key_path)
        if label in labelmap:
            inverted_row = TSVFile(pred_inverted_path).seek(labelmap.index(label))
            assert inverted_row[0] == label
            assert len(inverted_row) == 2
            idx_from_pred = map(int, inverted_row[1].split(' '))
            keys_from_pred = [pred_keys[i] for i in idx_from_pred]
        else:
            keys_from_pred = []
        inverted_test_split = test_dataset.load_inverted_label(test_data_split, version=-1,
                label=label)
        if label in inverted_test_split:
            idx_from_gt = inverted_test_split[label]
        else:
            idx_from_gt = []
        rows = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=idx_from_gt, unique=True)
        keys_from_gt = [row[0] for row in rows]
        target_keys = list(set(keys_from_pred + keys_from_gt))
        target_keys = [k for k in target_keys if k in pred_keys]
        target_idx_in_pred = [pred_keys.index(k) for k in target_keys]
        gt_keys = test_dataset.load_keys(test_data_split)
        target_idx_in_gt = [gt_keys.index(k) for k in target_keys]
        rows_in_gt = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=target_idx_in_gt)
        pred_tsv = TSVFile(pred_full_path)
        rows_in_pred = (pred_tsv.seek(i) for i in target_idx_in_pred)
        target_aps = []
        for row_in_gt, row_in_pred in zip(rows_in_gt, rows_in_pred):
            assert row_in_gt[0] == row_in_pred[0]
            assert len(row_in_gt) == 2
            assert len(row_in_pred) == 2
            rects_gt = json.loads(row_in_gt[1])
            rects_pred = json.loads(row_in_pred[1])
            rects_gt = [r for r in rects_gt if r['class'] == label]
            rects_pred = [r for r in rects_pred if r['class'] == label]
            ap = calculate_image_ap([r['rect'] for r in rects_gt],
                    [r['rect'] for r in rects_pred])
            target_aps.append(ap)
        key_idxGT_idxPred_aps = zip(target_keys, target_idx_in_gt,
                target_idx_in_pred, target_aps)
        key_idxGT_idxPred_aps = sorted(key_idxGT_idxPred_aps, key=lambda x:
                x[-1])
        tsv_writer(key_idxGT_idxPred_aps, pred_sorted_cache_path)

    tsv = TSVFile(pred_sorted_cache_path)
    total_num = tsv.num_rows()
    if total_num == 0:
        return
    while start_id < 0:
        start_id = start_id + total_num
    while start_id >= total_num:
        start_id = start_id - total_num
    i = start_id
    tsv_pred = TSVFile(pred_full_path)
    for i in range(start_id, total_num):
        key, idx_gt, idx_pred, ap = tsv.seek(i)
        idx_gt, idx_pred, ap = int(idx_gt), int(idx_pred), float(ap)
        row_gt = next(test_dataset.iter_data(test_data_split,
            filter_idx=[idx_gt]))
        row_pred = tsv_pred.seek(idx_pred)
        assert row_gt[0] == row_pred[0], (row_gt[0], row_pred[0])

        rects_gt = json.loads(next(test_dataset.iter_data(test_data_split, 'label',
            filter_idx=[idx_gt], version=-1))[1])

        rects_pred = json.loads(row_pred[1])
        rects_pred = [r for r in rects_pred if r['conf'] > threshold]
        im_origin = img_from_base64(row_gt[-1])
        yield key, im_origin, rects_gt, rects_pred, ap

def visualize_predict(full_expid, predict_file, label, start_id, threshold):
    test_data, test_data_split = parse_test_data(predict_file)
    pred_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    pred_key_path = '{}.key.tsv'.format(pred_full_path)
    pred_label_path = '{}.labelmap.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_inverted_path = '{}.inverted.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_sorted_cache_path = '{}.key_idxGT_idxPred_ap.th{}.{}.tsv'.format(
            pred_full_path, threshold, label)

    test_dataset = TSVDataset(test_data)
    if not op.isfile(pred_sorted_cache_path):
        if not op.isfile(pred_key_path) or \
                not op.isfile(pred_label_path) or \
                not op.isfile(pred_inverted_path):
            logging.info('loading {}'.format(pred_full_path))
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(pred_full_path), threshold)
            pred_labels = inverted.keys()
            logging.info('writing {}'.format(pred_key_path))
            tsv_writer([[k] for k in pred_keys], pred_key_path)
            tsv_writer(([l] for l in pred_labels), pred_label_path)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_path)
        keys_from_pred = []
        labelmap = load_list_file(pred_label_path)
        pred_keys = load_list_file(pred_key_path)
        if label in labelmap:
            inverted_row = TSVFile(pred_inverted_path).seek(labelmap.index(label))
            assert inverted_row[0] == label
            assert len(inverted_row) == 2
            idx_from_pred = map(int, inverted_row[1].split(' '))
            keys_from_pred = [pred_keys[i] for i in idx_from_pred]
        else:
            keys_from_pred = []
        inverted_test_split = test_dataset.load_inverted_label(test_data_split, version=-1,
                label=label)
        if label in inverted_test_split:
            idx_from_gt = inverted_test_split[label]
        else:
            idx_from_gt = []
        rows = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=idx_from_gt, unique=True)
        keys_from_gt = [row[0] for row in rows]
        target_keys = list(set(keys_from_pred + keys_from_gt))
        target_keys = [k for k in target_keys if k in pred_keys]
        target_idx_in_pred = [pred_keys.index(k) for k in target_keys]
        gt_keys = test_dataset.load_keys(test_data_split)
        target_idx_in_gt = [gt_keys.index(k) for k in target_keys]
        rows_in_gt = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=target_idx_in_gt)
        pred_tsv = TSVFile(pred_full_path)
        rows_in_pred = (pred_tsv.seek(i) for i in target_idx_in_pred)
        target_aps = []
        for row_in_gt, row_in_pred in zip(rows_in_gt, rows_in_pred):
            assert row_in_gt[0] == row_in_pred[0]
            assert len(row_in_gt) == 2
            assert len(row_in_pred) == 2
            rects_gt = json.loads(row_in_gt[1])
            rects_pred = json.loads(row_in_pred[1])
            rects_gt = [r for r in rects_gt if r['class'] == label]
            rects_pred = [r for r in rects_pred if r['class'] == label]
            ap = calculate_image_ap([r['rect'] for r in rects_gt],
                    [r['rect'] for r in rects_pred])
            target_aps.append(ap)
        key_idxGT_idxPred_aps = zip(target_keys, target_idx_in_gt,
                target_idx_in_pred, target_aps)
        key_idxGT_idxPred_aps = sorted(key_idxGT_idxPred_aps, key=lambda x:
                x[-1])
        tsv_writer(key_idxGT_idxPred_aps, pred_sorted_cache_path)

    tsv = TSVFile(pred_sorted_cache_path)
    total_num = tsv.num_rows()
    if total_num == 0:
        return
    while start_id < 0:
        start_id = start_id + total_num
    while start_id >= total_num:
        start_id = start_id - total_num
    i = start_id
    tsv_pred = TSVFile(pred_full_path)
    for i in range(start_id, total_num):
        key, idx_gt, idx_pred, ap = tsv.seek(i)
        idx_gt, idx_pred, ap = int(idx_gt), int(idx_pred), float(ap)
        row_gt = next(test_dataset.iter_data(test_data_split,
            filter_idx=[idx_gt]))
        row_pred = tsv_pred.seek(idx_pred)
        assert row_gt[0] == row_pred[0], (row_gt[0], row_pred[0])

        rects_gt = json.loads(row_gt[1])
        rects_pred = json.loads(row_pred[1])
        rects_pred = [r for r in rects_pred if r['conf'] > threshold]
        rects_gt_target = [r for r in rects_gt if r['class'] == label]
        rects_pred_target = [r for r in rects_pred if r['class'] == label]
        if len(rects_gt_target) == 0 and len(rects_pred_target) == 0:
            logging.info('skipping to next')
            continue
        im_origin = img_from_base64(row_gt[-1])
        im_gt_target = np.copy(im_origin)
        draw_bb(im_gt_target, [r['rect'] for r in rects_gt_target],
                [r['class'] for r in rects_gt_target])
        im_pred_target = np.copy(im_origin)
        draw_bb(im_pred_target, [r['rect'] for r in rects_pred_target],
                [r['class'] for r in rects_pred_target])
        im_gt = np.copy(im_origin)
        draw_bb(im_gt, [r['rect'] for r in rects_gt],
                [r['class'] for r in rects_gt])
        im_pred = np.copy(im_origin)
        draw_bb(im_pred, [r['rect'] for r in rects_pred],
                [r['class'] for r in rects_pred])
        yield key, im_origin, im_gt_target, im_pred_target, im_gt, im_pred, ap

def visualize_box_no_draw(data, split, version, label, start_id, color_map={}):
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    if split is None:
        # guess which split should be used. only support non-composite tsv
        candidate_split = ['train', 'trainval', 'test']
        for c in candidate_split:
            if not op.isfile(dataset.get_data(c)):
                continue
            inverted = dataset.load_inverted_label(c, version, label)
            if label not in inverted:
                continue
            n = len(inverted[label])
            if n <= start_id:
                start_id = start_id - n
            else:
                logging.info('split = {}'.format(split))
                split = c
                break
        if not split:
            logging.info('cannot find the valid')
            return
    if label is not None:
        idx = dataset.load_inverted_label(split, version, label)[label]
    else:
        idx = list(range(dataset.num_rows(split, t='label', version=version)))
    if len(idx) == 0:
        return
    while start_id > len(idx):
        start_id = start_id - len(idx)
    while start_id < 0:
        start_id = start_id + len(idx)
    logging.info('start to read')
    rows_image = dataset.iter_data(split, filter_idx=idx[start_id:])
    rows_label = dataset.iter_data(split, 'label', version=version,
            filter_idx=idx[start_id:])
    for row_image, row_label in zip(rows_image, rows_label):
        key = row_image[0]
        assert key == row_label[0]
        assert len(row_image) == 3
        assert len(row_label) == 2
        label_str = row_label[-1]
        img_str = row_image[-1]
        im = img_from_base64(img_str)
        origin = np.copy(im)
        rects = try_json_parse(label_str)
        if type(rects) is list:
            def get_rect_class(rects):
                all_class = []
                all_rect = []
                for rect in rects:
                    label_class = rect['class']
                    rect = rect['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                return all_rect, all_class

            yield (key, origin, rects)
        else:
            yield key, origin, [{'class': label_str, 'rect': [0, 0, 0, 0]}]

def visualize_box(data, split, version, label, start_id, color_map={}):
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    if split is None:
        # guess which split should be used. only support non-composite tsv
        candidate_split = ['train', 'trainval', 'test']
        for c in candidate_split:
            if not op.isfile(dataset.get_data(c)):
                continue
            inverted = dataset.load_inverted_label(c, version, label)
            if label not in inverted:
                continue
            n = len(inverted[label])
            if n <= start_id:
                start_id = start_id - n
            else:
                logging.info('split = {}'.format(split))
                split = c
                break
        if not split:
            logging.info('cannot find the valid')
            return
    else:
        inverted = dataset.load_inverted_label(split, version, label)
    logging.info('inverted label loaded')
    logging.info('keys: {}'.format(inverted.keys()))
    if label not in inverted:
        return
    idx = inverted[label]
    if len(idx) == 0:
        return
    while start_id > len(idx):
        start_id = start_id - len(idx)
    while start_id < 0:
        start_id = start_id + len(idx)
    logging.info('start to read')
    rows_image = dataset.iter_data(split, filter_idx=idx[start_id:])
    rows_label = dataset.iter_data(split, 'label', version=version,
            filter_idx=idx[start_id:])
    for row_image, row_label in zip(rows_image, rows_label):
        key = row_image[0]
        assert key == row_label[0]
        assert len(row_image) == 3
        assert len(row_label) == 2
        label_str = row_label[-1]
        img_str = row_image[-1]
        im = img_from_base64(img_str)
        origin = np.copy(im)
        rects = try_json_parse(label_str)
        new_name = key.replace('/', '_').replace(':', '')
        if type(rects) is list:
            #rects = [l for l in rects if 'conf' not in l or l['conf'] > 0.3]
            def get_rect_class(rects):
                all_class = []
                all_rect = []
                for rect in rects:
                    label_class = rect['class']
                    rect = rect['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                return all_rect, all_class
            all_rect, all_class = get_rect_class(rects)
            draw_bb(im, all_rect, all_class)
            target_rects = [l for l in rects if l['class']
                    == label]
            all_rect, all_class = get_rect_class(target_rects)
            im_label = np.copy(origin)
            draw_bb(im_label, all_rect, all_class)
            yield new_name, origin, im_label, im, pformat(target_rects)
        else:
            yield new_name, origin, im, im, ''


def visualize_tsv2(data, split, label):
    '''
    by default, pass split as 'train'
    TODO: try to refactor it with visualize_box
    '''
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    inverted = dataset.load_inverted_label(split)
    logging.info('inverted label loaded')
    logging.info('keys: {}'.format(inverted.keys()))
    assert label in inverted
    idx = inverted[label]
    is_composite = False
    if split == 'train' and not op.isfile(dataset.get_data(split)):
        is_composite = True
        tsvs = [TSVFile(f) for f in dataset.get_train_tsvs()]
        shuffle_tsv_rows = tsv_reader(dataset.get_shuffle_file(split))
        shuffle = []
        for row in shuffle_tsv_rows:
            shuffle.append([int(row[0]), int(row[1])])
    else:
        tsv = TSVFile(dataset.get_data(split))
    num_image = 0
    num_rows = 2
    num_cols = 2
    num_image_one_fig = num_rows * num_cols
    idx.extend([0] * (num_image_one_fig - len(idx) % num_image_one_fig))
    idx = np.asarray(idx)
    idx = idx.reshape((-1, num_image_one_fig))
    logging.info('start to read')
    color_map = {}
    for i in idx:
        all_image = []
        for j in i:
            logging.info(j)
            if is_composite:
                row_image = tsvs[shuffle[j][0]].seek(shuffle[j][1])
            else:
                row_image = tsv.seek(j)
            im = img_from_base64(row_image[-1])
            labels = try_json_parse(row_image[1])
            num_image = num_image + 1
            if type(labels) is list:
                labels = [l for l in labels if 'conf' not in l or l['conf'] > 0.3]
                all_class = []
                all_rect = []
                for label in labels:
                    label_class = label['class']
                    rect = label['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                new_name = row_image[0].replace('/', '_').replace(':', '')
                draw_bb(im, all_rect, all_class, color=color_map)
            all_image.append(im)
        logging.info('start to show')
        show_images(all_image, num_rows, num_cols)

    logging.info('#image: {}'.format(num_image))

def visualize_tsv(tsv_image, tsv_label, out_folder=None, label_idx=1):
    '''
    deprecated, use visualize_tsv2
    '''
    rows_image = tsv_reader(tsv_image)
    rows_label = tsv_reader(tsv_label)
    assert out_folder == None or not op.exists(out_folder)
    source_folder = op.dirname(tsv_image)
    num_image = 0
    for row_image, row_label in zip(rows_image, rows_label):
        assert row_image[0] == row_label[0]
        im = img_from_base64(row_image[-1])
        labels = try_json_parse(row_label[label_idx])
        num_image = num_image + 1
        if type(labels) is list:
            labels = [l for l in labels if 'conf' not in l or l['conf'] > 0.3]
            all_class = []
            all_rect = []
            for label in labels:
                label_class = label['class']
                rect = label['rect']
                all_class.append(label_class)
                if not (rect[0] == 0 and rect[1] == 0
                        and rect[2] == 0 and rect[3] == 0):
                    all_rect.append(rect)
                else:
                    all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
            new_name = row_image[0].replace('/', '_').replace(':', '')
            draw_bb(im, all_rect, all_class)
            if out_folder:
                fname = os.path.join(out_folder,
                        '_'.join(set(c.replace(' ', '_') for c in all_class)),
                        new_name +'.png')
        else:
            fname = op.join(out_folder, row_image[0],
                    '{}_{}_{}.png'.format(num_image, labels, row_image[0]))
        if out_folder:
            save_image(im, fname)
        else:
            show_image(im)

    logging.info('#image: {}'.format(num_image))

def iou(wh1, wh2):
    w1, h1 = wh1
    w2, h2 = wh2
    return min(w1, w2) * min(h1, h2) / max(w1, w2) / max(h1, h2)

class ImageTypeParser(object):
    def __init__(self):
        self.m = None
        pass

    def parse_type(self, im_binary):
        return imghdr.what('', im_binary)
        self._ensure_init()
        mime_type = self.m.buffer(im_binary)
        t = op.basename(mime_type)
        return t

    def _ensure_init(self):
        if self.m is None:
            import magic
            #m = magic.open(magic.MAGIC_MIME_TYPE)
            m = magic.from_file(magic.MAGIC_MIME_TYPE)
            m.load()
            self.m = m

def collect_label(row, stat, **kwargs):
    labels = json.loads(row[1])
    labels = [label['class'] for label in labels]
    remove_labels = kwargs['remove_image'].split(',')
    is_remove = False
    for label in labels:
        if label in remove_labels or remove_labels == 'all':
            if random.random() <= kwargs['remove_image_prob']:
                is_remove = True
                break

    stat.append((is_remove, labels))

class TSVDatasetSource(TSVDataset):
    def __init__(self, name, root=None, version=-1,
            valid_splits=['train', 'trainval', 'test'], cleaness=10,
            use_all=False, use_negative_label=False):
        super(TSVDatasetSource, self).__init__(name)
        self._noffset_count = {}
        self._type = None
        self._root = root
        # the list of <datasetlabel, rootlabel>
        self._sourcelabel_to_targetlabels = None
        self._targetlabel_to_sourcelabels = None
        self._initialized = False
        self._type_to_datasetlabel_to_split_idx = None
        self._type_to_datasetlabel_to_count = None
        self._type_to_split_label_idx = None
        self._version = version
        self._valid_splits = valid_splits
        self._use_all = use_all
        self.cleaness = cleaness
        self.use_negative_label = use_negative_label 

    def populate_info(self, root):
        self._ensure_initialized()
        types = ['with_bb', 'no_bb']
        for node in root.iter_search_nodes():
            if root == node:
                continue
            if node.name in self._targetlabel_to_sourcelabels:
                sourcelabels = self._targetlabel_to_sourcelabels[node.name]
                node.add_feature(self.name, ','.join(sourcelabels))
                if any(is_noffset(l) for l in sourcelabels):
                    node.add_feature(self.name + '_readable',
                            ','.join(get_nick_name(noffset_to_synset(l)) if
                                is_noffset(l) else l for l in sourcelabels))
                total = 0
                for sourcelabel in sourcelabels:
                    for t in self._type_to_datasetlabel_to_count:
                        datasetlabel_to_count = self._type_to_datasetlabel_to_count[t]
                        c = datasetlabel_to_count.get(sourcelabel, 0)
                        total = total + c
                        key = 'images_{}'.format(t)
                        node.add_feature(key,
                                node.__getattribute__(key) + c)
                node.add_feature('{}_total'.format(self.name), total)

    def _ensure_initialized(self):
        if self._initialized:
            return
        populate_dataset_details(self.name)

        self.update_label_mapper()

        self._load_inverted()

        self._initialized = True

    def _load_inverted(self):
        # make sure self.update_label_mapper() is called
        types = ['with_bb', 'no_bb']
        self._type_split_label_idx = []
        for split in self._valid_splits:
            logging.info('loading the inverted file: {}-{}'.format(self.name,
                split))
            if not self.has(split, 'label', version=self._version):
                continue
            # load inverted list
            type_to_inverted = {}
            for i, t in enumerate(types):
                rows = self.iter_data(split, 'inverted.label.{}'.format(t), 
                        version=self._version)
                type_to_inverted[t] = {r[0]: map(int, r[1].split(' ')) for r in rows if 
                    r[0] in self._sourcelabel_to_targetlabels and 
                    len(r[1]) > 0}
            
            # register the positive labels
            for i, t in enumerate(types):
                inverted = type_to_inverted[t]
                inverted = {l: inverted[l] for l in inverted if not l.startswith('-')}
                label_idx = dict_to_list(inverted, 0)
                for label, idx in label_idx:
                    self._type_split_label_idx.append((t, split, label, idx))
                    # for no_bb, we need to add the with_bb into the list
                    if t == 'with_bb':
                        self._type_split_label_idx.append(('no_bb', split, label, idx))

            if self.use_negative_label:
                # currently, we only have a scenario where we have negative
                # annotations in the no_bb data source and the need to apply it
                # to no_bb target set.
                inverted = type_to_inverted['no_bb']
                inverted = {l: inverted[l] for l in inverted if l.startswith('-')}
                label_idx = dict_to_list(inverted, 0)
                for label, idx in label_idx:
                    #self._type_split_label_idx.append(('with_bb', split, label, idx))
                    self._type_split_label_idx.append(('no_bb', split, label, idx))

        self._type_split_label_idx = list(set(self._type_split_label_idx))
        self._type_to_split_label_idx = list_to_dict(
                self._type_split_label_idx, 0)
        self._type_to_datasetlabel_to_split_idx = {}
        for t in self._type_to_split_label_idx:
            split_label_idx = self._type_to_split_label_idx[t]
            label_to_split_idx = list_to_dict(split_label_idx, 1)
            self._type_to_datasetlabel_to_split_idx[t] = label_to_split_idx
        self._type_to_datasetlabel_to_count = {}
        for t in self._type_to_datasetlabel_to_split_idx:
            datasetlabel_to_split_idx = self._type_to_datasetlabel_to_split_idx[t]
            datasetlabel_to_count = {l: len(datasetlabel_to_split_idx[l]) for l in
                    datasetlabel_to_split_idx}
            self._type_to_datasetlabel_to_count[t] = datasetlabel_to_count

        self._split_to_num_image = {split: self.num_rows(split) for split in
                self._valid_splits if self.has(split)}

    def update_label_mapper(self):
        root = self._root
        # load the labelmap for all splits, self.load_labelmap is not correct,
        # since we will update the label and will not update the labelmap
        labelmap = []
        for split in self._valid_splits:
            if self.has(split, 'labelmap', self._version):
                for row in self.iter_data(split, 'labelmap', self._version):
                    labelmap.append(row[0])
        # if it has a prefix of -, it means it has no that tag.
        labelmap = [l for l in labelmap if not l.startswith('-')]
        hash_labelmap = set(labelmap)
        labelmap = list(hash_labelmap)

        tree_noffsets = {}
        for node in root.iter_search_nodes():
            if node == root or not node.noffset:
                continue
            for s in node.noffset.split(','):
                tree_noffsets[s] = node.name
        name_to_targetlabels = {}
        targetlabel_has_whitelist = set()
        invalid_list = []
        any_source_key = 'label_names_in_all_dataset_source'
        for node in root.iter_search_nodes():
            if node == root:
                continue
            if hasattr(node, self.name) or hasattr(node, any_source_key):
                if hasattr(node, self.name):
                    # this is like a white-list
                    values = node.__getattribute__(self.name)
                else:
                    values = node.__getattribute__(any_source_key)
                if values is not None:
                    source_terms = values.split(',')
                    for t in source_terms:
                        t = t.strip()
                        if t not in name_to_targetlabels:
                            name_to_targetlabels[t] = set()
                        if t not in hash_labelmap:
                            invalid_list.append((t, self.name, node.name))
                            continue
                        name_to_targetlabels[t].add(node.name)
                # even if it is None, we will also add it to white-list so that
                # we will not automatically match the term.
                targetlabel_has_whitelist.add(node.name)
            else:
                # we will keep the lower case always for case-insensitive
                # comparison
                all_candidate_src_names = [node.name.lower()]
                if hasattr(node, 'alias_names'):
                    all_candidate_src_names.extend([s.strip() for s in
                        node.alias_names.split(',')])
                for t in set(all_candidate_src_names):
                    if t not in name_to_targetlabels:
                        name_to_targetlabels[t] = set()
                    name_to_targetlabels[t].add(node.name)

        sourcelabel_targetlabel = []
        if len(invalid_list) != 0:
            logging.warn('invalid white list information: {}'.format(pformat(invalid_list)))

        #result = {}
        label_to_synset = LabelToSynset()
        for l in labelmap:
            matched = False
            if l.lower() in name_to_targetlabels:
                matched = True
                for t in name_to_targetlabels[l.lower()]:
                    sourcelabel_targetlabel.append((l, t))
            if l in name_to_targetlabels:
                for t in name_to_targetlabels[l]:
                    sourcelabel_targetlabel.append((l, t))
                matched = True
            if not matched:
                succeed, ns = label_to_synset.convert(l)
                if not succeed:
                    continue
                for n in ns:
                    n = synset_to_noffset(n)
                    if n in tree_noffsets:
                        t = tree_noffsets[n]
                        if t in targetlabel_has_whitelist:
                            # if it has white list, we will not respect the
                            # noffset to do the autmatic matching
                            continue
                        sourcelabel_targetlabel.append((l, t))
                        #result[l] = t
        if self.use_negative_label:
            # we just add the mapping here. no need to check if -s is in the
            # source label list
            sourcelabel_targetlabel.extend([('-' + s, '-' + t) for s, t in
                sourcelabel_targetlabel])

        self._sourcelabel_to_targetlabels = list_to_dict_unique(sourcelabel_targetlabel,
                0)
        self._targetlabel_to_sourcelabels = list_to_dict_unique(sourcelabel_targetlabel,
                1)


        return self._sourcelabel_to_targetlabels

    def select_tsv_rows(self, label_type):
        self._ensure_initialized()
        result = []
        if label_type in self._type_to_split_label_idx:
            split_label_idx = self._type_to_split_label_idx[label_type]
            datasetlabel_to_splitidx = list_to_dict(split_label_idx, 1) 
            for datasetlabel in datasetlabel_to_splitidx:
                if datasetlabel in self._sourcelabel_to_targetlabels:
                    split_idxes = datasetlabel_to_splitidx[datasetlabel]
                    targetlabels = self._sourcelabel_to_targetlabels[datasetlabel]
                    for rootlabel in targetlabels:
                        result.extend([(rootlabel, split, idx) for split, idx in
                            split_idxes])
        if self._use_all:
            split_to_rootlabel_idx = list_to_dict(result, 1)
            for s in split_to_rootlabel_idx:
                rootlabel_idxes = split_to_rootlabel_idx[s]
                idx_to_rootlabel = list_to_dict(rootlabel_idxes, 1)
                num_image = self._split_to_num_image[s]
                idxes = set(range(num_image)).difference(set(idx_to_rootlabel.keys()))
                for i in idxes:
                    # for these images, the root label is hard-coded as None
                    result.append((None, s, i))
            for s in self._valid_splits:
                if s in split_to_rootlabel_idx:
                    continue
                if s not in self._split_to_num_image:
                    continue
                result.extend([(None, s, i) for i in
                    range(self._split_to_num_image[s])])
        return result

def initialize_images_count(root):
    for node in root.iter_search_nodes():
        node.add_feature('images_with_bb', 0)
        node.add_feature('images_no_bb', 0)

def trainval_split(dataset, num_test_each_label):
    if op.isfile(dataset.get_train_tsv()):
        logging.info('skip to run trainval split for {} because it has been done'.format(
            dataset.name))
        return
    random.seed(777)
    label_to_idx = {}
    for i, row in enumerate(tsv_reader(dataset.get_trainval_tsv('label'))):
        rects = json.loads(row[1])
        if len(rects) == 0:
            logging.info('{} has empty label for {}'.format(dataset.name, i))
            continue
        random.shuffle(rects)
        label = rects[0]['class']
        if label in label_to_idx:
            label_to_idx[label].append(i)
        else:
            label_to_idx[label] = [i]

    test_idx = []
    train_idx = []
    for label in label_to_idx:
        if len(label_to_idx[label]) < num_test_each_label:
            logging.fatal('dataset {} has less than {} images for label {}'.
                    format(dataset.name, num_test_each_label, label))
        random.shuffle(label_to_idx[label])
        test_idx.extend(label_to_idx[label][: num_test_each_label])
        train_idx.extend(label_to_idx[label][num_test_each_label: ])

    trainval = TSVFile(dataset.get_trainval_tsv())

    def gen_train():
        for i in train_idx:
            row = trainval.seek(i)
            yield row
    tsv_writer(gen_train(), dataset.get_train_tsv())

    def gen_test():
        for i in test_idx:
            yield trainval.seek(i)
    tsv_writer(gen_test(), dataset.get_test_tsv_file())

def convert_one_label(rects, label_mapper):
    to_remove = []
    rects2 = []
    for rect in rects:
        if rect['class'] in label_mapper:
            for t in label_mapper[rect['class']]:
                r2 = copy.deepcopy(rect)
                assert type(t) is str or type(t) is unicode
                r2['class'] = t
                rects2.append(r2)
    rects[:] = []
    rects.extend(rects2)

def convert_label(label_tsv, idx, label_mapper, with_bb):
    tsv = TSVFile(label_tsv)
    result = None
    for i in tqdm(idx):
        row = tsv.seek(i)
        assert len(row) == 2
        rects = json.loads(row[1])
        def eval_with_bb(r):
            if not r['class'].startswith('-'):
                return 'rect' in r and any(x != 0 for x in r['rect'])
            else:
                return 'rect' not in r or all(x == 0 for x in r['rect'])
        if with_bb:
            # in the case with -, we will not add rect
                                # with all zeros, thus, no need to check if it is all zeros
                                # when it is negative samples
            rects = [r for r in rects if eval_with_bb(r)]
        # all annotations if eval_with_bb(r) is valid for no_bb. Thus, disable
        # the following
        #else:
            #rects = [r for r in rects if not eval_with_bb(r)]
        if result is None:
            # don't use this, because all list will be shared
            #result = [len(row) * ['d']] * tsv.num_rows()
            result = [None] * tsv.num_rows()
            for _ in range(len(result)):
                result[_] = ['d'] * len(row)
        rects2 = []
        # the following code should use convert_one_label
        for rect in rects:
            if rect['class'] in label_mapper:
                for t in label_mapper[rect['class']]:
                    r2 = copy.deepcopy(rect)
                    r2['class'] = t
                    if rect['class'] != t:
                        # keep this for logging
                        r2['class_from'] = rect['class']
                    rects2.append(r2)
        row[1] = rects2
        result[i] = row
    return result

def create_info_for_ambigous_noffset(name, noffsets):
    definitions = [str(noffset_to_synset(n).definition()) for n in noffsets]
    de = [{'noffset': n,
          'definition': d.replace("`", '').replace("'", ''),
          'nick_name': str(get_nick_name(noffset_to_synset(n)))}
            for n, d in zip(noffsets, definitions)]
    d = {'name': name,
            'definitions': de,
            'noffset': None,
            'markdown_url': create_markdown_url(noffsets)}
    return d

def node_should_have_images(root, th, fname):
    enough = True
    few_training_with_bb = []
    for node in root.iter_search_nodes():
        if node == root:
            continue
        if node.cum_images_with_bb < th:
            few_training_with_bb.append({'name': node.name, 
                'cum_images_with_bb': node.cum_images_with_bb,
                'parent list': [p.name for p in node.get_ancestors()[:-1]]})
            enough = False
            logging.warn('less images: {} ({})'.format(
                node.name.encode('utf-8'),
                node.cum_images_with_bb))
    if enough:
        logging.info('con. every node has at least {} images'.format(th))
    else:
        write_to_yaml_file(few_training_with_bb, fname)

def clean_dataset2(source_dataset_name, dest_dataset_name):
    source_dataset = TSVDataset(source_dataset_name)
    dest_dataset = TSVDataset(dest_dataset_name)
    splits = ['train', 'trainval', 'test']
    for split in splits:
        src_tsv = source_dataset.get_data(split)
        if op.isfile(src_tsv):
            valid_idxs = []
            dest_tsv = dest_dataset.get_data(split)
            def gen_rows():
                rows = tsv_reader(src_tsv)
                num_removed_images = 0
                num_removed_rects = 0
                for i, row in enumerate(tqdm(rows)):
                    if (i % 1000) == 0:
                        logging.info('{} - #removedImages={};#removedRects={}'.format(
                            i, num_removed_images, num_removed_rects))
                    im = img_from_base64(row[-1])
                    if im is None:
                        num_removed_images = num_removed_images + 1
                        continue
                    height, width = im.shape[:2]
                    rects = json.loads(row[1])
                    invalid = False
                    to_remove = []
                    for rect in rects:
                        r = rect['rect']
                        if all(s == 0 for s in r):
                            continue
                        changed = False
                        origin = copy.deepcopy(rect)
                        for j in range(4):
                            if r[j] < 0:
                                r[j] = 0
                                changed = True
                        for j in range(2):
                            if r[2 * j] >= width -1:
                                changed = True
                                r[2 * j] = width - 1
                            if [2 * j + 1] >= height - 1:
                                changed = True
                                r[2 * j + 1] = height - 1
                        if changed:
                            rect['changed_from_rect'] = origin
                        cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                        w, h = r[2] - r[0], r[3] - r[1]
                        if cx < 0 or cy < 0 or cx >= width or \
                                cy >= height or w <= 1 or h <= 1 or \
                                w >= width or h >= height:
                            to_remove.append(rect)
                    if len(to_remove) > 0:
                        num_removed_rects = num_removed_rects + len(to_remove)
                    for rect in to_remove:
                        rects.remove(rect)
                    valid_idxs.append(i)
                    if len(to_remove) > 0:
                        yield row[0], json.dumps(rects), row[2]
                    else:
                        yield row
            tsv_writer(gen_rows(), dest_tsv)
            dest_dataset.write_data(source_dataset.iter_data(split, 'label',
                filter_idx=valid_idxs), split, 'label')

def clean_dataset(source_dataset_name, dest_dataset_name):
    '''
    use version 2
    '''
    source_dataset = TSVDataset(source_dataset_name)
    dest_dataset = TSVDataset(dest_dataset_name)
    splits = ['train', 'trainval', 'test']
    for split in splits:
        src_tsv = source_dataset.get_data(split)
        if op.isfile(src_tsv):
            dest_tsv = dest_dataset.get_data(split)
            def gen_rows():
                rows = tsv_reader(src_tsv)
                num_removed_images = 0
                num_removed_rects = 0
                for i, row in enumerate(rows):
                    if (i % 1000) == 0:
                        logging.info('{} - #removedImages={};#removedRects={}'.format(
                            i, num_removed_images, num_removed_rects))
                    im = img_from_base64(row[-1])
                    height, width = im.shape[:2]
                    rects = json.loads(row[1])
                    invalid = False
                    to_remove = []
                    for rect in rects:
                        r = rect['rect']
                        if all(s == 0 for s in r):
                            continue
                        cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                        w, h = r[2] - r[0], r[3] - r[1]
                        if cx < 0 or cy < 0 or cx >= width or \
                                cy >= height or w <= 1 or h <= 1 or \
                                w >= width or h >= height:
                            to_remove.append(rect)
                    if len(to_remove) > 0:
                        logging.info('before removing {}'.format(len(rects)))
                        num_removed_rects = num_removed_rects + len(to_remove)
                    for rect in to_remove:
                        rects.remove(rect)
                        r = rect['rect']
                        logging.info('removing {}'.format(','.join(map(str,
                                r))))
                    if len(to_remove) > 0:
                        logging.info('after removing {}'.format(len(rects)))
                        if len(rects) > 0:
                            yield row[0], json.dumps(rects), row[2]
                        else:
                            num_removed_images = num_removed_images + 1
                            logging.info('removing image {}'.format(row[0]))
                    else:
                        yield row
            tsv_writer(gen_rows(), dest_tsv)

def parallel_convert_label(func, all_task, num_worker=128):
    num_split = num_worker
    num_task_each_split = (len(all_task) + num_split - 1) / num_split
    all_sub_tasks = []
    for i in range(num_split):
        start_idx = i * num_task_each_split
        if start_idx >= len(all_task):
            break
        end_idx = start_idx + num_task_each_split
        if end_idx > len(all_task):
            end_idx = len(all_task)
        all_sub_tasks.append(all_task[start_idx:end_idx])
    from pathos.multiprocessing import ProcessingPool as Pool
    m = Pool(num_worker)
    all_sub_results = m.map(func, all_sub_tasks)
    result = all_sub_results[0]
    for s in all_sub_results[1:]:
        assert len(result) == len(s)
        for i in range(len(result)):
            assert len(result[i]) == 2
            assert len(s[i]) == 2
            if result[i][1] == 'd' and s[i][1] != 'd':
                result[i][0] = s[i][0]
                result[i][1] = s[i][1]
            else:
                assert not (result[i][1] != 'd' and s[i][1] != 'd')
    return result

def parallel_map_to_array(func, all_task, num_worker=128):
    num_split = num_worker * 2
    num_task_each_split = (len(all_task) + num_split - 1) / num_split
    all_sub_tasks = []
    for i in range(num_split):
        start_idx = i * num_task_each_split
        if start_idx > len(all_task):
            break
        end_idx = start_idx + num_task_each_split
        if end_idx > len(all_task):
            end_idx = len(all_task)
        all_sub_tasks.append(all_task[start_idx:end_idx])
    from pathos.multiprocessing import ProcessingPool as Pool
    m = Pool(num_worker)
    all_sub_results = m.map(func, all_sub_tasks)
    result = []
    for s in all_sub_results:
        result.extend(s)
    return result

def create_trainX(train_ldtsi, extra_dtsi, tax, out_dataset,
        version=-1, lift_train=False):
    t_to_ldsi = list_to_dict(train_ldtsi, 2)
    extra_t_to_ldsi = list_to_dict(extra_dtsi, 1)
    train_ldtsik = []
    extra_dtsik = []
    shuffle_idx = []
    for label_type in t_to_ldsi:
        ldsi = t_to_ldsi[label_type]
        extra_ldsi = extra_t_to_ldsi.get(label_type, [])
        d_to_lsi = list_to_dict(ldsi, 1)
        extra_d_to_lsi = list_to_dict(extra_ldsi, 1)
        k = 0
        sources = []
        sources_origin_label = []
        sources_label = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            extra_lsi = extra_d_to_lsi.get(dataset, [])
            s_li = list_to_dict(lsi, 1)
            extra_s_li = list_to_dict(extra_lsi, 1)
            for split in s_li:
                li = s_li[split]
                idx_to_l = list_to_dict(li, 1)
                idx = idx_to_l.keys()
                extra_li = extra_s_li.get(split, [])
                # link the data tsv
                source = dataset.get_data(split)
                out_split = 'train{}'.format(k)
                train_ldtsik.extend([(l, dataset, label_type, split, i,
                    k) for l, i in li])
                extra_dtsik.extend([(dataset, label_type, split, i, k)
                    for l, i in extra_li])
                k = k + 1
                dest = out_dataset[label_type].get_data(
                        out_split)
                sources.append(source)
                logging.info('converting labels: {}-{}'.format(
                    dataset.name, split))
                source_origin_label = dataset.get_data(split,
                    'label', version=version)

                converted_label = convert_label(source_origin_label,
                        idx, dataset._sourcelabel_to_targetlabels,
                        with_bb=with_bb)
                sources_origin_label.append(source_origin_label)
                # convert the file name
                logging.info('delifting the labels')
                for i in tqdm(idx):
                    l = converted_label[i]
                    if lift_train:
                        l[1] = json.dumps(lift_one_image(l[1], tax))
                    else:
                        l[1] = json.dumps(delift_one_image(l[1], tax))
                    l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                label_file = out_dataset[label_type].get_data(out_split, 'label')
                logging.info('writing the label file {}'.format(label_file))
                tsv_writer(converted_label, label_file)
                sources_label.append(label_file)
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('trainX'))
        write_to_file('\n'.join(sources_label),
                out_dataset[label_type].get_data('trainX', 'label'))
        write_to_file('\n'.join(sources_origin_label),
                out_dataset[label_type].get_data('trainX', 'origin.label'))

    logging.info('saving the shuffle file')
    type_to_ldsik = list_to_dict(train_ldtsik, 2)
    extra_type_to_dsik = list_to_dict(extra_dtsik, 1)
    for label_type in type_to_ldsik:
        ldsik = type_to_ldsik[label_type]
        shuffle_info = [(str(k), str(i)) for l, d, s, i, k in ldsik]
        shuffle_info = list(set(shuffle_info))
        if label_type in extra_type_to_dsik:
            dsik = extra_type_to_dsik[label_type]
            # we should not de-duplicate it because it comes from the duplicate
            # policy
            extra_shuffle_info = [(str(k), str(i) ) for d, s, i, k in dsik]
            shuffle_info.extend(extra_shuffle_info)
        random.shuffle(shuffle_info)
        tsv_writer(shuffle_info,
                out_dataset[label_type].get_shuffle_file('train'))

    populate_output_num_images(train_ldtsik, 'toTrain', tax.root)

def create_testX(test_ldtsi, tax, out_dataset, version):
    t_to_ldsi = list_to_dict(test_ldtsi, 2)
    for label_type in t_to_ldsi:
        sources = []
        sources_origin_label = []
        sources_label = []
        ldsi = t_to_ldsi[label_type]
        d_to_lsi = list_to_dict(ldsi, 1)
        k = 0
        all_ki = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            s_to_li = list_to_dict(lsi, 1)
            for split in s_to_li:
                li = s_to_li[split]
                idx = list_to_dict(li, 1).keys()
                out_split = 'test{}'.format(k)
                s_file = dataset.get_data(split)
                sources.append(s_file)
                src_img_tsv = TSVFile(s_file)
                source_origin_label = dataset.get_data(split, 'label',
                        version=version)
                sources_origin_label.append(source_origin_label)
                converted_label = convert_label(source_origin_label,
                        idx, dataset._sourcelabel_to_targetlabels,
                        with_bb=with_bb)
                for i in tqdm(idx):
                    l = converted_label[i]
                    l[1] = json.dumps(lift_one_image(l[1], tax))
                    l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                all_ki.extend([(str(k), str(i)) for i in idx])
                label_file = out_dataset[label_type].get_data(out_split, 'label')
                tsv_writer(converted_label, label_file)
                sources_label.append(label_file)
                k = k + 1
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('testX'))
        write_to_file('\n'.join(sources_label),
                out_dataset[label_type].get_data('testX', 'label'))
        write_to_file('\n'.join(sources_origin_label),
                out_dataset[label_type].get_data('testX', 'origin.label'))
        tsv_writer(all_ki, out_dataset[label_type].get_shuffle_file('test'))

def remove_or_duplicate(train_ldtsi, min_image, label_to_max_image):
    label_to_dtsi = list_to_dict(train_ldtsi, 0)
    extra_dtsi = []
    for label in label_to_dtsi:
        dtsi = label_to_dtsi[label]
        max_image = label_to_max_image[label]
        if len(dtsi) > max_image:
            # first remove the images with no bounding box
            num_remove = len(dtsi) - max_image
            type_to_dsi = list_to_dict(dtsi, 1)
            if 'no_bb' in type_to_dsi:
                dsi = type_to_dsi['no_bb']
                if num_remove >= len(dsi):
                    # remove all this images
                    del type_to_dsi['no_bb']
                    num_remove = num_remove - len(dsi)
                else:
                    random.seed(9999)
                    random.shuffle(dsi)
                    type_to_dsi['no_bb'] = dsi[: len(dsi) - num_remove]
                    num_remove = 0
            if num_remove > 0:
                assert 'with_bb' in type_to_dsi
                dsi = type_to_dsi['with_bb']
                random.seed(9999)
                random.shuffle(dsi)
                dsi = sorted(dsi, key=lambda x: -x[0].cleaness)
                assert len(dsi) > num_remove
                type_to_dsi['with_bb'] = dsi[: len(dsi) - num_remove]
                num_remove = 0
            dtsi = dict_to_list(type_to_dsi, 1)
        elif len(dtsi) < min_image:
            num_duplicate = int(np.ceil(float(min_image) / len(dtsi)))
            logging.info('duplicate images for label of {}: {}->{}, {}'.format(
                label, len(dtsi), min_image, num_duplicate))
            extra_dtsi = (num_duplicate - 1) * dtsi
        label_to_dtsi[label] = dtsi
    logging.info('# train instances before duplication: {}'.format(len(train_ldtsi)))
    train_ldtsi = dict_to_list(label_to_dtsi, 0)
    logging.info('# train instances after duplication: {}'.format(len(train_ldtsi)))
    return train_ldtsi, extra_dtsi

def remove_test_in_train(train_ldtsi, test_ldtsi):
    logging.info('before len(train_ldtsi) = {}'.format(len(train_ldtsi)))
    set_train_dtsi = set((d, t, s, i) for l, d, t, s, i in train_ldtsi)
    if len(train_ldtsi) > 0 and type(train_ldtsi[0]) is list:
        # we need to convert it to immutable tuple since list is not hashable
        train_ldtsi = [(l, d, t, s, i) for l, d, t, s, i in train_ldtsi]
    set_train_ldtsi = set(train_ldtsi)
    #assert len(set_train_ldtsi) == len(train_ldtsi)
    set_test_dtsi = set((d, t, s, i) for l, d, t, s, i in test_ldtsi)
    result = [(l, d, t, s, i) for l, d, t, s, i in train_ldtsi
            if (d, t, s, i) not in set_test_dtsi]
    logging.info('after len(train_ldtsi) = {}'.format(len(result)))
    return result

def split_train_test(ldtsi, num_test):
    # group by label_type
    t_to_ldsi = list_to_dict(ldtsi, 2)
    train_ldtsi = []
    test_ldtsi = []
    for label_type in sorted(t_to_ldsi.keys()):
        ldsi= t_to_ldsi[label_type]
        l_to_dsi = list_to_dict(ldsi, 0)
        for rootlabel in sorted(l_to_dsi.keys()):
            dsi = l_to_dsi[rootlabel]
            if len(dsi) < num_test:
                logging.info('rootlabel={}; label_type={}->less than {} images'.format(
                    rootlabel, label_type, len(dsi)))
            curr_num_test = min(num_test, int(len(dsi) / 2))
            random.shuffle(dsi)
            test_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i
                in dsi[:curr_num_test]])
            train_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i
                in dsi[curr_num_test:]])
    return train_ldtsi, test_ldtsi

def regularize_data_sources(data_infos):
    result = []
    default_splits = ['train', 'trainval', 'test']
    for data_info in data_infos:
        if type(data_info) is str:
            r = {'name': data_info,
                    'cleaness': 10,
                    'valid_splits': default_splits,
                    'use_all': False}
            result.append(r)
        elif type(data_info) is tuple or type(data_info) is list:
            assert len(data_info) > 0
            r = {'name': data_info[0],
                    'cleaness': data_info[1] if len(data_info) > 1 else 10,
                    'valid_splits': data_info[2] if len(data_info) > 2 else
                                default_splits,
                    'use_all': data_info[3] if len(data_info) > 3 else False,
                    }
            assert len(data_info) < 5
            result.append(r)
        elif type(data_info) is dict:
            r = data_info
            result.append(r)
        else:
            raise Exception('unkwown data_info = {}'.format(data_info))
    return result

def parse_data_clean_splits(data_infos):
    '''
    use regularize_data_sources
    '''
    datas, cleaness, all_valid_splits = [], [], []
    default_splits = ['train', 'trainval', 'test']
    for data_info in data_infos:
        if type(data_info) is str:
            datas.append(data_info)
            cleaness.append(10)
            all_valid_splits.append(default_splits)
        elif type(data_info) is tuple or type(data_info) is list:
            assert len(data_info) > 0
            datas.append(data_info[0])
            cleaness.append(data_info[1] if len(data_info) > 1 else 10)
            all_valid_splits.append(data_info[2] if len(data_info) > 2 else default_splits)
        elif type(data_info) is dict:
            datas.append(data_info['data'])
            cleaness.append(data_info.get('cleaness', 10))
            all_valid_splits.append(data_info.get('valid_splits', default_splits))
        else:
            raise Exception('unkwown data_info = {}'.format(data_info))
    return datas, cleaness, all_valid_splits

def attach_properties(src_nodes, dst_tree):
    name_to_dst_node = {n.name: n for n in dst_tree.iter_search_nodes() if
            n != dst_tree}
    confusings = []
    for src_node in src_nodes:
        if src_node.name not in name_to_dst_node:
            continue
        dst_node = name_to_dst_node[src_node.name]
        for f in src_node.features:
            if f in ['support', 'name', 'dist', 'sub_group']:
                continue
            if f in dst_node.features:
                if dst_node.__getattribute__(f) == src_node.__getattribute__(f):
                    continue
                else:
                    confusings.append({'name': src_node.name,
                        'feature': f,
                        'value in tree': dst_node.__getattribute__(f),
                        'value in property list': src_node.__getattribute__(f)})
            else:
                dst_node.add_feature(f, src_node.__getattribute__(f))
    assert len(confusings) == 0, pformat(confusings)

def build_taxonomy_impl(taxonomy_folder, **kwargs):
    random.seed(777)
    dataset_name = kwargs.get('data',
            op.basename(taxonomy_folder))
    overall_dataset = TSVDataset(dataset_name)
    if op.isfile(overall_dataset.get_labelmap_file()):
        logging.info('ignore to build taxonomy since {} exists'.format(
            overall_dataset.get_labelmap_file()))
        return
    init_logging()
    logging.info('building {}'.format(dataset_name))
    all_tax = load_all_tax(taxonomy_folder)
    tax = merge_all_tax(all_tax)
    tax.update()
    initialize_images_count(tax.root)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k_448', tax.root)
    if op.isfile(imagenet22k.get_labelmap_file()):
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('there is no imagenet22k_448 dataset to help identify the noffset')
    populate_url_for_offset(tax.root)

    ambigous_noffset_file = op.join(overall_dataset._data_root,
            'ambigous_noffsets.yaml')
    output_ambigous_noffsets(tax.root, ambigous_noffset_file)

    data_infos = regularize_data_sources(kwargs['datas'])

    logging.info('extract the images from: {}'.format(pformat(data_infos)))

    label_version = kwargs.get('version', -1)
    data_sources = [TSVDatasetSource(root=tax.root, version=label_version, **d)
            for d in data_infos]

    for s in data_sources:
        s.populate_info(tax.root)

    populate_cum_images(tax.root)

    labels, child_parent_sgs = child_parent_print_tree2(tax.root, 'name')

    label_map_file = overall_dataset.get_labelmap_file() 
    write_to_file('\n'.join(labels), label_map_file)
    # save the parameter
    write_to_yaml_file((taxonomy_folder, kwargs), op.join(overall_dataset._data_root,
            'generate_parameters.yaml'))
    dest_taxonomy_folder = op.join(overall_dataset._data_root,
            'taxonomy_folder')
    if op.isdir(dest_taxonomy_folder):
        shutil.rmtree(dest_taxonomy_folder)
    shutil.copytree(taxonomy_folder, dest_taxonomy_folder)

    out_dataset = {'with_bb': TSVDataset(dataset_name + '_with_bb'),
            'no_bb': TSVDataset(dataset_name + '_no_bb')}

    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_labelmap_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(label_map_file, target_file)

    logging.info('cum_images_with_bb: {}'.format(tax.root.cum_images_with_bb))
    logging.info('cum_images_no_bb: {}'.format(tax.root.cum_images_no_bb))

    # write the simplified version of the tree
    dest = op.join(overall_dataset._data_root, 'root.simple.yaml')
    write_to_yaml_file(tax.dump(['images_with_bb']), dest)

    tree_file = overall_dataset.get_tree_file()
    write_to_file('\n'.join(['{} {}{}'.format(c, p, '' if sg < 0 else ' {}'.format(sg))
                             for c, p, sg in child_parent_sgs]),
            tree_file)
    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_tree_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(tree_file, target_file)

    node_should_have_images(tax.root, 200,
            op.join(overall_dataset._data_root, 'labels_with_few_images.yaml'))

    # get the information of all train val
    ldtsi = []
    logging.info('collecting all candidate images')
    for label_type in out_dataset:
        for dataset in data_sources:
            targetlabel_split_idxes = dataset.select_tsv_rows(label_type)
            for rootlabel, split, idx in targetlabel_split_idxes:
                ldtsi.append((rootlabel, dataset, label_type, split, idx))
    # we need to remove the duplicates. the duplicates could come from such
    # cases: for example, we have Laptop and laptop in the image. Both of the
    # labels are mapped to laptop, which is in the target domain. In this case,
    # the image could be in the list twice
    ldtsi = list(set(ldtsi))

    num_test = kwargs.get('num_test', 50)

    # for each label, let's duplicate the image or remove the image
    default_max_image = kwargs.get('max_image_per_label', 1000)
    label_to_max_image = {n.name: n.__getattribute__('max_image_extract_for_train')
            if 'max_image_extract_for_train' in n.features and n.__getattribute__('max_image_extract_for_train') > default_max_image
            else default_max_image for n in tax.root.iter_search_nodes() if n != tax.root}
    label_to_max_image = {l: max(label_to_max_image[l], num_test) for l in label_to_max_image}
    # negative images constraint
    labels = list(label_to_max_image.keys())
    for l in labels:
        label_to_max_image['-' + l] = label_to_max_image[l]
    label_to_max_image[None] = 10000000000
    min_image = kwargs.get('min_image_per_label', 200)

    logging.info('keep a small image pool to split')
    label_to_max_augmented_images = {l: label_to_max_image[l] * 3 for l in label_to_max_image}
    # reduce the computing cost
    ldtsi, extra_dtsi = remove_or_duplicate(ldtsi, 0,
            label_to_max_augmented_images)
    assert len(extra_dtsi) == 0

    logging.info('select the best test image')
    if num_test == 0:
        test_ldtsi = []
    else:
        # generate the test set from the best data source
        label_to_max_images_for_test = {l: num_test for l in
            label_to_max_image}
        test_ldtsi, extra_dtsi = remove_or_duplicate(ldtsi, 0,
                label_to_max_images_for_test)
        assert len(extra_dtsi) == 0

    logging.info('removing test images from image pool')
    train_ldtsi = remove_test_in_train(ldtsi, test_ldtsi)

    logging.info('select the final training images')
    train_ldtsi, extra_dtsi = remove_or_duplicate(train_ldtsi, min_image,
            label_to_max_image)

    logging.info('creating the train data')
    create_trainX(train_ldtsi, extra_dtsi, tax, out_dataset,
            version=label_version, lift_train=kwargs.get('lift_train', False))

    populate_output_num_images(test_ldtsi, 'toTest', tax.root)

    # dump the tree to yaml format
    dest = op.join(overall_dataset._data_root, 'root.yaml')
    d = tax.dump()
    write_to_yaml_file(d, dest)
    for label_type in out_dataset:
        target_file = op.join(out_dataset[label_type]._data_root, 'root.yaml')
        ensure_directory(op.dirname(target_file))
        shutil.copy(dest, target_file)

    create_testX(test_ldtsi, tax, out_dataset, version=label_version)

    logging.info('done')

def delift_one_image(rects, tax):
    # currently, for the training, we need to delift the label. That is, if it
    # is man, we should remove the person label
    rects2 = []
    for curr_r in rects:
        if curr_r['class'].startswith('-'):
            # if it is a background, we do nothing here. In the future, we
            # might change this logic
            rects2.append(curr_r)
            continue
        curr_label = curr_r['class']
        if 'rect' not in curr_r:
            same_place_rects = rects2
        else:
            ious = [calculate_iou(r['rect'], curr_r['rect']) for r in rects2]
            same_place_rects = [r for i, r in zip(ious, rects2) if i > 0.9]
        # if current label is one of parent of the same_place rects, ignore it
        ignore = False
        for same_place_r in same_place_rects:
            ancestors = tax.name_to_ancestors[same_place_r['class']]
            if curr_label in ancestors or same_place_r['class'] == curr_label:
                ignore = True
                break
        if ignore:
            continue
        ancestors = tax.name_to_ancestors[curr_label]
        to_removed = []
        for same_place_r in same_place_rects:
            if same_place_r['class'] in ancestors:
                to_removed.append(same_place_r)
        for t in to_removed:
            rects2.remove(t)
        rects2.append(curr_r)
    return rects2

def lift_one_image(rects, tax):
    rects2 = []
    for curr_r in rects:
        if curr_r['class'].startswith('-'):
            rects2.append(curr_r)
            continue
        label = curr_r['class']
        all_label = tax.name_to_ancestors[label]
        all_label.add(label)
        for l in all_label:
            same_label_rects = [r for r in rects2 if r['class'] == l]
            if 'rect' in curr_r:
                ious = [calculate_iou(r['rect'], curr_r['rect']) for r in
                    same_label_rects if 'rect' in r]
                if len(ious) > 0 and max(ious) > 0.9:
                    continue
                else:
                    r = copy.deepcopy(curr_r)
                    r['class'] = l
                    rects2.append(r)
            else:
                if len(same_label_rects) == 0:
                    r = copy.deepcopy(curr_r)
                    r['class'] = l
                    rects2.append(r)
    return rects2

def populate_output_num_images(ldtX, suffix, root):
    label_to_node = {n.name: n for n in root.iter_search_nodes() if n != root}
    targetlabel_to_dX = list_to_dict(ldtX, 0)
    for targetlabel in targetlabel_to_dX:
        if not targetlabel or targetlabel.startswith('-'):
            # currently, we ignore this background case. In the future, we
            # might change this logic
            continue
        if not targetlabel:
            # it means background images
            continue
        dtX = targetlabel_to_dX[targetlabel]
        dataset_to_X = list_to_dict(dtX, 0)
        for dataset in dataset_to_X:
            X = dataset_to_X[dataset]
            if len(X) == 0:
                continue
            key = '{}_{}'.format(dataset.name, suffix)
            value = len(X)
            label_to_node[targetlabel].add_feature(key, value)
        labeltype_to_dX = list_to_dict(dtX, 1)
        for labeltype in labeltype_to_dX:
            dX = labeltype_to_dX[labeltype]
            key = '{}_{}'.format(labeltype, suffix)
            value = len(dX)
            label_to_node[targetlabel].add_feature(key, value)

def output_ambigous_noffsets(root, ambigous_noffset_file):
    ambigous = []
    for node in root.iter_search_nodes():
        if hasattr(node, 'noffsets') and node.noffset is None:
            noffsets = node.noffsets.split(',')
            d = create_info_for_ambigous_noffset(node.name, noffsets)
            d['parent_name'] = ','.join([n.name for n in
                        node.get_ancestors()[:-1]])
            ambigous.append(d)
    if len(ambigous) > 0:
        logging.info('output ambigous terms to {}'.format(ambigous_noffset_file))
        write_to_yaml_file(ambigous, ambigous_noffset_file)
    else:
        logging.info('Congratulations on no ambigous terms.')

def output_ambigous_noffsets_main(tax_input_folder, ambigous_file_out):
    all_tax = load_all_tax(tax_input_folder)
    tax = merge_all_tax(all_tax)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k')
    if op.isfile(imagenet22k.get_labelmap_file()):
        logging.info('remove the noffset if it is not in imagenet22k')
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('no imagenet22k data used to help remove noffset ambiguities')

    populate_url_for_offset(tax.root)

    output_ambigous_noffsets(tax.root, ambigous_file_out)

def convert_inverted_file(name):
    d = TSVDataset(name)
    splits = ['train', 'trainval', 'test']
    for split in splits:
        logging.info('loading {}-{}'.format(name, split))
        x = d.load_inverted_label(split)
        def gen_rows():
            for label in x:
                idx = x[label]
                yield label, ' '.join(map(str, idx))

        inverted_file = d.get_data(split, 'inverted.label')
        target_file = op.splitext(inverted_file)[0] + '.tsv'
        if not op.isfile(inverted_file):
            if op.isfile(target_file):
                os.remove(target_file)
            continue
        if op.isfile(target_file):
            continue
        tsv_writer(gen_rows(), target_file)

def standarize_crawled(tsv_input, tsv_output):
    rows = tsv_reader(tsv_input)
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info(i)
            image_str = row[-1]
            image_label = row[0]
            rects = [{'rect': [0, 0, 0, 0], 'class': image_label}]
            image_name = '{}_{}'.format(op.basename(tsv_input), i)
            yield image_name, json.dumps(rects), image_str
    tsv_writer(gen_rows(), tsv_output)

def get_data_sources(public_only=False, version=None):
    '''
    gradually deprecate the parameter of public_only. Use the parameter of
    version
    '''
    if version:
        if version=='exclude_golden_test':
            return [
                    {
                        'name': 'coco2017', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10
                    },
                    {
                        'name': 'voc0712', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10
                    },
                    {
                        'name': 'OpenImageV4_448', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10
                    },
                    ('Naturalist', 10),
                    ('elder', 10),
                    ('imagenet200Diff', 10),
                    ('open_images_clean_1', 9),
                    ('open_images_clean_2', 9),
                    ('open_images_clean_3', 9),
                    ('imagenet1kLocClean', 9),
                    ('imagenet3k_448Clean', 9),
                    ('VisualGenomeClean', 9),
                    ('brand1048Clean', 8),
                    ('mturk700_url_as_keyClean', 8),
                    ('crawl_office_v1', 8),
                    ('crawl_office_v2', 8),
                    ('Materialist', 8),
                    ('MSLogoClean', 8),
                    ('clothingClean', 8),
                    ('imagenet22k_448', 7),
                    ('4000_Full_setClean', 7),
            ]
        elif version == 'exclude_golden_use_voc_coco_all':
            return [
                    {
                        'name': 'coco2017', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'voc0712', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'OpenImageV4_448', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 9
                    },
                    ('Naturalist', 9),
                    ('elder', 9),
                    ('imagenet200Diff', 9),
                    ('open_images_clean_1', 8),
                    ('open_images_clean_2', 8),
                    ('open_images_clean_3', 8),
                    ('imagenet1kLocClean', 8),
                    ('imagenet3k_448Clean', 8),
                    ('VisualGenomeClean', 8),
                    ('brand1048Clean', 7),
                    ('mturk700_url_as_keyClean', 7),
                    ('crawl_office_v1', 7),
                    ('crawl_office_v2', 7),
                    ('Materialist', 7),
                    ('MSLogoClean', 7),
                    ('clothingClean', 7),
                    ('imagenet22k_448', 6),
                    ('4000_Full_setClean', 6),
            ]
        elif version == 'logo':
            return [
                    {
                        'name': 'coco2017', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'voc0712', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'brand1048Clean',
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 7,
                        'use_all': True,
                    },
            ]
        elif version == 'logo2':
            return [
                    {
                        'name': 'coco2017', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'voc0712', 
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'brand1048',
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 7,
                        'use_all': True,
                    },
                    {
                        'name': 'FlickrLogos-32',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'BelgaLogos',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'FlickrLogos_47',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'LogosInTheWild-v2Clean',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'openlogo',
                        'cleaness': 10,
                        'use_all': True,
                    },
            ]
        elif version == 'logo2_novoccoco':
            return [
                    {
                        'name': 'brand1048',
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 7,
                        'use_all': True,
                    },
                    {
                        'name': 'FlickrLogos-32',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'BelgaLogos',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'FlickrLogos_47',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'LogosInTheWild-v2Clean',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'openlogo',
                        'cleaness': 10,
                        'use_all': True,
                    },
            ]
        elif version == 'logo_db_only':
            return [
                    {
                        'name': 'brand1048',
                        'valid_splits': ['train', 'trainval'],
                        'cleaness': 7,
                        'use_all': True,
                    },
                    {
                        'name': 'FlickrLogos-32',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'BelgaLogos',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'FlickrLogos_47',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'LogosInTheWild-v2Clean',
                        'cleaness': 10,
                        'use_all': True,
                    },
                    {
                        'name': 'openlogo',
                        'cleaness': 10,
                        'use_all': True,
                    },
            ]
        else:
            raise ValueError('Unknown version = {}'.format(version))
    if not public_only:
        return [
            ('coco2017', 10),
            ('voc0712', 10),
            ('Naturalist', 10),
            ('elder', 10),
            ('imagenet200Diff', 10),
            ('OpenImageV4_448', 10),
            ('open_images_clean_1', 9),
            ('open_images_clean_2', 9),
            ('open_images_clean_3', 9),
            ('imagenet1kLocClean', 9),
            ('imagenet3k_448Clean', 9),
            ('VisualGenomeClean', 9),
            ('brand1048Clean', 8),
            ('mturk700_url_as_keyClean', 8),
            ('crawl_office_v1', 8),
            ('crawl_office_v2', 8),
            ('Materialist', 8),
            ('MSLogoClean', 8),
            ('clothingClean', 8),
            ('imagenet22k_448', 7),
            ('4000_Full_setClean', 7),
            ]
    else:
        return [
            ('coco2017', 10),
            ('voc0712', 10),
            ('Naturalist', 10),
            ('elder', 10),
            ('imagenet200Diff', 10),
            ('OpenImageV4_448', 10),
            #('open_images_clean_1', 9),
            #('open_images_clean_2', 9),
            #('open_images_clean_3', 9),
            ('imagenet1kLocClean', 9),
            ('imagenet3k_448Clean', 9),
            ('VisualGenomeClean', 9),
            ('brand1048Clean', 8),
            #('mturk700_url_as_keyClean', 8),
            #('crawl_office_v1', 8),
            #('crawl_office_v2', 8),
            ('Materialist', 8),
            #('4000_Full_setClean', 7),
            #('MSLogoClean', 8),
            #('clothingClean', 8),
            ('imagenet22k_448', 7),
            ]


def get_img_url2(img_key):
    # don't use get_image_url
    clean_name = _map_img_key_to_name2(img_key)
    url = _get_url_from_name(clean_name)
    return url

def get_img_url(img_key):
    # use version 2
    clean_name = _map_img_key_to_name(img_key)
    url = _get_url_from_name(clean_name)
    return url

def _map_img_key_to_name2(key):
    assert len(key) > 0
    ext = '.jpg'
    pattern = '^([0-9]|-|[a-z]|[A-Z]|\.|_)*$'
    if not re.match(pattern, key):
        key = hash_sha1(key)
    key = key.lower()
    if key.endswith(ext):
        return key
    else:
        return key + ext

def _map_img_key_to_name(key):
    # use version 2
    _EXT = ".jpg"
    if key.startswith("brand"):
        return "brand" + str(hash(key)) + _EXT
    else:
        key = key.lower()
        if key.endswith(_EXT):
            return key
        else:
            return key + _EXT

def _get_url_from_name(name):
    _SITE = 'https://cogsimagestorage.blob.core.windows.net/'
    _CONTAINER_NAME = "detectortrain"
    return _SITE + _CONTAINER_NAME + "/" + name

def parse_combine(key):
    pattern = '(.*?)_(train|trainval|test)_(.*)'
    result = re.match(pattern, key)
    if result is None:
        return None, None, key
    else:
        return result.groups()

def convert_to_uhrs_with_url(data):
    dataset = TSVDataset(data)
    for split in ['train', 'trainval', 'test']:
        if not dataset.has(split, 'label'):
            continue
        v = dataset.get_latest_version(split, 'label')
        def gen_rows():
            for row in dataset.iter_data(split, 'label', version=v):
                key = row[0]
                _, _, key = parse_combine(key)
                row.append(get_img_url(key))
                yield row
        dataset.write_data(gen_rows(), split,
                'url', version=v)

def find_same_location_rects(target, rects, iou=0.95):
    return [r for r in rects if 
        calculate_iou(target['rect'], r['rect']) > iou]

def find_same_rects(target, rects, iou=0.95):
    same_class_rects = [r for r in rects if r['class'] == target['class']]
    return [r for r in same_class_rects if
        calculate_iou(target['rect'], r['rect']) > iou]

def rect_in_rects(target, rects, iou=0.95):
    same_class_rects = [r for r in rects if r['class'] == target['class']]
    if 'rect' not in target:
        return len(same_class_rects) > 0
    else:
        return any(r for r in same_class_rects if 'rect' in r and
            calculate_iou(target['rect'], r['rect']) > iou)

def strict_rect_in_rects(target, rects):
    return any(target == r for r in rects)

def load_key_rects(iter_data):
    result = []
    logging.info('loading key rects')
    for row in tqdm(iter_data):
        assert len(row) == 2
        result.append([row[0], json.loads(row[1])])
    return result

def convert_uhrs_result_back_to_sources(in_tsv, debug=True, tree_file=None):
    rows = tsv_reader(in_tsv)
    key_rects3 = []
    num_yes, num_no, num_un = 0, 0, 0
    for row in rows:
        # key, yes, no, uncertain
        assert len(row) == 4
        rects_yes = json.loads(row[1])
        rects_no = json.loads(row[2])
        rects_un = json.loads(row[3])
        num_yes = num_yes + len(rects_yes)
        num_no = num_no + len(rects_no)
        num_un = num_un + len(rects_un)
        key_rects3.append([row[0], [rects_yes, rects_no, rects_un]])

    logging.info('#yes={}; #no={}; #un={}; #yes/(#yes+#no+#un)={}'.format(
        num_yes, num_no, num_un, 1.*num_yes/(num_yes+num_no+num_un)))

    if tree_file:
        tax = Taxonomy(load_from_yaml_file(tree_file))
        mapper = LabelToSynset()
        mapper.populate_noffset(tax.root)

    datas = get_data_sources()
    datas = [data for data, _ in datas]
    datasplitkey_rects3 = [[parse_combine(key), rects3]
            for key, rects3 in key_rects3]
    data_split_key_rects3 = [(data, split, key, rects3)
            for (data, split, key), rects3 in datasplitkey_rects3]

    data_to_split_key_rects3 = list_to_dict(data_split_key_rects3, 0)

    for data in data_to_split_key_rects3:
        logging.info(data)
        if tree_file:
            source_dataset = TSVDatasetSource(data, tax.root)
            source_dataset._ensure_initialized()
        else:
            source_dataset = TSVDataset(data)
        split_key_rects3 = data_to_split_key_rects3[data]
        split_to_key_rects3 = list_to_dict(split_key_rects3, 0)
        for split in split_to_key_rects3:
            logging.info(split)
            key_rects3 = split_to_key_rects3[split]
            key_to_rects3 = list_to_dict(key_rects3, 0)
            v = source_dataset.get_latest_version(split, 'label')
            logging.info('{} - {}'.format(data, split))
            source_key_rects = load_key_rects(source_dataset.iter_data(split, 'label', version=v))
            is_equal = True
            num_added, num_removed = 0, 0
            meta = {'in_tsv': in_tsv}
            for i, (key, origin_rects) in tqdm(enumerate(source_key_rects)):
                if debug:
                    old_origin_rects = copy.deepcopy(origin_rects)
                yes_rects, no_rects, un_rects = key_to_rects3.get(key, [[[], [], []]])[0]
                # if no_rects are in original, remove it. remove first
                for r in no_rects:
                    delete_rects = find_same_rects(r, origin_rects)
                    if tree_file:
                        if r['class'] in source_dataset._targetlabel_to_sourcelabels:
                            for reverse_class in source_dataset._targetlabel_to_sourcelabels[r['class']]:
                                r2 = copy.deepcopy(r)
                                r2['class'] = reverse_class
                                delete_rects2 = find_same_rects(r2, origin_rects)
                                delete_rects.extend(delete_rects2)
                    if 'class_from' in r:
                        r2 = copy.deepcopy(r)
                        r2['class'] = r2['class_from']
                        delete_rects2 = find_same_rects(r2, origin_rects)
                        delete_rects.extend(delete_rects2)
                    for d in delete_rects:
                        # delete rects may include duplicate terms
                        if d in origin_rects:
                            origin_rects.remove(d)
                            num_removed = num_removed + 1
                            is_equal = False
                # if yes_rects are not in original, add it
                for r in yes_rects:
                    same_rects = find_same_rects(r, origin_rects)
                    if len(same_rects) > 0:
                        for s in same_rects:
                            if s.get('uhrs_confirm', 0) == 0:
                                is_equal = False
                            s['uhrs_confirm'] = s.get('uhrs_confirm', 0) + 1
                    else:
                        r['uhrs_confirm'] = r.get('uhrs_confirm', 0) + 1
                        origin_rects.append(copy.deepcopy(r))
                        is_equal = False
                        num_added = num_added + 1
                for r in un_rects:
                    same_rects = find_same_rects(r, origin_rects)
                    for s in same_rects:
                        if s.get('uhrs_uncertain', 0) == 0:
                            is_equal = True
                        s['uhrs_uncertain'] = s.get('uhrs_uncertain', 0) + 1
                if debug:
                    if len(origin_rects) != len(old_origin_rects):
                        for _, _, im_str in source_dataset.iter_data(split, filter_idx=[i]):
                            im = img_from_base64(im_str)
                            old_im = im.copy()
                            draw_bb(old_im, [r['rect'] for r in old_origin_rects],
                                    [r['class'] for r in old_origin_rects])
                            new_im = im.copy()
                            draw_bb(new_im, [r['rect'] for r in origin_rects],
                                    [r['class'] for r in origin_rects])
                            yes_im = im.copy()
                            draw_bb(yes_im, [r['rect'] for r in yes_rects],
                                    [r['class'] for r in yes_rects])
                            no_im = im.copy()
                            draw_bb(no_im, [r['rect'] for r in no_rects],
                                    [r['class'] for r in no_rects])

                            logging.info(pformat(old_origin_rects))
                            logging.info(pformat(origin_rects))
                            logging.info(pformat(yes_rects))
                            logging.info(pformat(no_rects))
                            show_images([old_im, new_im, yes_im, no_im], 2, 2)

            assert not source_dataset.has(split, 'label', v + 1)
            if not is_equal:
                source_dataset.write_data(((key, json.dumps(rects)) for key, rects in source_key_rects),
                        split, 'label', version=v+1)
                meta_file = source_dataset.get_data(split, 'label.metadata', version=v+1) + '.yaml'
                meta['num_added_rects'] = num_added
                meta['num_removed_rects'] = num_removed
                meta['total_number_images'] = len(source_key_rects)

                meta['avg_added_rects'] = 1. * num_added / meta['total_number_images']
                meta['avg_removed_rects'] = 1. * num_removed / meta['total_number_images']
                write_to_yaml_file(meta, meta_file)
                logging.info(pformat(meta))
            else:
                logging.info('equal - {} - {}'.format(data, split))
        populate_dataset_details(data)


def process_tsv_main(**kwargs):
    if kwargs['type'] == 'gen_tsv':
        input_folder = kwargs['input']
        output_folder = kwargs['ouput']
        gen_tsv_from_labeling(input_folder, output_folder)
    elif kwargs['type'] == 'gen_term_list':
        tax_folder = kwargs['input']
        term_list = kwargs['output']
        gen_term_list(tax_folder, term_list)
    elif kwargs['type'] == 'gen_noffset':
        tax_input_folder = kwargs['input']
        tax_output_folder = kwargs['output']
        gen_noffset(tax_input_folder, tax_output_folder)
    elif kwargs['type'] == 'ambigous_noffset':
        tax_input_folder = kwargs['input']
        ambigous_file_out = kwargs['output']
        output_ambigous_noffsets_main(tax_input_folder, ambigous_file_out)
    elif kwargs['type'] == 'standarize_crawled':
        tsv_input = kwargs['input']
        tsv_output = kwargs['output']
        standarize_crawled(tsv_input, tsv_output)
    elif kwargs['type'] == 'taxonomy_to_tsv':
        taxonomy_folder = kwargs['input']
        build_taxonomy_impl(taxonomy_folder, **kwargs)
    elif kwargs['type'] == 'yolo_model_convert':
        old_proto = kwargs['prototxt']
        old_model = kwargs['model']
        new_model = kwargs['output']
        yolo_old_to_new(old_proto, old_model, new_model)
    elif kwargs['type'] == 'build_data_index':
        data = kwargs['input']
        populate_dataset_details(data)
    elif kwargs['type'] == 'build_all_data_index':
        populate_all_dataset_details()
    elif kwargs['type'] == 'extract_for_uhrs':
        taxonomy_folder = kwargs['input']
        data = kwargs['data']
        build_taxonomy_impl(taxonomy_folder,
                data=data,
                datas=get_data_sources(),
                max_image_per_label=kwargs.get('max_image_per_label', 10000000),
                min_image_per_label=kwargs.get('min_image_per_label', 0),
                num_test=0)
        convert_to_uhrs_with_url(data + '_with_bb')
    elif kwargs['type'] == 'merge_labels':
        in_tsv = kwargs['input']
        convert_uhrs_result_back_to_sources(in_tsv, debug=False)
    elif kwargs['type'] == 'ensure_inject_dataset':
        ensure_inject_dataset(kwargs['data'])
    elif kwargs['type'] == 'ensure_inject_expid':
        ensure_inject_expid(kwargs['full_expid'])
    else:
        logging.info('unknown task {}'.format(kwargs['type']))

def parse_args():
    parser = argparse.ArgumentParser(description='TSV Management')
    parser.add_argument('-c', '--config_file', help='config file',
            type=str)
    parser.add_argument('-t', '--type', help='what type it is: gen_tsv',
            type=str, required=False)
    parser.add_argument('-i', '--input', help='input',
            type=str, required=False)
    parser.add_argument('-p', '--prototxt', help='proto file',
            type=str, required=False)
    parser.add_argument('-m', '--model', help='model file',
            type=str, required=False)
    parser.add_argument('-o', '--output', help='output',
            type=str, required=False)
    parser.add_argument('-d', '--datas',
            default=argparse.SUPPRESS,
            nargs='*',
            help='which data are used for taxonomy_to_tsv',
            type=str,
            required=False)
    parser.add_argument('-da', '--data',
            default=argparse.SUPPRESS,
            help='the dataset name under data/',
            type=str,
            required=False)
    parser.add_argument('-fe', '--full_expid',
            default=argparse.SUPPRESS,
            type=str,
            required=False)
    parser.add_argument('-maxi', '--max_image_per_label',
            default=argparse.SUPPRESS,
            type=int,
            required=False)
    parser.add_argument('-mini', '--min_image_per_label',
            default=argparse.SUPPRESS,
            type=int,
            required=False)
    return parser.parse_args()

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    kwargs = vars(args)
    if kwargs.get('config_file'):
        logging.info('loading parameter from {}'.format(kwargs['config_file']))
        kwargs = load_from_yaml_file(kwargs['config_file'])
    process_tsv_main(**kwargs)

