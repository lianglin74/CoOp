from qd.qd_common import load_from_yaml_file
from qd.qd_common import gen_uuid
from pymongo import MongoClient
import pymongo
import copy
from bson import ObjectId
from datetime import datetime
from collections import OrderedDict
from collections import defaultdict
import logging
from tqdm import tqdm
from future.utils import viewitems
from qd.tsv_io import is_verified_rect

def is_positive_uhrs_verified(r):
    uhrs = r['uhrs']
    y, n = uhrs.get('1', 0), uhrs.get('2', 0)
    return y > n

def is_negative_uhrs_verified(r):
    uhrs = r['uhrs']
    y, n = uhrs.get('1', 0), uhrs.get('2', 0)
    return n > y

def create_mongodb_client():
    config = load_from_yaml_file('./aux_data/configs/mongodb_credential.yaml')
    host = config['host']
    return MongoClient(host)

def create_bbverification_db(db_name='qd', collection_name='uhrs_bounding_box_verification'):
    return BoundingBoxVerificationDB(db_name, collection_name)

def objectid_to_str(result):
    # convert the type of ObjectId() to string
    result = list(result)
    for r in result:
        r['_id'] = str(r['_id'])
    return result

def ensure_objectid(result):
    for r in result:
        if type(r['_id']) is str:
            r['_id'] = ObjectId(r['_id'])

class AnnotationDB(object):
    '''
    gradually move all the Annotation db related function call to this wrapper.
    The related table incldues: image, ground_truth, label, prediction_result
    '''
    def __init__(self):
        self._qd = create_mongodb_client()
        self._gt = self._qd['qd']['ground_truth']
        self._label = self._qd['qd']['label']

    def update_label(self, query, update):
        self._label.update_one(query, update)

    def iter_label(self):
        return self._label.find()

    def insert_label(self, **kwargs):
        if 'uuid' not in kwargs:
            kwargs['uuid'] = gen_uuid()
        if 'create_time' not in kwargs:
            kwargs['create_time'] = datetime.now()

        self._label.insert_one(kwargs)

    def iter_query_label(self, query):
        return self._label.find(query)

    def build_label_index(self):
        self._label.create_index([('uuid', 1)], unique=True)
        self._label.create_index([('unique_name', 1)], unique=True,
                collation={'locale': 'en', 'strength':2})

    def drop_ground_truth_index(self):
        self._gt.drop_indexes()

    def build_ground_truth_index(self):
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('key', 1),
            ('class', 1)])
        self._gt.create_indexes
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('class', 1)])
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('class', 1),
            ('version', 1)])
        # used for deleting all before inserting
        self._gt.create_index([('data', 1),
            ('split', 1),
            ('version', 1)])

class BoundingBoxVerificationDB(object):
    status_requested = 'requested'
    status_retrieved = 'retrieved'
    status_submitted = 'submitted'
    status_completed = 'completed'
    status_merged = 'merged'
    urgent_priority_tier = -10000
    def __init__(self, db_name='qd', collection_name='uhrs_bounding_box_verification'):
        self.client = None
        self.db_name = db_name
        self.collection_name = collection_name

    def query_by_pipeline(self, pipeline):
        result = self.collection.aggregate(pipeline, allowDiskUse=True)
        return list(result)

    def request_by_insert(self, all_box_task):
        def get_bb_task_id(rect_info):
            from .qd_common import hash_sha1
            rect = rect_info['rect']
            return hash_sha1([rect_info['url'], rect['class'], rect['rect']])
        all_box_task = copy.deepcopy(all_box_task)
        for b in all_box_task:
            assert 'status' not in b
            assert 'priority_tier' in b, 'priority' in b
            assert 'url' in b
            assert 'rect' in b
            b['status'] = self.status_requested
            b['last_update_time'] = {'last_{}'.format(self.status_requested):
                    datetime.now()}
            if 'rect' not in b:
                b['rect'] = b['rects'][0]
            b['bb_task_id'] = get_bb_task_id(b)
        self.collection.insert_many(all_box_task)

    def retrieve(self, max_box, urgent_task=False):
        assert max_box > 0
        sort_config = OrderedDict()
        sort_config['priority_tier'] = pymongo.ASCENDING
        sort_config['priority'] = pymongo.ASCENDING
        match_criteria = {'status': self.status_requested}
        if urgent_task:
            match_criteria['priority_tier'] = self.urgent_priority_tier
        pipeline = [
                {'$match': match_criteria},
                {'$sort': sort_config},
                {'$limit': max_box},
                ]
        result = self.query_by_pipeline(pipeline)
        # we need to update the status to status_retrieved to avoid duplicate
        # retrieve && submit
        self.update_status([r['_id'] for r in result],
                self.status_retrieved)
        return objectid_to_str(result)

    def update_status(self, all_id, new_status, allowed_original_statuses=None):
        all_id = list(set(all_id))
        for i in range(len(all_id)):
            if type(all_id[i]) is str:
                all_id[i] = ObjectId(all_id[i])
        query = {'_id': {'$in': all_id}}
        if allowed_original_statuses:
            query['status'] = {'$in': allowed_original_statuses}
        time_key = 'last_update_time.last_{}'.format(new_status)
        result = self.collection.update_many(filter=query,
                update={'$set': {'status': new_status,
                                 time_key: datetime.now()}})
        assert result.modified_count == len(all_id)

    def reset_status_to_requested(self, all_bb_task):
        self.update_status([b['_id'] for b in all_bb_task],
                self.status_requested)

    def submitted(self, submitted):
        self.adjust_status(submitted, 'uhrs_submitted_result',
                'uhrs_submitted_result', self.status_submitted)

    def adjust_status(self, uhrs_results, uhrs_result_field, db_field,
            new_status, allowed_original_statuses=None):
        uhrs_results = list(uhrs_results)
        for s in uhrs_results:
            assert uhrs_result_field in s
            assert '_id' in s

        ensure_objectid(uhrs_results)
        all_id = [s['_id'] for s in uhrs_results]

        # save the result from uhrs
        for s in uhrs_results:
            self.collection.update_one(filter={'_id': s['_id']},
                    update={'$set': {db_field: s[uhrs_result_field]}})

        # update the status
        self.update_status(all_id, new_status, allowed_original_statuses)

    def query_submitted(self, topk=None):
        pipeline = [
                {'$match': {'status': self.status_submitted}},
                ]
        if topk:
            pipeline.append({'$limit': topk})
        result = self.query_by_pipeline(pipeline)
        return objectid_to_str(result)

    def complete(self, completed):
        self.adjust_status(completed, 'uhrs_completed_result',
                'uhrs_completed_result',
                self.status_completed,
                allowed_original_statuses=[self.status_submitted])

    def set_status_as_merged(self, all_id):
        self.update_status(all_id, self.status_merged,
                allowed_original_statuses=[self.status_completed])

    def get_completed_uhrs_result(self):
        merge_multiple_verification = False # True if we submit one rect multiple times, not tested
        pipeline = [
                {'$match': {'status': self.status_completed}},
                ]

        if merge_multiple_verification:
            pipeline.append(
                {'$group': {'_id': {'data': '$data',
                                    'split': '$split',
                                    'key': '$key',
                                    'bb_task_id': '$bb_task_id'},
                            'rects': {'$first': '$rects'},
                            'uhrs': {'$push': '$uhrs_completed_result'},
                            'related_ids': {'$push': '$_id'},
                            }}
                    )
        data_split_to_key_rects = defaultdict(list)
        all_id = []
        logging.info('querying the completed tasks')
        for rect_info in tqdm(self.query_by_pipeline(pipeline)):
            data = rect_info['data']
            split = rect_info['split']
            rect = rect_info['rect']
            all_id.append(rect_info['_id'])
            rect['uhrs'] = rect_info['uhrs_completed_result']
            key = rect_info['key']
            data_split_to_key_rects[(data, split)].append((key, rect))
        return data_split_to_key_rects, all_id

    @property
    def collection(self):
        if self.client is None:
            self.client = create_mongodb_client()
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('split', pymongo.ASCENDING),
                ('key', pymongo.ASCENDING)])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rects.0.class', pymongo.ASCENDING),
                ('rects.0.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rect.class', pymongo.ASCENDING),
                ('rect.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rects.0.class', pymongo.ASCENDING),
                ('rects.0.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ('priority', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING),
                ('rect.class', pymongo.ASCENDING),
                ('rect.from', pymongo.ASCENDING),
                ('status', pymongo.ASCENDING),
                ('priority', pymongo.ASCENDING),
                ])
            self.collection.create_index([('data', pymongo.ASCENDING)])
            self.collection.create_index([('priority_tier', pymongo.ASCENDING)])
            self.collection.create_index([('status', pymongo.ASCENDING)])
            self.collection.create_index([('priority', pymongo.ASCENDING)])
            self.collection.create_index([('rects.0.from', pymongo.ASCENDING)])
            self.collection.create_index([('rect.from', pymongo.ASCENDING)])
        return self.client[self.db_name][self.collection_name]

def uhrs_verify_db_merge_to_tsv(collection_name='uhrs_logo_verification'):
    set_interpretation_result_for_uhrs_result(collection_name)
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
    from qd.process_tsv import find_best_matched_rect
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
    from qd.tsv_io import TSVDataset
    from qd.qd_common import list_to_dict
    import simplejson as json
    from qd.qd_common import json_dump
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


def set_interpretation_result_for_uhrs_result(collection_name='uhrs_logo_verification'):
    c = create_bbverification_db(collection_name)
    query = {'status': c.status_completed,
            'interpretation_result': None}
    positive_ids, negative_ids, uncertain_ids = [], [], []
    for rect_info in tqdm(c.collection.find(query)):
        rect = rect_info['rect']
        rect.update({'uhrs': rect_info['uhrs_completed_result']})
        if is_positive_uhrs_verified(rect):
            positive_ids.append(rect_info['_id'])
        elif is_negative_uhrs_verified(rect):
            negative_ids.append(rect_info['_id'])
        else:
            uncertain_ids.append(rect_info['_id'])

    logging.info('num pos ids = {}'.format(len(positive_ids)))
    logging.info('num neg ids = {}'.format(len(negative_ids)))
    logging.info('num uncertain ids = {}'.format(len(uncertain_ids)))

    query = {'_id': {'$in': positive_ids}}
    c.collection.update_many(filter=query,
            update={'$set': {'interpretation_result': 1}})

    query = {'_id': {'$in': negative_ids}}
    c.collection.update_many(filter=query,
            update={'$set': {'interpretation_result': -1}})

    query = {'_id': {'$in': uncertain_ids}}
    c.collection.update_many(filter=query,
            update={'$set': {'interpretation_result': 0}})

