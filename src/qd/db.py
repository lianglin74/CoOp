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

def create_mongodb_client():
    config = load_from_yaml_file('./aux_data/configs/mongodb_credential.yaml')
    host = config['host']
    return MongoClient(host)

def create_bbverification_db(db_name='qd', collection_name='uhrs_bounding_box_verification'):
    '''
    use create_bbverificationdb_client since the naming is not precise
    '''
    return BoundingBoxVerificationDB(db_name, collection_name)

def create_bbverificationdb_client(db_name='qd',
        collection_name='uhrs_bounding_box_verification'):
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

def ensure_to_objectid(r):
    if type(r) is str:
        return ObjectId(r)
    else:
        return r

def create_annotation_db():
    return AnnotationDB()

class AnnotationDB(object):
    '''
    gradually move all the Annotation db related function call to this wrapper.
    The related table incldues: image, ground_truth, label, prediction_result
    '''
    def __init__(self):
        self._qd = create_mongodb_client()
        self._gt = self._qd['qd']['ground_truth']
        self._label = self._qd['qd']['label']
        self._acc = self._qd['qd']['acc']
        self._phillyjob = self._qd['qd']['phillyjob']

    def insert_phillyjob(self, **kwargs):
        if 'create_time' not in kwargs:
            kwargs['create_time'] = datetime.now()
        self._phillyjob.insert_one(kwargs)

    def remove_phillyjob(self, **kwargs):
        self._phillyjob.delete_many(kwargs)

    def update_phillyjob(self, query, update):
        if 'create_time' not in update:
            update['create_time'] = datetime.now()
        return self._phillyjob.update(query, update)

    def iter_phillyjob(self, **kwargs):
        return self._phillyjob.find(**kwargs)

    # acc related
    def insert_acc(self, **kwargs):
        if 'create_time' not in kwargs:
            kwargs['create_time'] = datetime.now()
        self._acc.insert_one(kwargs)

    def iter_acc(self, **query):
        return self._acc.find(query)

    def iter_unique_test_info_in_acc(self):
        pipeline = [
                {'$group': {'_id': {'test_data': '$test_data',
                                    'test_split': '$test_split',
                                    'test_version': '$test_version'}}}
                ]
        for result in self._acc.aggregate(pipeline):
            yield result['_id']

    def exist_acc(self, **query):
        try:
            next(self.iter_acc(**query))
            return True
        except:
            return False

    # label related
    def update_label(self, query, update):
        self._label.update_one(query, update)

    def iter_label(self):
        return self._label.find()

    def iter_query_label(self, query):
        return self._label.find(query)

    def insert_label(self, **kwargs):
        if 'uuid' not in kwargs:
            kwargs['uuid'] = gen_uuid()
        if 'create_time' not in kwargs:
            kwargs['create_time'] = datetime.now()

        self._label.insert_one(kwargs)

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

    def query_verified_correct_rects(self, data, split, key):
        pipeline = [{'$match': {'data': data,
                                'split': split,
                                'key': key,
                                'interpretation_result': 1,
                                'status': {'$in': [self.status_completed,
                                                   self.status_merged]}}},
                    ]
        return [info['rect'] for info in self.query_by_pipeline(pipeline)]

    def query_verified_incorrect_rects(self, data, split, key):
        pipeline = [{'$match': {'data': data,
                                'split': split,
                                'key': key,
                                'interpretation_result': {'$ne': 1},
                                'status': {'$in': [self.status_completed,
                                                   self.status_merged]}}},
                    ]
        return [info['rect'] for info in self.query_by_pipeline(pipeline)]

    def query_nonverified_rects(self, data, split, key):
        pipeline = [{'$match': {'data': data,
                                'split': split,
                                'key': key,
                                'status': {'$nin': [self.status_completed,
                                                   self.status_merged]}}},
                    ]
        result = []
        for info in self.query_by_pipeline(pipeline):
            info['rect']['_id'] = str(info['_id'])
            result.append(info['rect'])
        return result

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

    def update_priority_tier(self, all_id, new_priority_tier):
        all_id = [ensure_to_objectid(i) for i in all_id]
        self.collection.update_many({'_id': {'$in': all_id}},
                                    {'$set': {'priority_tier': new_priority_tier}})

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

    def get_completed_uhrs_result(self, extra_match=None):
        merge_multiple_verification = False # True if we submit one rect multiple times, not tested
        match_criteria = {'status': self.status_completed}
        if extra_match:
            match_criteria.update(extra_match)

        pipeline = [
                {'$match': match_criteria},
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

