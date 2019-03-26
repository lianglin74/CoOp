from .qd_common import load_from_yaml_file
from pymongo import MongoClient
import pymongo
import copy
from bson import ObjectId
from datetime import datetime
from collections import OrderedDict

def create_mongodb_client():
    config = load_from_yaml_file('./aux_data/configs/mongodb_credential.yaml')
    host = config['host']
    return MongoClient(host)

def create_bbverification_db():
    return BoundingBoxVerificationDB()

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
