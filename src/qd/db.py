from .qd_common import load_from_yaml_file
from pymongo import MongoClient
import pymongo
import copy
from bson import ObjectId

def create_mongodb_client():
    config = load_from_yaml_file('./aux_data/configs/mongodb_credential.yaml')
    host = config['host']
    return MongoClient(host)

def create_bbverification_db():
    return BoundingBoxVerificationDB()

def objectid_to_str(result):
    # convert the type of ObjectId() to string
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
    def __init__(self):
        self.client = None
        self.db_name = 'qd'
        self.collection_name = 'uhrs_bounding_box_verification'
        self.collection = None

    def query_by_pipeline(self, pipeline):
        self._ensure_client_opened()
        result = self.collection.aggregate(pipeline, allowDiskUse=True)
        return list(result)

    def request_by_insert(self, all_box_task):
        self._ensure_client_opened()
        all_box_task = copy.deepcopy(all_box_task)
        for b in all_box_task:
            assert 'status' not in b
            b['status'] = self.status_requested
        self.collection.insert_many(all_box_task)

    def retrieve(self, max_box):
        assert max_box > 0
        self._ensure_client_opened()
        pipeline = [
                {'$match': {'status': self.status_requested}},
                {'$sort': {'priority': pymongo.ASCENDING}},
                {'$limit': max_box},
                ]
        result = self.query_by_pipeline(pipeline)
        # we need to update the status to status_retrieved to avoid duplicate
        # retrieve && submit
        self.update_status([r['_id'] for r in result],
                self.status_retrieved)
        return objectid_to_str(result)

    def update_status(self, all_id, new_status):
        from datetime import datetime
        for i in range(len(all_id)):
            if type(all_id[i]) is str:
                all_id[i] = ObjectId(all_id[i])
        self.collection.update_many(filter={'_id': {'$in': all_id}},
                update={'$set': {'status': new_status,
                                 'last_update_time': datetime.now()}})

    def reset_status_to_requested(self, all_bb_task):
        self.update_status([b['_id'] for b in all_bb_task],
                self.status_requested)

    def submitted(self, submitted):
        self.adjust_status(submitted, 'uhrs_submitted_result',
                'uhrs_submitted_result', self.status_submitted)

    def adjust_status(self, uhrs_results, uhrs_result_field, db_field,
            new_status):
        self._ensure_client_opened()
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
        self.update_status(all_id, new_status)

    def query_submitted(self, topk=None):
        self._ensure_client_opened()
        pipeline = [
                {'$match': {'status': self.status_submitted}},
                ]
        if topk:
            pipeline.append({'$limit': topk})
        result = self.query_by_pipeline(pipeline)
        return objectid_to_str(result)

    def complete(self, completed):
        self.adjust_status(completed, 'uhrs_completed_result',
                'uhrs_completed_result', self.status_completed)

    def _ensure_client_opened(self):
        if self.client is None or self.collection is None:
            self.client = create_mongodb_client()
            self.collection = self.client[self.db_name][self.collection_name]
            self.collection.create_index([('data', pymongo.ASCENDING)])
            self.collection.create_index([('status', pymongo.ASCENDING)])
            self.collection.create_index([('priority', pymongo.ASCENDING)])

