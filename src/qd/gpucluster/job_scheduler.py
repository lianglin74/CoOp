from qd.db import create_annotation_db
from tqdm import tqdm
from qd.qd_common import process_run
import logging
from qd.qd_common import print_trace
from qd.qd_common import dict_has_path, dict_get_path_value
from qd.qd_common import dict_get_all_path
from qd.qd_common import dict_remove_path
from bson import ObjectId
from qd.gpucluster.aml_client import create_aml_client
from pprint import pformat
import datetime
from qd.gpucluster.aml_client import AMLClient



def job_finished(_id):
    scheduler = JobScheduler()
    return scheduler.is_completed(_id)

def create_cluster_client(platform, **kwargs):
    if platform == 'philly':
        from qd.gpucluster.philly_client import create_philly_client
        return create_philly_client(**kwargs)
    else:
        assert platform == 'aml'
        return create_aml_client(**kwargs)

def remove_key_with_dot(info):
    all_path = dict_get_all_path(info)
    all_path = [p for p in all_path if '.' in p]
    for p in all_path:
        dict_remove_path(info, p)

class JobScheduler(object):
    # None, held -> RequestSubmit
    status_request_submit = 'RequestSubmit'
    status_submitting = 'Submitting'
    status_submitted = 'Submitted'

    # Submitted -> Completed/Failed
    status_completed = 'Completed'
    status_failed = 'Failed'

    # RequestSubmit, Submitted, Completed, Failed, Canceled -> RequestHold
    status_request_hold = 'RequestHold'
    status_holding = 'Holding'
    status_held = 'Held'

    # RequestSubmit, Submitted, Completed, Failed, Held -> RequestCancel
    status_request_cancel = 'RequestCancel'
    status_canceling = 'Canceling'
    status_canceled = 'Canceled'

    key_status = 'JobStatus'
    collection = 'scheduler'
    collection_cluster = 'CurrentClusterStatus'

    def __init__(self):
        self.c = create_annotation_db()
        # some important field
        # JobStatus: the standarized status

    def is_completed(self, _id):
        info = next(self.c.iter_general(self.collection,
                            _id=_id))
        return info[self.key_status] == self.status_completed

    # by client
    def submit_to_scheduler(self, env, execute_func, cmd=None):
        kwargs = {
                self.key_status: self.status_request_submit,
                'env': env,
            }
        if execute_func is None and cmd is not None:
            kwargs['cmd'] = cmd
        else:
            kwargs['execute_func'] = execute_func

        result = self.c.insert_one(
            self.collection,
            **kwargs,
        )
        scheduler_id = result.inserted_id
        return scheduler_id

    # by server
    def submit_to_cluster(self, scheduler_id, force=False):
        from_status = [
                self.status_request_submit,
                self.status_canceled,
                self.status_failed,
                self.status_completed,
        ]
        if force:
            from_status.append(self.status_submitting)
        self.mark_status(
            scheduler_id,
            from_status,
            self.status_submitting,
        )
        scheduler_info = next(self.c.iter_general(self.collection, _id=scheduler_id))
        env = scheduler_info['env']
        history = scheduler_info.get('history', [])
        if 'appID' in scheduler_info:
            history.append(scheduler_info['appID'])
        import copy
        submit_param = copy.deepcopy(env)
        submit_param.pop('run_type')
        if 'execute_func' in scheduler_info:
            from qd.pipeline import aml_func_run
            func = {
                'from': scheduler_info['execute_func']['from'],
                'import': scheduler_info['execute_func']['import'],
            }
            appID = aml_func_run(
                func,
                scheduler_info['execute_func']['param'],
                **submit_param,
            )
            aml_client = create_aml_client(**submit_param)
        else:
            aml_client = create_aml_client(**submit_param)
            appID = aml_client.submit(scheduler_info['cmd'])
        aml_client.inject(appID)

        self.c.update_one(
            self.collection,
            query={'_id': scheduler_id,
                   self.key_status: self.status_submitting},
            update={
                '$set': {
                    self.key_status: self.status_submitted,
                    'appID': appID,
                    'history': history,
                    'submit_time': datetime.datetime.now()
                }
            }
        )
        return appID

    def monitor(self):
        self.update_all()
        self.submit_for_request_submit_job()
        self.smart_redirect_cluster()
        #self.resubmit_low_priority()

    def resubmit_low_priority(self):
        for scheduler_job in self.c.iter_general(
            self.collection, **{
                self.key_status:
                self.status_submitted,
            }):
            try:
                self.resubmit_low_priority_one(scheduler_job)
            except:
                print_trace()

    def smart_redirect_cluster(self):
        for scheduler_job in self.c.iter_general(
            self.collection, **{
                self.key_status:
                self.status_submitted,
            }):
            try:
                self.smart_redirect_cluster_one(scheduler_job)
            except KeyboardInterrupt:
                break
            except:
                print_trace()

    def resubmit_low_priority_one(self, scheduler_job):
        return
        job_in_db = next(self.c.iter_phillyjob(appID=scheduler_job['appID']))
        # we only redirect the job if it is in queued status
        if job_in_db['status'] != 'Queued':
            return

        if (datetime.datetime.now() - job_in_db['update_time']).seconds > 10 * 60:
            # the job may have already started
            logging.info('job status is stale {}'.format(job_in_db['_id']))
            return

        # check whether the job has been in queue for more than a certain
        # time
        if 'submit_time' not in scheduler_job:
            return

        waiting = datetime.datetime.now() - scheduler_job['submit_time']
        if waiting.seconds < 60 * 10:
            logging.info('waiting time is not enough')
            return
        client = self.create_cluster_client_by_job(scheduler_job)

        if client.aks_compute_global_dispatch:
            return

        if client.unique_gjd_cluster:
            return

        logging.info('resubmit {}'.format(
            scheduler_job['_id']
        ))
        process_run(resubmit_to_cluster, scheduler_job['_id'], None)
        #self.abort(scheduler_job['_id'])
        #self.submit_to_cluster(scheduler_job['_id'])

    def smart_redirect_cluster_one(self, scheduler_job):
        # if there is no candidate clusters, we will also ignore to
        # re-direct this job
        candidate_clusters = scheduler_job['env'].get('candidate_clusters', [])
        candidate_clusters = [c for c in candidate_clusters if c !=
                              scheduler_job['env']['cluster']]
        if len(candidate_clusters) == 0:
            return

        job_in_db = next(self.c.iter_phillyjob(appID=scheduler_job['appID']))
        # we only redirect the job if it is in queued status
        if job_in_db['status'] != 'Queued':
            return

        if (datetime.datetime.now() - job_in_db['update_time']).seconds > 5 * 60:
            logging.info('job status is stale {}'.format(job_in_db['_id']))
            return

        # check whether the job has been in queue for more than a certain
        # time
        if 'submit_time' not in scheduler_job:
            return

        waiting = datetime.datetime.now() - scheduler_job['submit_time']
        if waiting.seconds < 60 * 10:
            logging.info('waiting time is not enough')
            return

        # check if there is free resource in current cluster.
        cluster_status = next(self.c.iter_general('current_cluster', cluster=job_in_db['cluster']))
        if (datetime.datetime.now() -
                cluster_status['last_update_time']).seconds > 5 * 60:
            logging.info('cluster status is old')
            # the cluster status is out of date, and we cannot trust it.
            return

        if cluster_status['total_free_gpu'] > job_in_db['num_gpu']:
            # current cluster has enough resource. let's give it more time
            return

        all_candidate_cluster_status = list(self.c.iter_general(
            'current_cluster', cluster={'$in': candidate_clusters}))
        c_to_idx = {c: i for i, c in enumerate(candidate_clusters)}
        all_candidate_cluster_status = sorted(all_candidate_cluster_status, key=lambda x: c_to_idx[x['cluster']])
        found = None
        for s in all_candidate_cluster_status:
            if (datetime.datetime.now() - s['last_update_time']
                    ).seconds > 5 * 60:
                continue
            if 'last_redirect_job' in s:
                last_redirect_job = s['last_redirect_job']
                last_redirect_job_info = next(
                    self.c.iter_general('scheduler', _id=last_redirect_job))
                last_redirect_job_aml_info = next(self.c.iter_general(
                    'phillyjob',
                    appID=last_redirect_job_info['appID']))
                if last_redirect_job_aml_info['status'] not in [
                        AMLClient.status_running,
                        AMLClient.status_completed,
                        ]:
                    continue
            if s['total_free_gpu'] >= job_in_db['num_gpu']:
                found = s
                break

        if found is None:
            return

        logging.info('redirect {} to {}'.format(
            scheduler_job['_id'], found['cluster']
        ))
        self.abort(scheduler_job['_id'])
        self.update_cluster(scheduler_job['_id'], found['cluster'])
        self.submit_to_cluster(scheduler_job['_id'])
        self.c.update_one(
            'current_cluster',
            query={'cluster': found['cluster']},
            update={'$set': {'last_redirect_job': scheduler_job['_id']}})

    def submit_for_request_submit_job(self):
        for scheduler_job in self.c.iter_general(
            self.collection, **{
                self.key_status: self.status_request_submit,
            }):
            try:
                self.submit_to_cluster(scheduler_job['_id'])
            except:
                print_trace()

    def update_cluster(self, _id, new_cluster):
        self.c.update_one(
            self.collection,
            query={'_id': _id},
            update={'$set': {'env.cluster': new_cluster}}
        )


    #def standarize_status(self):
        ## populate JobStatus if it is None. In different clusters, the status name
        ## might be different. Thus, here, we unify them
        #num = 0
        ##for job in tqdm(self.c.iter_general(self.collection, JobStatus=None)):
        #for job in tqdm(self.c.iter_general(self.collection)):
            #if 'status' not in job:
                #continue
            #def get_standarized_status(status):
                #if status in ['Pass', 'Completed']:
                    #return self.status_completed
                #elif status in ['Killed', 'Canceled']:
                    #return self.status_canceled
                #elif status in ['Running', 'Queued']:
                    #return self.status_submitted
                #elif status in ['Failed']:
                    #return self.status_failed
                #elif status in ['Finalizing']:
                    #return self.status_submitted
                #elif status in ['CancelRequested']:
                    #return self.status_submitted
                #logging.info('unknown status: {}'.format(status))
                #return status
            #num += 1
            #self.c.update_one(self.collection, query={'_id': job['_id']},
                    #update={'$set': {'JobStatus':
                        #get_standarized_status(job['status'])}})
        #logging.info('standarized job = {}'.format(num))

    def check_queued_job(self):
        num = 0
        result = self.c.update_one(self.collection,
                                   query={self.key_status: self.status_queued},
                                   update={'$set': {self.key_status: self.status_submitting}})
        for job in tqdm(self.c.iter_general(self.collection,
            JobStatus='Queued')):
            if 'cluster' not in job:
                continue
            if job['cluster'] != 'auto':
                cluster = self.create_cluster_client_by_job(cluster=job['cluster'],
                        num_gpu=job['num_gpu'])
                job_id = cluster.submit_job()
        logging.info('submitted job = {}'.format(num))

    def inject_from_cluster(self):
        all_cluster = load_from_yaml_file('./aux_data/job_scheduler/clusters.yaml')
        all_client = [create_cluster_client(**cluster) for cluster in all_cluster]
        job_infos = []
        for c in all_client:
            job_infos.extend(c.query(by_status=AMLClient.status_running))
        for job_info in job_infos:
            # we need to replace '.' in job_info because of the requirement
            # from mongodb
            from qd.qd_common import dict_get_all_path
            all_path = dict_get_all_path(job_info)
            for p in all_path:
                if '.' in p:
                    v = dict_get_path_value(job_info, p)
                    dict_remove_path(job_info, p)
                    p = p.replace('.', '_')
                    dict_update_path_value(job_info, p, v)
            self.c.update_one(self.collection,
                    query={'appID': job_info['appID']},
                    update={'$setOnInsert': job_info},
                    upsert=True)

    def inject_cluster_status(self):
        clusters = load_from_yaml_file('./aux_data/job_scheduler/clusters.yaml')
        for c in clusters:
            client = create_cluster_client(**c)
            cluster_status = client.get_cluster_status()
            logging.info('updating {}'.format(pformat(c)))
            self.c.update_one(self.collection_cluster, query={'cluster':
                c['cluster']}, update={'$set': cluster_status}, upsert=True)

    def create_cluster_client_by_job(self, job_info):
        return create_cluster_client(
            platform='aml',
            cluster=job_info['env']['cluster'],
            num_gpu=job_info['env']['num_gpu'])

    def mark_status(self, _id, status_from, status_to):
        if isinstance(status_from, str):
            status_from = [status_from]
        result = self.c.update_one(
            self.collection,
            query={
                '_id': _id,
                self.key_status: {'$in': status_from},
            },
            update={
                '$set': {
                    self.key_status: status_to,
                }
            },
        )
        assert result.matched_count == 1

    def initiate_from(self, _id, num_gpu=None):
        info = next(self.c.iter_general(
            self.collection,
            _id=_id
        ))
        remove_keys = ['_id', 'appID', 'history']
        for k in remove_keys:
            if k in info:
                del info[k]
        info[self.key_status] = self.status_request_submit
        info['initiate_from'] = _id
        if num_gpu:
            info['env']['num_gpu'] = num_gpu
        insert_result = self.c.insert_one(
            self.collection,
            **info
        )
        return insert_result.inserted_id

    def abort(self, job_info):
        if isinstance(job_info, ObjectId):
            job_info = next(self.c.iter_general(self.collection, _id=job_info))
        logging.info('current status = {}'.format(job_info[self.key_status]))
        if job_info[self.key_status] in [
                self.status_completed,
                self.status_failed,
                self.status_canceled,
                self.status_submitting,
        ]:
            logging.info('job {}({}) cannot be aborted'.format(
                job_info['_id'],
                job_info[self.key_status],
            ))
            return
        self.mark_status(job_info['_id'], [
            self.status_submitted,
            self.status_canceling,
        ], self.status_canceling)
        client = self.create_cluster_client_by_job(job_info)
        client.abort(job_info['appID'])
        self.mark_status(job_info['_id'],
                         self.status_canceling,
                         self.status_canceled)

    def update_one(self, job_info):
        job_in_db = next(self.c.iter_phillyjob(appID=job_info['appID']))
        if job_in_db['status'] == 'Completed':
            if job_info[self.key_status] == self.status_submitted:
                self.mark_status(
                    job_info['_id'],
                    self.status_submitted,
                    self.status_completed,
                )
        elif job_in_db['status'] == 'Failed':
            if job_info[self.key_status] == self.status_submitted:
                from qd.gpucluster.aml_client import retriable_error_codes
                error_codes = job_in_db.get('result', '').split(',')
                if any(c in retriable_error_codes() for c in error_codes):
                    self.mark_status(
                        job_info['_id'],
                        self.status_submitted,
                        self.status_request_submit,
                    )
                else:
                    self.mark_status(
                        job_info['_id'],
                        self.status_submitted,
                        self.status_failed,
                    )
        #client = self.create_cluster_client_by_job(job_info)
        #result = client.query(job_info['appID'])
        #assert len(result) == 1
        #result = result[0]
        #self.c.update_one(self.collection,
                #query={'_id': job_info['_id']},
                #update={'$set': result})

    def iter_db_query(self, **kwargs):
        return self.c.iter_general(self.collection, **kwargs)

    def update_all(self):
        need_update_status = [self.status_submitted]
        for job_info in self.c.iter_general(self.collection,
                **{self.key_status: {'$in': need_update_status}}):
            #if str(job_info['_id']) != '603b575f94bc075f6e17146b':
                #continue
            self.update_one(job_info)

    def request_hold_jobs(self, query):
        query = copy.deepcopy(query)
        for job_info in self.c.iter_general(self.collection, **query):
            if job_info[self.key_status] == self.status_submitted:
                # we need to make sure the status is not changed to avoid race
                # condition if we have anohter process to run this
                result = self.c.update_one(self.collection,
                        query={'_id': job_info['_id'],
                               self.key_status: self.status_submitted},
                        update={'$set': {self.key_status:
                                        self.status_request_hold}}
                        )
                assert result.modified_count == 1, job_info['_id']

    def check_requesthold_job(self):
        for job_info in self.c.iter_general(self.collection,
                **{self.key_status: self.status_request_hold}):
            update_result = self.c.update_one(self.collection,
                query={'_id': job_info['_id'],
                       self.key_status: self.status_request_hold},
                update={'$set': {self.key_status:
                        self.status_holding}}
                    )
            if update_result.modified_count != 1:
                logging.info('job status changed = {}'.format(job_info['_id']))
                continue
            if 'appID' in job_info:
                self.abort(job_info)
                self.update(job_info)
            else:
                raise NotImplementedError('unknown state')
            # update teh status as onhold
            update_result = self.c.update_one(self.collection,
                query={'_id': job_info['_id'],
                       self.key_status: self.status_holding},
                update={'$set': {self.key_status: self.status_held}}
                    )

    def request_resubmit_held_jobs(self):
        for job_info in self.c.iter_general(self.collection, **{self.key_status:
            self.status_held}):
            update_result = self.c.update_one(self.collection,
                    query={'_id': job_info['_id'],
                        self.key_status: self.status_held},
                    update={'$set': {self.key_status:
                        self.status_request_submit}})
            if update_result.modified_count != 1:
                logging.info('job status changed = {}'.format(job_info['_id']))
                continue

    def submit(self, job_info):
        cluster = self.create_cluster_client_by_job(job_info)
        return cluster.submit(job_info['cmd'])

    def check_request_submit_job(self):
        for job_info in self.c.iter_general(self.collection,
                **{self.key_status: self.status_request_submit}):
            update_result = self.c.update_one(self.collection,
                    query={'_id': job_info['_id'],
                        self.key_status: self.status_request_submit},
                    update={'$set': {self.key_status:
                        self.status_submitting}})
            if update_result.modified_count != 1:
                logging.info('job status changed = {}'.format(job_info['_id']))
                continue
            logging.info(job_info['_id'])
            cluster = 'sc2'
            cluster = create_cluster_client(platform='philly', cluster=cluster, num_gpu=job_info['num_gpu'])
            appID = cluster.submit(job_info['cmd'])
            self.c.update_one(self.collection,
                    query={
                        '_id': job_info['_id'],
                        self.key_status: self.status_submitting},
                    update={
                        '$set': {
                            self.key_status: self.status_submitted,
                            'appID': appID,
                            'cluster': cluster,
                            }})

def resubmit_to_cluster(_id, to_cluster):
    j = JobScheduler()
    if isinstance(_id, str):
        _id = ObjectId(_id)
    j.abort(_id)
    if to_cluster is not None:
        j.update_cluster(_id, to_cluster)
    j.submit_to_cluster(_id, force=True)

def concurrent_submit(_id, to_cluster):
    j = JobScheduler()
    if isinstance(_id, str):
        _id = ObjectId(_id)
    j.concurrent_submit(_id, to_cluster)

def dup_to_cluster(_id, to_cluster, num_gpu=None):
    j = JobScheduler()
    if isinstance(_id, str):
        _id = ObjectId(_id)
    _id = j.initiate_from(_id, num_gpu=num_gpu)
    if to_cluster is not None:
        j.update_cluster(_id, to_cluster)
    j.submit_to_cluster(_id, force=True)
    return _id

def execute(task_type, **kwargs):
    j = JobScheduler()
    if task_type == 'resubmit':
        to_cluster = kwargs.get('to_cluster')
        main_process = kwargs.get('main_process')
        for _id in kwargs.get('remainders'):
            if not main_process:
                process_run(resubmit_to_cluster, _id, to_cluster)
            else:
                resubmit_to_cluster(_id, to_cluster)
    elif task_type == 'cc':
        to_cluster = kwargs.get('to_cluster')
        main_process = kwargs.get('main_process', True)
        dup_ids = []
        for _id in kwargs.get('remainders'):
            if not main_process:
                did = process_run(concurrent_submit, _id, to_cluster)
            else:
                did = concurrent_submit(_id, to_cluster)
            dup_ids.append(did)
        logging.info('new id:\n{}'.format([str(_id) for _id in dup_ids]))

    elif task_type == 'dup':
        to_cluster = kwargs.get('to_cluster')
        main_process = kwargs.get('main_process', True)
        dup_ids = []
        num_gpu = kwargs.get('num_gpu')
        for _id in kwargs.get('remainders'):
            if not main_process:
                did = process_run(dup_to_cluster, _id, to_cluster, num_gpu)
            else:
                did = dup_to_cluster(_id, to_cluster, num_gpu)
            dup_ids.append(did)
        logging.info('new id:\n{}'.format(' '.join(map(str, dup_ids))))
    elif task_type in ['q', 'query']:
        cluster_to_client = {}
        all_info = []
        _ids = kwargs.get('remainders', [])
        _ids = [ObjectId(i) for i in _ids]
        for job in j.iter_db_query(_id={'$in': _ids}):
            if job['env'].get('cluster') in cluster_to_client:
                client = cluster_to_client[job['env']['cluster']]
            else:
                client = create_aml_client(**job['env'])
                cluster_to_client[job['env']['cluster']] = client
            logging.info(pformat(job))
            all_info.extend(client.iter_query(
                run_id=job['appID'],
                with_log=kwargs.get('with_log', True),
                log_full=kwargs.get('log_full', True),
                with_details=kwargs.get('with_details', True),
                detect_error_if_failed=True,
            ))
        from qd.qd_common import print_job_infos
        print_job_infos(all_info)
    elif task_type in ['abort']:
        for _id in kwargs.get('remainders'):
            _id = ObjectId(_id)
            j.abort(_id)
    elif task_type in ['monitor']:
        j.monitor()
    elif task_type in ['submit_for_request_submit_job']:
        j.submit_for_request_submit_job()
    elif task_type in ['i', 'inject']:
        cluster_to_client = {}
        _ids = kwargs.get('remainders', [])
        _ids = [ObjectId(i) for i in _ids]
        for job in j.iter_db_query(_id={'$in': _ids}):
            if job['env'].get('cluster') in cluster_to_client:
                client = cluster_to_client[job['env']['cluster']]
            else:
                client = create_aml_client(**job['env'])
                cluster_to_client[job['env']['cluster']] = client
            logging.info(pformat(job))
            client.inject(
                run_id=job['appID'],
            )
    else:
        raise NotImplementedError(task_type)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Philly Interface')
    parser.add_argument('task_type',
            choices=[
                'resubmit',
                'dup', # copy the job and resubmit the job
                'q',
                'ssh', 'query', 'f', 'failed', 'a', 'abort', 'submit',
                'qf', # query failed jobs
                'qq', # query queued jobs
                'qr', # query running jobs
                'd', 'download',
                'u', 'upload',
                'monitor',
                'parse',
                'init',
                'initc', # init with compile
                'initi', # incremental init
                'initic', # incremental & compile
                'blame',
                'submit_for_request_submit_job',
                's', 'summary', 'i', 'inject'])
    parser.add_argument('-no-wl', dest='with_log', action='store_false')
    #parser.add_argument('-no-dt', dest='with_details', action='store_false')
    #parser.add_argument('-no-lf', dest='log_full', action='store_false')
    #parser.add_argument('-hold', dest='sleep_if_fail', default=False, action='store_true')
    #parser.add_argument('-c', '--cluster', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-t', '--to_cluster', default=argparse.SUPPRESS, type=str)
    #parser.add_argument('-f', '--from_cluster', default=argparse.SUPPRESS, type=str)
    #parser.add_argument('-p', '--param', help='parameter string, yaml format',
            #type=str)
    parser.add_argument('-n', '--num_gpu', default=argparse.SUPPRESS, type=int)
    #parser.add_argument('--max', default=None, type=int)
    ##parser.add_argument('-wg', '--with_gpu', default=True, action='store_true')
    #parser.add_argument('-no-wg', '--with_gpu', default=True, action='store_false')
    ##parser.add_argument('-m', '--with_meta', default=True, action='store_true')
    #parser.add_argument('-no-m', '--with_meta', default=True, action='store_false')

    #parser.add_argument('-no-s', '--real_submit', default=True, action='store_false')
    #parser.add_argument('-ic', '--inject_collection', default='phillyjob', type=str)

    parser.add_argument('remainders', nargs=argparse.REMAINDER,
            type=str)
    return parser.parse_args()

if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)

