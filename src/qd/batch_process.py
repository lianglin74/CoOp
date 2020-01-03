import os
import time
import os.path as op
import logging
from pprint import pformat
import multiprocessing as mp

from qd.qd_common import is_cluster
from qd.qd_common import write_to_yaml_file, load_from_yaml_file
from qd.gpu_util import gpu_available
from qd.remote_run import sync_qd, remote_python_run


def mpi_task_processor(resource, task, func):
    ssh_info, gpus = resource
    logging.info(ssh_info)
    logging.info(gpus)
    if ssh_info == {}:
        #yolotrain_main(**task)
        func(**task)
    else:
        #yolotrain(**task)
        cmd_prefix = 'HOROVOD_CACHE_CAPACITY=0 '
        if len(gpus) == 1:
            cmd_prefix += 'CUDA_VISIBLE_DEVICES={}'.format(
                    ','.join(map(str, gpus)))
        else:
            cmd_prefix += 'CUDA_VISIBLE_DEVICES={} mpirun -npernode {}'.format(
                    ','.join(map(str, gpus)), len(gpus))
        remote_python_run(func, task, ssh_info, cmd_prefix)

def task_processor(resource, task, func):
    ssh_info, gpus = resource
    logging.info(ssh_info)
    logging.info(gpus)
    task['gpus'] = gpus
    if ssh_info == {}:
        #yolotrain_main(**task)
        func(**task)
    else:
        #yolotrain(**task)
        cmd_prefix = ''
        remote_python_run(func, task, ssh_info, cmd_prefix)

def get_resources():
    # use a file to load the machines. The benefit is that sometimes the run.py
    # will be copied as anohter run_123.py, which can benefit from the new list
    machines = load_from_yaml_file('.machines.yaml')
    return machines

def remote_run_func(func, is_mpi=True, availability_check=True, **param):
    all_task = [param]
    all_resource = get_resources()
    if is_mpi:
        l_task_processor = lambda resource, task: mpi_task_processor(resource, task,
                func)
    else:
        l_task_processor = lambda resource, task: task_processor(resource, task,
                func)
    b = BatchProcess(all_resource, all_task, l_task_processor)
    b._availability_check = availability_check
    b.run()

class BatchProcess(object):
    def __init__(self, all_resource, all_task, processor):
        #self._all_task = Queue()
        self._all_task = mp.Manager().Queue()
        for task in all_task:
            self._all_task.put(task)

        self._all_resouce = mp.Manager().Queue()
        for resource in all_resource:
            self._all_resouce.put(resource)

        self._task_description = '{}\n\n{}'.format(pformat(all_task),
                pformat(all_resource))

        self._availability_check = True
        self._processor = processor
        self._in_use_resource_file = op.expanduser('~/.in_use_resource.yaml')

    def _run_in_lock(self, func):
        #import portalocker
        #logging.info('begin to lock')
        #with portalocker.Lock('/tmp/process_tsv.lock') as fp:
            #portalocker.lock(fp, portalocker.LOCK_EX)
            #logging.info('end to lock')
        result = func()
        return result

    def _try_use_resource(self, resource):
        def use_resource(resource):
            in_use = self._is_in_use(resource)
            if not in_use:
                in_use_resource = self._load_save_valid_in_use_status()
                in_use_resource.append({'pid': os.getpid(),
                    'ip': resource[0].get('ip', 'localhost'),
                    'port': resource[0].get('-p', 22),
                    'gpus': resource[1]})
                logging.info('writting to {}'.format(self._in_use_resource_file))
                write_to_yaml_file(in_use_resource, self._in_use_resource_file)
            else:
                logging.info('resource is in use: {}'.format(pformat(resource)))
            return not in_use

        return self._run_in_lock(lambda: use_resource(resource))

    def _release_resouce(self, resource):
        def release_resource(resource):
            in_use_resource = self._load_save_valid_in_use_status()
            my_pid = os.getpid()
            to_be_removed = []
            for record in in_use_resource:
                if record['pid'] == my_pid and \
                        record.get('ip', '0') == resource[0].get('ip', '0') and \
                        record.get('port', '0') == resource[0].get('-p', 22) and \
                        all(g1 == g2 for g1, g2 in zip(record['gpus'],
                            resource[1])):
                    to_be_removed.append(record)

            logging.info('to be removed: {}'.format(pformat(to_be_removed)))

            for r in to_be_removed:
                in_use_resource.remove(r)
            write_to_yaml_file(in_use_resource, self._in_use_resource_file)

        self._run_in_lock(lambda: release_resource(resource))

    def _load_save_valid_in_use_status(self):
        if not op.isfile(self._in_use_resource_file):
            in_use_resources = []
        else:
            from qd.qd_common import read_to_buffer
            content = read_to_buffer(self._in_use_resource_file)
            if len(content) == 0:
                in_use_resources = []
            else:
                in_use_resources = load_from_yaml_file(self._in_use_resource_file)
        in_use_resources2 = []
        for record in in_use_resources:
            in_use_process_id = record['pid']
            from qd.remote_run import process_exists
            if not process_exists(in_use_process_id):
                logging.info('{} is not running. ignore that record'.format(
                    in_use_process_id))
                continue
            in_use_resources2.append(record)
        write_to_yaml_file(in_use_resources2,
                self._in_use_resource_file)
        return in_use_resources2

    def _check_if_in_use(self, resource):
        result = False
        if op.isfile(self._in_use_resource_file):
            in_use_resources = load_from_yaml_file(self._in_use_resource_file)
            if in_use_resources is None:
                in_use_resources = []
            in_use_resources2 = []
            for record in in_use_resources:
                in_use_process_id = record['pid']
                in_use_ip = record.get('ip', '0')
                in_use_gpus = record['gpus']
                from qd.remote_run import process_exists
                if not process_exists(in_use_process_id):
                    logging.info('{} is not running. ignore that record'.format(
                        in_use_process_id))
                    continue
                in_use_resources2.append(record)
                if resource[0].get('ip', '0') == in_use_ip and \
                        record.get('port', '0') == resource[0].get('-p', 22) and \
                        any(g in resource[1] for g in in_use_gpus):
                    result = True
                    logging.info('resouce is in use \n{}'.format(pformat(resource)))
                    break
            write_to_yaml_file(in_use_resources2,
                    self._in_use_resource_file)
        else:
            logging.info('{} not exists'.format(self._in_use_resource_file))
        return result

    def _is_in_use(self, resource):
        return self._run_in_lock(lambda:  self._check_if_in_use(resource))

    def run(self):
        self._in_progress = []
        self._has_synced = {}
        while True:
            in_progress = []
            for resource, task, p in self._in_progress:
                if not p.is_alive():
                     p.join()
                     self._release_resouce(resource)
                     if p.exitcode != 0:
                        for _, __, x in self._in_progress:
                            x.terminate()
                        logging.info(pformat(resource))
                        logging.info(pformat(task))
                        assert False

                     self._all_resouce.put(resource)
                else:
                    in_progress.append((resource, task, p))
            self._in_progress = in_progress

            if self._all_task.empty():
                break
            if self._all_resouce.empty():
                time.sleep(5)
                continue
            resource = self._all_resouce.get()

            if self._availability_check:
                avail = True
                if not gpu_available([resource]):
                    logging.info('{} is occupied from nvidia-smi'.format(
                        resource))
                    avail = False
                if avail and self._is_in_use(resource):
                    logging.info('{} is in possesion of other process'.format(
                        resource))
                    avail = False
                if not avail:
                    self._all_resouce.put(resource)
                    logging.info('resouce ({}) is not available. #task left {}'.format(
                        resource,
                        self._all_task.qsize()))
                    time.sleep(5)
                    continue
                if not self._try_use_resource(resource):
                    logging.info('fails to try to use {}'.format(pformat(resource)))
                    continue

            logging.info('resource ({}) is available'.format(resource))

            task = self._all_task.get()
            if len(resource[0]) > 0 and resource[0]['ip'] not in self._has_synced:
                sync_qd(ssh_info=resource[0], delete=False)
                self._has_synced[resource[0]['ip']] = True

            p = mp.Process(target=self._processor, args=(resource, task))
            p.start()
            self._in_progress.append((resource, task, p))
            if is_cluster(resource[0]):
                time.sleep(5)

        for resource, task, p in self._in_progress:
            p.join()
            self._release_resouce(resource)
        logging.info('done')
        #notify_by_email('Job finished', self._task_description)


