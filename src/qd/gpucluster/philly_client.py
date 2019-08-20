#!/usr/bin/env python
import os.path as op
from qd.qd_common import retry_agent, cmd_run
from qd.qd_common import ensure_directory
from qd.qd_common import load_from_yaml_file
from qd.qd_common import load_from_yaml_str
from qd.qd_common import url_to_str
from qd.qd_common import dict_update_nested_dict
from qd.qd_common import print_table
from qd.qd_common import decode_general_cmd
from qd.qd_common import print_job_infos
from qd.qd_common import attach_log_parsing_result
from qd.cloud_storage import create_cloud_storage
import logging
import simplejson as json
from pprint import pformat
import random
import shutil
import time
import re
import base64
import copy
import os
from collections import OrderedDict
import glob
from deprecated import deprecated
from qd.qd_common import get_file_size

from qd.qd_common import dump_to_yaml_str
from qd.qd_common import init_logging
from qd.qd_common import list_to_nested_dict
from qd.qd_common import list_to_dict
from qd.gpu_util import parse_gpu_usage_dict


def get_philly_fs():
    return op.expanduser('~/code/philly/philly-fs.bash')

def infer_type(vc, cluster):
    all_prem = [('resrchvc', 'gcr'),
            ('pnrsy', 'rr1')]
    all_ap = [('input', 'philly-prod-cy4')]
    all_azure = [
            ('input', 'sc2'),
            ('input', 'eu2'),
            ('input', 'wu1'),
            ('pnrsy', 'eu1'),
            ('pnrsy', 'sc1'),
            ('pnrsy', 'et1')]
    if any(v == vc and c == cluster for v, c in all_prem):
        return 'prem'
    elif any(v == vc and c == cluster for v, c in all_ap):
        return 'ap'
    elif any(v == vc and c == cluster for v, c in all_azure):
        return 'azure'
    assert False

def philly_input_run(cmd, return_output=False):
    # the username here is not improved since we do not use this function any
    # longer. Keep it here for reference only.
    i = 0
    while True:
        try:
            working_dir = op.expanduser('~/code/philly/philly-fs-ap-v5')
            logging.info('working dir: {}'.format(working_dir))
            result = cmd_run(cmd, env={'PHILLY_USER': 'jianfw',
                'PHILLY_VC': 'input',
                'PHILLY_CLUSTER_HDFS_HOST': '131.253.41.35',
                'PHILLY_CLUSTER_HDFS_PORT':
                '81/nn/http/hnn.philly-prod-cy4.cy4.ap.gbl/50070'},
                working_dir=working_dir,
                return_output=return_output)
            logging.info('succeed: {}'.format(' '.join(cmd)))
            return result
        except Exception:
            logging.info('fails: try {}-th time'.format(i))
            i = i + 1
            import time
            time.sleep(5)

def philly_run(sub_cmd, vc, cluster, return_output=False, extra_env={}):
    '''
    sub_cmd = ['-ls', 'path']
    '''
    t = infer_type(vc, cluster)
    if t == 'prem' or t == 'ap':
        disk_type = 'hdfs'
    else:
        disk_type = 'gfs'
    folder_prefix = '{}://{}/{}/'.format(disk_type, cluster, vc)
    if sub_cmd[0] == '-mkdir' or \
            sub_cmd[0] == '-ls':
        assert len(sub_cmd) == 2
        sub_cmd[1] = '{}{}'.format(folder_prefix, sub_cmd[1])
    elif sub_cmd[0] == '-cp':
        '''
        sub_cmd[-1] is the index of which one is in philly
        '''
        sub_cmd[sub_cmd[-1]] = '{}{}'.format(folder_prefix,
                sub_cmd[sub_cmd[-1]])
        del sub_cmd[-1]
    else:
        assert False
    if t == 'prem' or t == 'azure':
        philly_cmd = get_philly_fs()
        cmd = []
        cmd.append(philly_cmd)
        cmd.extend(sub_cmd)
        env = {'PHILLY_VC': vc}
        env.update(extra_env)
        return retry_agent(cmd_run, cmd,
                return_output=return_output, env=env)
    elif t == 'ap':
        cmd = []
        cmd.append('./philly-fs')
        cmd.extend(sub_cmd)
        output = philly_input_run(cmd, return_output)
        if output:
            logging.info(output)
        return output

def philly_mkdir(dest_dir, vc, cluster):
    sub_cmd = ['-mkdir', dest_dir]
    philly_run(sub_cmd, vc, cluster)

def upload_through_blob(src_dir, dest_dir, vc, cluster, **kwargs):
    assert (len(dest_dir) > 0 and \
            dest_dir[0] != '/' and \
            dest_dir[0] != '\\')
    if dest_dir[-1] != '/':
        dest_dir = dest_dir + '/'
    account = 'vig'
    dest_url = op.join('https://{}.blob.core.windows.net/data',
            account,
            dest_dir,
            op.basename(src_dir))

    c = create_cloud_storage(account)
    if op.isfile(src_dir):
        # it seems like az_sync does not support file -> file though it claims
        # it supports
        dest_url, _ = c.az_upload2(src_dir, op.join(dest_dir, op.basename(src_dir)))
    else:
        dest_url, _ = c.az_sync(src_dir, op.join(dest_dir, op.basename(src_dir)))
    if kwargs.get('copy_to_hdfs', True):
        env = {'AZURE_STORAGE_ACCESS_KEY': c.account_key}

        sub_cmd = ['-cp', '-r', dest_url, op.join(dest_dir,
            op.basename(src_dir)), 3]
        philly_run(sub_cmd, vc, cluster, extra_env=env)

def philly_upload_dir(src_dir, dest_dir, vc='input', cluster='philly-prod-cy4',
        blob=True, **kwargs):
    philly_mkdir(dest_dir, vc, cluster)
    t = infer_type(vc, cluster)
    if t == 'prem' or t == 'ap':
        disk_type = 'hdfs'
    else:
        disk_type = 'gfs'
    src_dir = op.realpath(src_dir)
    folder_prefix = '{}://{}/{}/'.format(disk_type, cluster, vc)
    if t == 'prem' or t == 'azure':
        if not blob:
            philly_cmd = op.expanduser('~/code/philly/philly-fs.bash')
            cmd = []
            cmd.append(philly_cmd)
            cmd.append('-cp')
            cmd.append('-r')
            cmd.append(src_dir)
            cmd.append('{}{}'.format(folder_prefix, op.join(dest_dir,
                op.basename(src_dir))))
            retry_agent(cmd_run, cmd, env={'PHILLY_VC': vc})
        else:
            upload_through_blob(src_dir, dest_dir, vc, cluster, **kwargs)
    elif t == 'ap':
        cmd = []
        cmd.append('./philly-fs')
        cmd.append('-cp')
        cmd.append('-r')
        cmd.append(src_dir)
        cmd.append('{}{}'.format(folder_prefix, dest_dir))
        philly_input_run(cmd)

def philly_download(src, dest, vc, cluster):
    dest = op.realpath(dest)
    ensure_directory(dest)
    sub_cmd = ['-cp', '-r', src, dest, 2]
    philly_run(sub_cmd, vc, cluster)

def philly_upload(src_file, dest_dir, vc='input', cluster='philly-prod-cy4'):
    '''
    dest_dir: data/abc
    '''
    if not dest_dir.endswith('/'):
        dest_dir = dest_dir + '/'
    philly_run(['-cp', op.realpath(src_file), dest_dir, 2], vc, cluster)

def create_philly_client(**kwargs):
    param = load_from_yaml_file('./aux_data/configs/philly_vc.yaml')
    dict_update_nested_dict(param, kwargs)
    if not param.get('password'):
        param['password'] = os.environ['PHILLY_PASSWORD']
    return PhillyVC(**param)

def create_multi_philly_client(**kwargs):
    param = load_from_yaml_file('./aux_data/configs/multi_philly_vc.yaml')
    dict_update_nested_dict(param, kwargs)
    return MultiPhillyVC(**param)

def get_http_prefix(vc_type):
    if vc_type == 'ap':
        return 'http://phillyonap'
    elif vc_type == 'azure':
        return 'https://philly'
    else:
        return 'https://philly'

def decode_config_extra_param(extra_param):
    return load_from_yaml_str(base64.b64decode(extra_param))

class PhillyVC(object):
    status_running = 'Running'
    status_queued = 'Queued'
    def __init__(self, vc, cluster, user_name=None,
            **kwargs):
        self.vc = vc
        self.cluster = cluster
        vc_type = infer_type(vc, cluster)
        self.vc_type = vc_type
        self.random_id = None
        if 'num_gpu' in kwargs:
            self.num_gpu = kwargs['num_gpu']
        else:
            if self.vc_type == 'azure':
                self.num_gpu = 4
            elif self.vc_type == 'ap':
                self.num_gpu = 8
            else:
                self.num_gpu = 1
        self.isDebug = kwargs.get('isDebug', False)
        self.user_name = user_name

        self.password = kwargs.get('password')

        self.src_config_path = 'src/qd/gpucluster/philly_server.py'
        self.dest_config_folder = '{}/code'.format(self.user_name)
        self.dest_config_file = op.join(self.dest_config_folder,
                op.basename(self.src_config_path))

        self.blob_mount_point = kwargs['blob_mount_point']
        self.config_param = kwargs['config_param']
        self.docker = kwargs['docker']
        self.dry_run = kwargs.get('dry_run')
        self.multi_process = kwargs.get('multi_process')
        self.docker_tag = kwargs.get('docker_tag')

        # used in query()
        self.query_with_gpu = kwargs.get('with_gpu')
        # we will no longer use this and will figure out the gpu informatoin
        # from log since philly_server.py will print nvidia-smi results
        self.query_with_gpu = False
        self.query_with_log = kwargs.get('with_log')
        self.azure_blob_config_file = kwargs.get('azure_blob_config_file')

    @property
    def use_blob_as_input(self):
        return self.config_param['data_folder'].startswith(
                self.blob_mount_point)

    def get_cloud_storage(self):
        return create_cloud_storage('vig')

    def get_data_folder_in_blob(self):
        assert self.config_param['data_folder'].startswith(self.blob_mount_point)
        result = self.config_param['data_folder'][len(self.blob_mount_point): ]
        if result.startswith('/'):
            result = result[1:]
        return result

    def get_output_folder_in_blob(self):
        assert self.config_param['output_folder'].startswith(self.blob_mount_point)
        result = self.config_param['output_folder'][len(self.blob_mount_point): ]
        if result.startswith('/'):
            result = result[1:]
        return result

    def get_summary(self):
        cmd = 'https://philly/api/summary?clusterId={}'.format(
                self.cluster)
        summary = self.philly_rest_api(cmd)
        summary = json.loads(summary)
        result = {}
        if 'ExceptionType' in summary:
            result['quota'] = 0
            result['activeGpus'] = 400
        else:
            result['quota'] = summary['queueStatus']['virtualClusters'][self.vc]['quota']
            result['activeGpus'] = summary['activeGPUsByVc'][self.vc]

        return result

    def sync_code(self, random_id):
        self.random_id = random_id

        random_qd = 'quickdetection{}'.format(random_id)
        random_abs_qd = op.join('/tmp', '{}.zip'.format(random_qd))
        logging.info('{}'.format(random_qd))
        from qd.qd_common import zip_qd
        zip_qd(random_abs_qd)
        copy_to_hdfs = not self.use_blob_as_input
        if infer_type(self.vc, self.cluster) == 'azure':
            if self.use_blob_as_input:
                self.config_param['code_path']
                from qd.cloud_storage import blob_upload
                assert self.config_param['code_path'].startswith(self.blob_mount_point)
                rel_code_path = self.config_param['code_path'][
                        len(self.blob_mount_point): ]
                blob_upload(random_abs_qd, rel_code_path)
            else:
                philly_upload_dir(random_abs_qd, '{}/code'.format(self.user_name),
                        vc=self.vc,
                        cluster=self.cluster, blob=True, copy_to_hdfs=copy_to_hdfs)
        else:
            philly_upload(random_abs_qd, '{}/code'.format(self.user_name), vc=self.vc,
                    cluster=self.cluster)

    def search_candidates(self, partial_id, my_own=True):
        all_job_info = self.query_all_job(my_own=my_own)
        all_job_info = [j for j in all_job_info if
                j['appID'].endswith(partial_id)]
        return all_job_info

    def search_job_id(self, partial_id):
        all_job_info = self.query_all_job(my_own=False)
        all_job_info = [j for j in all_job_info if
                j['appID'].endswith(partial_id)]
        assert len(all_job_info) == 1, ([j['appID'] for j in all_job_info], partial_id)
        return all_job_info[0], all_job_info[0]['appID']

    def abort(self, application_id):
        cmd = 'https://philly/api/abort?clusterId={}&jobId={}'.format(
                self.cluster, application_id)
        self.philly_rest_api(cmd)

    def query_all_job(self, my_own=True, numFinishedJobs=None):
        cmd="{}/api/list?".format(self.get_http_prefix())
        param = ['clusterId={}'.format(self.cluster),
                'vcId={}'.format(self.vc),
                ]
        if my_own:
            param.append('userName={}'.format(self.user_name))
            if numFinishedJobs is None:
                numFinishedJobs = 250
        else:
            # default is 25. it is for all users if my_own is False, and
            # maybe very small
            if numFinishedJobs is None:
                numFinishedJobs = 500
        param.append('numFinishedJobs={}'.format(numFinishedJobs))
        cmd += '&'.join(param)
        while True:
            result = self.philly_rest_api(cmd)
            result = json.loads(result)
            if 'ExceptionType' in result:
                logging.info(pformat(result))
                time.sleep(1)
                continue
            else:
                break
        all_job_info = []
        for running_type in ['runningJobs', 'queuedJobs', 'finishedJobs']:
            all_job = result[running_type]
            for philly_job_status in all_job:
                job_status = self.parse_status(philly_job_status)
                job_info = {}
                for k in job_status.keys():
                    job_info[k] = job_status[k]
                job_info['ssh'] = self.get_ssh_command(philly_job_status)
                all_job_info.append(job_info)
        for job_info in all_job_info:
            hour, minute, second = list(map(float,
                    job_info['elapsedTime'].split(':')))
            job_info['elapsedTime'] = round(hour + minute / 60. + second /
                    3600., 2)
        for job_info in all_job_info:
            job_info['cluster'] = self.cluster
            job_info['vc'] = self.vc
            job_info['vc_type'] = self.vc_type

        return all_job_info

    def update_config(self):
        if not self.use_blob_as_input:
            self.upload_file(self.src_config_path, self.dest_config_folder)
        else:
            c = create_cloud_storage(
                    config_file=self.azure_blob_config_file)
            c.az_upload2(self.src_config_path,
                    op.join(self.dest_config_folder,
                        op.basename(self.src_config_path)))

    def attach_log(self, job_info):
        job_info['latest_log'] = self.philly_job_log(job_info)

    def attach_gpu_utility(self, all_job_info):
        for job_info in all_job_info:
            if not job_info['ssh']:
                continue
            try:
                gpu_result = cmd_run('{} nvidia-smi'.format(
                    job_info['ssh']).split(' '), shell=True, return_output=True)
                gpu_info = parse_gpu_usage_dict(gpu_result)
                import numpy as np
                gpu_info = {t: np.mean([g[t] for g in gpu_info]) for t in ['mem_used',
                        'mem_total', 'gpu_util']}
                for k in gpu_info:
                    job_info[k] = gpu_info[k]
            except Exception as e:
                logging.info(str(e))

    def query(self, valid_job_checker=None, my_own=True):
        all_job_info = self.query_all_job(my_own)
        if valid_job_checker is not None:
            all_job_info = [j for j in all_job_info if valid_job_checker(j)]
        if self.query_with_gpu:
            self.attach_gpu_utility([j for j in all_job_info if j['status'] ==
                    'Running'])
        self.attach_meta(all_job_info)
        if self.query_with_log:
            for job_info in all_job_info:
                if job_info['status'] != self.status_running:
                    continue
                self.attach_log(job_info)
                attach_log_parsing_result(job_info)
        for j in all_job_info:
            j['appID-s'] = j['appID'][-5:]
        return all_job_info

    def attach_meta(self, all_job_info):
        all_meta = self.query_meta_data([j['appID'] for j in all_job_info])
        self.parse_meta_data(all_meta)
        for job_info, meta in zip(all_job_info, all_meta):
            job_info['meta'] = meta

        # we want to access these 3 fields if it has
        keys = ['data', 'net', 'expid']
        for job_info in all_job_info:
            for k in keys:
                job_info[k] = job_info['meta'].get('param',
                        {}).get(k)

        meta_keys = ['num_gpu']
        for job_info in all_job_info:
            for k in meta_keys:
                job_info[k] = job_info['meta'][k]

    def query_meta_data(self, job_ids):
        result = []
        for job_id in job_ids:
            meta = json.loads(self.philly_job_meta(job_id))
            result.append(meta)
        return result

    def parse_meta_data(self, all_meta):
        for meta in all_meta:
            extraParam = meta['cmd']
            re_result = re.match('.* -- (.*)', extraParam)
            if re_result and len(re_result.groups()) == 1:
                meta['extra_param'] = re_result.groups()[0]
                ps = load_from_yaml_str(base64.b64decode(meta['extra_param']))
                command_parse = re.match('python .* -bp (.*)', ps['command'])
                if command_parse:
                    param = load_from_yaml_str(base64.b64decode(command_parse.groups()[0]))
                    meta['param'] = param.get('param', {})
        for meta in all_meta:
            meta['num_gpu'] = meta['gpusPerContainer'] * meta['minContainers']

    def get_config_extra_param(self, command):
        dict_param = {
                'code_path': self.config_param['code_path'],
                'data_folder': self.config_param['data_folder'],
                'model_folder': self.config_param['model_folder'],
                'output_folder': self.config_param['output_folder'],
                'command': command}
        extraParam = base64.b64encode(dump_to_yaml_str(dict_param)).decode()

        return extraParam

    def philly_submit_v2(self, jobname, num_gpu, command):
        cluster, vc = self.cluster, self.vc
        if cluster == 'philly-prod-cy4':
            submit_url = 'http://phillyonap/api/v2/submit'
            registry = 'phillyregistry.azurecr.io.apdocker.ap.gbl'
        else:
            registry = 'phillyregistry.azurecr.io'
            submit_url = 'https://philly/api/v2/submit'
        tag = self.docker['tag']
        if self.docker_tag:
            tag = self.docker_tag
        assert len(command) > 0
        extraParam = self.get_config_extra_param(command)
        logging.info('extraParam: {}'.format(extraParam))
        # no matter it is blobfuse or hdfs, we always use blobfuse for config
        # file
        config_file = op.join(self.blob_mount_point,
                self.dest_config_file)
        #config_file = "/hdfs/{}/{}/{}".format(self.vc,
            #self.dest_config_folder, op.basename(self.src_config_path))
        logging.info('cmd:\n{} d d d {}'.format(config_file, extraParam))
        custom_mpi_args = 'env CUDA_CACHE_DISABLE=1 NCCL_DEBUG=INFO OMP_NUM_THREADS=2'
        disable_ib = False
        if disable_ib:
            custom_mpi_args += ' NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0'
        if num_gpu > 4:
            if tag == 'py36pt1.1':
                # without this flag, distributed training will crash in multi-node
                # more information: https://github.com/horovod/horovod/issues/893
                custom_mpi_args += ' NCCL_TREE_THRESHOLD=0'
                # without the following flag, the multi-node distributed training
                # will hangs in pytorch1.1, which uses NCCL 2.4.2. The problem got
                # fixed in NCCL 2.4.6.
                # https://github.com/pytorch/pytorch/issues/20630
                custom_mpi_args += ' NCCL_LL_THRESHOLD=0'
                # not sure if the following can fix the problem, described also
                # in https://github.com/NVIDIA/nccl/issues/230
                custom_mpi_args += ' NCCL_IB_TIMEOUT=24'

        data = {
            "ClusterId": cluster,
            "VcId": vc,
            "JobName": jobname,
            "UserName": self.user_name,
            "BuildId": 0,
            "ToolType": None,
            "ConfigFile": config_file,
            "Inputs": [{
                "Name": "dataDir",
                "Path": "/hdfs/{}/{}".format(self.vc, self.user_name)
            }],
            "Outputs": [],
            "IsDebug": self.isDebug,
            "CustomDockerName": None,
            "RackId": "anyConnected",
            "MinGPUs": num_gpu,
            "PrevModelPath": None,
            "ExtraParams": extraParam,
            "SubmitCode": "p",
            "IsMemCheck": False,
            "IsCrossRack": False,
            "Registry": registry,
            "Repository": self.docker['image'],
            "Tag": tag,
            "CustomMPIArgs": custom_mpi_args,
            "Timeout":None,
            }
        gpus_per_node = 4
        assert cluster in ['sc2', 'wu1'], 'need to update gpus_per_node'
        num_containers = (num_gpu + gpus_per_node - 1) // gpus_per_node
        #data["NumOfContainers"] = "2"
        data["NumOfContainers"] = str(num_containers)
        if not self.multi_process:
            data["OneProcessPerContainer"] = True
            data["DynamicContainerSize"] = False
        else:
            data["OneProcessPerContainer"] = False
            data["DynamicContainerSize"] = False

        cloud_blob = create_cloud_storage(
                config_file=self.azure_blob_config_file)
        blob_container = cloud_blob.container_name
        blob_key = cloud_blob.account_key

        data['volumes'] = {'blob': {'type': 'blobfuseVolume',
            'storageAccount': cloud_blob.account_name,
            'containerName': blob_container,
            'path': self.blob_mount_point,
            "options": [
                "-o", "attr_timeout=240",
                "-o", "entry_timeout=240",
                "-o", "negative_timeout=120",
                "--log-level=LOG_WARNING",
                "-o", "allow_other",
                "--file-cache-timeout-in-seconds=1000000",
                ]
            }}
        data['credentials'] = {'storageAccounts': {cloud_blob.account_name: {
            '_comments': 'redentials for accessing the storage account.',
            'key': blob_key,}}}

        if self.use_blob_as_input:
            data['Inputs'][0]['Path'] = '/blob/{}'.format(self.user_name)

        en_data = json.dumps(data)

        cmd = ['curl', '-H', 'Content-Type: application/json',
            '-H', 'WWW-Authenticate: Negotiate',
            '-H', 'WWW-Authenticate: NTLM',
            '--user', 'redmond\\{}:{}'.format(self.user_name, self.password),
            '-X', 'POST', submit_url, '-k', '--ntlm',
            '-n', '-d', "{}".format(en_data)]
        if self.dry_run:
            if self.password is None:
                user_info = '"redmond\\{}"'.format(self.user_name)
            else:
                user_info = '"redmond\\{}:{}"'.format(self.user_name, self.password)
            cmd = ['curl', '-H', '"Content-Type: application/json"',
                '-H', '"WWW-Authenticate: Negotiate"',
                '-H', '"WWW-Authenticate: NTLM"',
                '--user', user_info,
                '-X', 'POST', submit_url, '-k', '--ntlm',
                '-n', '-d', "'{}'".format(en_data)]
            logging.info('\n' + ' '.join(cmd))
        else:
            result_str = cmd_run(cmd, return_output=True)
            return result_str

    def submit_without_sync(self, extraParam, num_gpu=None):
        '''
        use submit() because of bad naming here.
        '''
        if num_gpu is None:
            num_gpu = self.num_gpu
        result = self.philly_submit_v2(str(self.random_id), num_gpu,
                extraParam)
        if result:
            result = json.loads(result)
            if 'error' in result:
                logging.info(result['error'])
                raise Exception
            if 'ExceptionType' in result and result['ExceptionType']:
                logging.info(result.get('StackTrace'))
                raise Exception
            job_id = result['jobId']
            logging.info('job_id = {}'.format(job_id))
            return job_id

    def get_random_qd(self):
        return 'quickdetection_{}'.format(self.random_id)

    def track_job_once(self, app_info):
        job_id = app_info['appID']
        result = self.philly_job_log(app_info)
        logging.info(result)

        status = json.loads(self.philly_job_status(job_id))

        meta = decode_general_cmd(json.loads(self.philly_job_meta(job_id))['cmd'])
        logging.info(pformat(meta))

        logging.info('satus = {}'.format(status['status']))
        ssh_command = self.get_ssh_command(status)
        logging.info(ssh_command)
        url = 'https://philly/#/job/{}/{}/{}'.format(self.cluster,
                self.vc, job_id[len('application_'): ])
        logging.info(url)

        return status

    def get_ssh_command_ap(self, status):
        if 'appID' not in status:
            logging.info('appID not in status: {}'.format(pformat(status)))
            return
        src_id_rsa_file = '/mnt/proxy/proxypap/sys/jobs/{}/id_rsa'.format(status['appID'])
        if not op.isfile(src_id_rsa_file):
            logging.info('the file not exists: {}'.format(src_id_rsa_file))
            return
        dest_id_rsa_file = '/tmp/{}/id_rsa'.format(status['appID'])
        if not op.isfile(dest_id_rsa_file):
            ensure_directory(op.dirname(dest_id_rsa_file))
            shutil.copyfile(src_id_rsa_file, dest_id_rsa_file)
            cmd_run(['chmod', '400', dest_id_rsa_file])
        for d in status['detail']:
            if not d['isMaster']:
                port = d['port']
                ip = d['ip']
        cmd = ['ssh',
                '-o', "ProxyCommand='ssh pap -W %h:%p'",
                '-i', dest_id_rsa_file,
                '-p', str(port),
                '{}@'.format(self.user_name) + ip]
        result = ' '.join(cmd)
        logging.info('ssh command: {}'.format(result))
        return result

    def parse_status(self, status):
        if 'appID' not in status:
            logging.info('appID not in status: {}'.format(pformat(status)))
            return
        port = None
        for d in status['detail']:
            if not d['isMaster']:
                port = d['port']
                ip = d['ip']
        result = copy.deepcopy(status)
        result['name'] = status['name'][:-6] if status['name'].endswith('!~!~!4') else status['name']
        if port:
            result['port'] = port
            result['ip'] = ip
        return result

    def get_ssh_command(self, status):
        if self.cluster == 'philly-prod-cy4':
            return self.get_ssh_command_ap(status)
        info = self.parse_status(status)
        if info and 'port' in info and 'ip' in info:
            result = ['ssh -tt {}@sshproxy.{}.philly.selfhost.corp.microsoft.com -p 2200'.format(
                self.user_name,
                self.cluster)]
            result.append('ssh -tt -o StrictHostKeyChecking=no {}@{} -p {}'.format(
                info['username'],
                info['ip'], info['port']))
            result.append('-i')
            result.append('/var/storage/shared/input/sys/jobs/{}/.ssh/id_rsa'.format(info['appID']))
            cmd = ' '.join(result)
            return cmd

    def get_http_prefix(self):
        if self.vc_type == 'ap':
            return 'http://phillyonap'
        elif self.vc_type == 'azure':
            return 'https://philly'
        else:
            return 'https://philly'

    def philly_job_log(self, job_info, logRev='latest'):
        job_id = job_info['appID']
        cmd_str = \
            '{}/api/log?clusterId={}&vcId={}&jobId={}&jobType=cust&logType=stdout&logRev={}&content=partial'.format(
                self.get_http_prefix(),
                self.cluster,
                self.vc,
                job_id, logRev)
        result = self.philly_rest_api(cmd_str)

        if 'WARNING' in result and 'too big for preview' in result:
            # wget https://storage.sc2.philly.selfhost.corp.microsoft.com/input/sys/jobs/application_1544809666047_4657/stdout/1/stdout.txt
            retries = job_info.get('retries') + 1
            url = 'https://storage.{}.philly.selfhost.corp.microsoft.com/input/sys/jobs/{}/stdout/{}/stdout.txt'.format(
                self.cluster, job_id, retries)
            logging.info('get log from {}'.format(url))
            result = url_to_str(url)

        return result

    def philly_job_status(self, job_id):
        cmd_str = \
            '{}/api/status?clusterId={}&vcId={}&jobId={}&jobType=cust&content=full'.format(
                    self.get_http_prefix(),
                    self.cluster,
                    self.vc,
                    job_id)
        result = self.philly_rest_api(cmd_str)
        return result

    def philly_job_meta(self, job_id):
        cmd_str = \
            '{}/api/metadata?clusterId={}&vcId={}&jobId={}'.format(
                    self.get_http_prefix(),
                    self.cluster,
                    self.vc,
                    job_id)
        result = self.philly_rest_api(cmd_str)
        return result

    def upload_file(self, file_from, file_target, blob=False):
        if self.cluster in ['sc2', 'wu1']:
            philly_upload_dir(file_from, file_target, self.vc,
                    self.cluster, blob=blob)
        else:
            philly_upload(file_from, file_target, self.vc, self.cluster)

    def increment_upload_dir_to_hdfs(self, src_dir, dest_dir):
        '''
        exg. ./data/coco2017Full, jianfw/data/qd_data/
        '''
        if not dest_dir.endswith('/'):
            dest_dir = dest_dir + '/'
        file_infos = parse_philly_ls_output_rich(philly_ls(
            dest_dir, return_output=True, vc=self.vc, cluster=self.cluster))
        if not any(f['type'] == 'dir' and f['name'] == op.basename(src_dir)
                for f in file_infos):
            philly_mkdir(op.join(dest_dir, op.basename(src_dir)),
                    vc=self.vc, cluster=self.cluster)
        philly_file_infos = parse_philly_ls_output_rich(philly_ls(
            op.join(dest_dir, op.basename(src_dir)),
            return_output=True, vc=self.vc, cluster=self.cluster))
        for src_file in glob.glob(op.join(src_dir, '*')):
            if op.isfile(src_file):
                file_size = get_file_size(src_file)
                no_need = False
                for f in philly_file_infos:
                    if f['type'] == 'file' and \
                            f['name'] == op.basename(src_file) and \
                            f['file_size_in_byte'] == file_size:
                        no_need = True
                        break
                if not no_need:
                    self.upload_file(src_file, op.join(dest_dir,
                        op.basename(src_dir)), blob=True)
            elif op.isdir(src_file):
                self.increment_upload_dir_to_hdfs(src_file, op.join(dest_dir,
                    op.basename(src_dir)))

    def upload_qd_data(self, d):
        from qd.tsv_io import TSVDataset
        data_root = TSVDataset(d)._data_root
        if self.use_blob_as_input:
            cloud = self.get_cloud_storage()
            cloud.az_sync(data_root,
                    op.join(self.get_data_folder_in_blob(), d))
        else:
            self.increment_upload_dir_to_hdfs(data_root,
                    self.get_qd_data_rel_path_in_hdfs(self.config_param['data_folder'])
                    )

    def upload_qd_model(self, model_file):
        def split_all_path(fpath):
            path_splits = []
            dirname, basename = op.split(fpath)
            while basename:
                path_splits.append(basename)
                dirname, basename = op.split(dirname)
            path_splits.append(dirname)
            return path_splits[::-1]

        assert(op.isfile(model_file))
        path_splits = split_all_path(model_file)
        assert(len(path_splits) >= 3 and path_splits[-2] == "snapshot")
        target_path = op.join(self.get_output_folder_in_blob(),
                path_splits[-3], path_splits[-2], path_splits[-1])
        cloud = self.get_cloud_storage()
        if not cloud.exists(target_path):
            cloud.az_upload2(model_file, target_path)

    def get_qd_data_rel_path_in_hdfs(self, hdfs_path):
        prefix = '/hdfs/{}/'.format(self.vc)
        assert hdfs_path.startswith(prefix)
        hdfs_path = hdfs_path.replace(prefix, '')
        return hdfs_path

    def qd_data_exists(self, fname):
        # this assume the data folder in config
        if self.use_blob_as_input:
            cloud = self.get_cloud_storage()
            return cloud.exists(op.join(self.get_data_folder_in_blob(),
                    fname))
        else:
            # hdfs
            hdfs_file = op.join(self.config_param['data_folder'],
                    fname)
            hdfs_folder = op.dirname(hdfs_file)
            hdfs_folder = self.get_qd_data_rel_path_in_hdfs(hdfs_folder)
            philly_ls_result = philly_ls(hdfs_folder, vc=self.vc, cluster=self.cluster,
                    return_output=True)
            folder_info = parse_philly_ls_output_rich(philly_ls_result)
            return any(f for f in folder_info if f['name'] ==
                    op.basename(hdfs_file))

    def philly_rest_api(self, CMD):
        user_name, password = self.user_name, self.password
        cmd = ['curl', '-k', '--ntlm', '--user',
                '"redmond\\{}:{}"'.format(user_name, password),
                '"{}"'.format(CMD)]
        result_str = cmd_run(cmd, shell=True, return_output=True)
        return result_str

class MultiPhillyVC(object):
    def __init__(self, **kwargs):
        self.all_cluster = ['sc2', 'wu1']
        if 'cluster' in kwargs:
            # this is usually the case when we call philly.py from command line
            # by providing the cluster name
            self.all_cluster = [kwargs['cluster']]
            del kwargs['cluster']
        elif 'clusters' in kwargs:
            # this is the case when we load the parameters from the files
            self.all_cluster = kwargs['clusters']

        self.kwargs = kwargs
        self.clients = [create_philly_client(cluster=c, **self.kwargs) for c in
                self.all_cluster]
        self.cluster_to_client = dict(zip(self.all_cluster, self.clients))

    def select_client_for_submit(self):
        all_summary = [c.get_summary() for c in self.clients]
        avail_gpus = [s['quota'] - s['activeGpus'] for s in all_summary]
        max_idx = max(list(range(len(avail_gpus))), key=lambda x:
                avail_gpus[x])
        if avail_gpus[max_idx] > self.clients[0].num_gpu * 2:
            target_client = self.clients[max_idx]
            logging.info('select {} because of {} free gpus'.format(
                target_client.cluster, avail_gpus[max_idx]))
            return target_client

        all_jobs = [c.query_all_job(my_own=True) for c in self.clients]

        all_queue_jobs = [[j for j in jobs if j['status'] == 'Queued']
            for jobs in all_jobs]
        for c, q in zip(self.clients, all_queue_jobs):
            logging.info('cluster: {}; #jobs in queue: {}'.format(
                c.cluster, len(q)))
        min_idx = min(range(len(all_queue_jobs)), key=lambda x: len(all_queue_jobs[x]))
        target_client = self.clients[min_idx]
        logging.info('select {} because of less queuing jobs'.format(
            target_client.cluster))
        return target_client

    def search_job_id_and_track(self, partial_id):
        client, app_info = self.search_job_id(partial_id)
        return client.track_job_once(app_info)

    def search_job_id(self, partial_id):
        clients = [create_philly_client(cluster=c, **self.kwargs) for c in
                self.all_cluster]
        all_cans = [c.search_candidates(partial_id)
                for c in clients]
        all_cluster_cans = [(c, cans) for c, cans in zip(clients, all_cans) if len(cans) > 0]
        assert (len(all_cluster_cans) == 1 and
                len(all_cluster_cans[0][1]) == 1), \
            'ambigous job found: {}'.format('; '.join('{}: {}'.format(c.cluster,
                ','.join(s['appID'] for s in cans)) for c, cans in all_cluster_cans))
        client = all_cluster_cans[0][0]
        job_info = all_cluster_cans[0][1][0]
        return client, job_info

    def search_job_id_and_abort(self, partial_id):
        clients = [create_philly_client(cluster=c, **self.kwargs) for c in
                self.all_cluster]
        all_cans = [[x['appID'] for x in c.search_candidates(partial_id)]
                for c in clients]
        all_cluster_cans = [(c, cans) for c, cans in zip(clients, all_cans) if len(cans) > 0]
        assert (len(all_cluster_cans) == 1 and
                len(all_cluster_cans[0][1]) == 1), \
            'ambigous job found: {}'.format('; '.join('{}: {}'.format(c.cluster,
                ','.join(cans)) for c, cans in all_cluster_cans))
        client = all_cluster_cans[0][0]
        job_id = all_cluster_cans[0][1][0]
        client.abort(job_id)
        logging.info('aborted {} in the cluster of {}'.format(job_id,
            client.cluster))

    def query(self):
        all_job_info = []
        for c in self.clients:
            all_job_info.extend(c.query())

        if op.isfile('./aux_data/configs/extra_tracking_philly_jobs.yaml'):
            extra_jobs = load_from_yaml_file('./aux_data/configs/extra_tracking_philly_jobs.yaml')
            cluster_id = [(j['cluster'], j['appID']) for j in extra_jobs]
            cluster_to_ids = list_to_dict(cluster_id, 0)
            cluster_to_client = dict(zip(self.all_cluster, self.clients))
            for cluster, ids in cluster_to_ids.items():
                all_job_info.extend(cluster_to_client[cluster].query(valid_job_checker=lambda
                        j: j['appID'] in ids, my_own=False))
        all_status = list(set([j['status'] for j in all_job_info]))
        all_status = sorted(all_status, reverse=True)
        for status in all_status:
            print_job_infos([j for j in all_job_info if j['status'] == status])
        return all_job_info

    def inject(self):
        all_job_info = self.query()
        from qd.db import update_cluster_job_db
        update_cluster_job_db(all_job_info)

    def print_summary(self):
        all_summary = [c.get_summary() for c in self.clients]
        table = []

        for s, c in zip(all_summary, self.clients):
            row = {}
            row['usage'] = 100. * s['activeGpus'] / s['quota']
            row['cluster'] = c.cluster
            table.append(row)
        print_table(table, ['cluster', 'usage'])

    def auto_resubmit(self):
        all_job_info = [j for c in self.clients
                for j in c.query_all_job(my_own=True)]

        queued_jobs = [j for j in all_job_info if j['status'] == PhillyVC.status_queued]
        cluster_queued_jobs = [[j['cluster'], j] for j in queued_jobs]
        cluster_to_queued_jobs = list_to_dict(cluster_queued_jobs, 0)
        queued_job_clusters = set([j['cluster'] for j in queued_jobs])
        no_queued_job_clusters = set(self.all_cluster).difference(queued_job_clusters)
        if len(no_queued_job_clusters) == 0:
            logging.info('no cluster can be used to resubmit')
            return
        # select one job which is in the cluster containing more than 1 queued
        # jobs
        resubmit_job = None
        candidate_resubmit_jobs = [j for c, queued_jobs in cluster_to_queued_jobs.items() if
            len(queued_jobs) > 1 for j in queued_jobs if j['elapsedTime'] > 1.]
        if len(candidate_resubmit_jobs) == 0:
            logging.info('no job can be resubmitted')
            return
        resubmit_job = max(candidate_resubmit_jobs, key=lambda x: x['elapsedTime'])
        origin_client = self.cluster_to_client[resubmit_job['cluster']]
        origin_client.attach_meta([resubmit_job])
        cmd = parse_job_command(resubmit_job)['command']
        new_client = self.cluster_to_client[list(no_queued_job_clusters)[0]]
        new_client.submit_without_sync(cmd, num_gpu=resubmit_job['num_gpu'])
        origin_client.abort(resubmit_job['appID'])
        logging.info('done')

def parse_philly_ls_output_rich(output):
    lines = output.split('\n')
    assert len(lines) > 0
    while True:
        line = lines[0]
        import re
        r = re.match('total ([0-9]*)', line)
        if r:
            break
        else:
            logging.info('ignore {}'.format(line))
            lines = lines[1:]
    num_rows = int(float(r.groups()[0]))
    result = []
    for i in range(num_rows):
        line = lines[i + 1]
        # -rwxrwxrwx       1  519178854    519178854 4,218,683,392 2018-10-19 17:03:29 .train.tsv.lHKomx
        p = '(.{1}).* ([0-9,]*) ([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) *(.*)'
        r = re.match(p, line)
        file_type, file_size_in_byte, time_stamp, file_name = r.groups()
        info = {}
        if file_type == '-':
            info['type'] = 'file'
        else:
            assert file_type == 'd'
            info['type'] = 'dir'
        info['file_size_in_byte'] = int(file_size_in_byte.replace(',', ''))
        from dateutil.parser import parse
        info['time'] = parse(time_stamp)
        info['name'] = file_name
        result.append(info)
    return result

def parse_philly_ls_output(output):
    lines = output.split('\n')
    assert len(lines) > 0
    while True:
        line = lines[0]
        import re
        r = re.match('total ([0-9]*)', line)
        if r:
            break
        else:
            logging.info('ignore {}'.format(line))
            lines = lines[1:]
    num_rows = int(float(r.groups()[0]))
    all_file = []
    all_dir = []
    for i in range(num_rows):
        line = lines[i + 1]
        # -rwxrwxrwx       1  519178854    519178854 4,218,683,392 2018-10-19 17:03:29 .train.tsv.lHKomx
        p = '(.{1}).*[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}[ ]*(.*)'
        r = re.match(p, line)
        file_type, file_name = r.groups()
        if file_type == '-':
            all_file.append(file_name)
        else:
            assert file_type == 'd'
            all_dir.append(file_name)
    return all_file, all_dir

def upload_qdoutput(src_path, dest_path, ssh_info):
    # make sure the folder of dest_path in philly exists
    remote_run('mkdir -p {}'.format(dest_path), ssh_info)

    # upload all the files under src_path
    all_src_file = glob.glob(op.join(src_path, '*'))
    for f in all_src_file:
        if op.isfile(f):
            scp(f, dest_path, ssh_info)

    # for the model and the tested files, only upload the best
    all_src_file = glob.glob(op.join(src_path, 'snapshot', 'model_iter_*'))
    all_iter = [parse_iteration2(f) for f in all_src_file]
    max_iters = max(all_iter)
    need_copy_files = [f for f, i in zip(all_src_file, all_iter) if i == max_iters]
    dest_snapshot = op.join(dest_path, 'snapshot')
    remote_run('mkdir -p {}'.format(dest_snapshot), ssh_info)
    for f in need_copy_files:
        scp(f, dest_snapshot, ssh_info)


def philly_ls(dest_dir, vc='input', return_output=False, cluster='philly-prod-cy4'):
    sub_cmd = ['-ls', dest_dir]
    return philly_run(sub_cmd, vc, cluster, return_output)

@deprecated('pls use convert_to_command_line')
def convert_to_philly_extra_command(param, script='scripts/tools.py'):
    from qd.qd_common import convert_to_command_line
    return convert_to_command_line(param, script)

def submit_without_sync(cmd, **kwargs):
    all_extra_param = []
    if cmd == 'gc':
        params = {'type': 'del_intermediate_models'}
        extra_param = convert_to_philly_extra_command(params,
                script='garbage_collector')
        all_extra_param.append(extra_param)
    elif cmd == 'ssh':
        kwargs.update({'isDebug': True})
        extra_param = 'ls'
        all_extra_param.append(extra_param)
    else:
        all_extra_param.append(cmd)

    logging.info(all_extra_param)
    p = create_philly_client(**kwargs)
    if kwargs.get('real_submit', True):
        list(map(lambda extra_param: p.submit_without_sync(extra_param),
                all_extra_param))

def tracking(app_id, **kwargs):
    p = MultiPhillyVC(**kwargs)
    p.search_job_id_and_track(app_id)

def list_to_dict_full(l, idx):
    result = OrderedDict()
    for x in l:
        if x[idx] not in result:
            result[x[idx]] = []
        result[x[idx]].append(x)
    return result

def blame(**kwargs):
    p = create_philly_client(**kwargs)
    all_job_info = p.query_all_job(False)

    all_username_status_gpus_queue = []
    for job in all_job_info:
        username = job['username']
        status = job['status']
        num_gpu = sum([len(d['gpus']) for d in job['detail']
                if not d['isMaster']])
        queue = job['queue']
        all_username_status_gpus_queue.append((username, status,
            num_gpu, queue))

    status_to_username_to_gpus = list_to_nested_dict([x[:-1] for x in all_username_status_gpus_queue], [1, 0])
    status_to_username_num_jobs_num_gpus = {}
    for status, username_to_gpus in status_to_username_to_gpus.items():
        username_num_jobs_num_gpus = []
        for username, all_gpus in username_to_gpus.items():
            username_num_jobs_num_gpus.append((username,
                len(all_gpus), sum(all_gpus)))
        status_to_username_num_jobs_num_gpus[status] = \
            sorted(username_num_jobs_num_gpus, key=lambda x: -x[-1])
    logging.info('\n{}'.format(pformat(status_to_username_num_jobs_num_gpus)))

    status_to_queue_to_username_to_gpus = \
        list_to_nested_dict(all_username_status_gpus_queue, [1, 3, 0])
    status_to_queue_to_username_num_jobs_num_gpus = {}
    for status, queue_to_username_to_gpus in status_to_queue_to_username_to_gpus.items():
        queue_info = []
        for queue, username_to_gpus in queue_to_username_to_gpus.items():
            username_num_jobs_num_gpus = []
            for username, gpus in username_to_gpus.items():
                username_num_jobs_num_gpus.append(
                        (username, len(gpus), sum(gpus)))
            x = sorted(username_num_jobs_num_gpus, key=lambda x: -x[-1])
            s = sum([y[-1] for y in x])
            queue_info.append((queue, s, x))
        queue_info = sorted(queue_info, key=lambda one_queue: -one_queue[1])
        status_to_queue_to_username_num_jobs_num_gpus[status] = queue_info
    logging.info('\n{}'.format(pformat(status_to_queue_to_username_num_jobs_num_gpus)))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Philly Interface')
    parser.add_argument('task_type',
            choices=['ssh', 'q', 'query', 'a', 'abort', 'submit', 'sync',
                'update_config', 'gc', 'blame', 'init', 'resubmit',
                'summary', 'i', 'inject'])
    parser.add_argument('-wl', '--with_log', default=False, action='store_true')
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-c', '--cluster', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-n', '--num_gpu', default=argparse.SUPPRESS, type=int)
    #parser.add_argument('-wg', '--with_gpu', default=True, action='store_true')
    parser.add_argument('-no-wg', '--with_gpu', default=True, action='store_false')
    #parser.add_argument('-m', '--with_meta', default=True, action='store_true')
    parser.add_argument('-no-m', '--with_meta', default=True, action='store_false')

    parser.add_argument('-no-s', '--real_submit', default=True, action='store_false')

    parser.add_argument('remainders', nargs=argparse.REMAINDER,
            type=str)
    return parser.parse_args()

def ensure_init_config_files():
    def infinite_check(config_file, config_template):
        if not op.isfile(config_file):
            logging.info('Please create {}. Template: {}'.format(
                config_file, config_template))
        while not op.isfile(config_file):
            time.sleep(5)
    config_file = './aux_data/configs/vigblob_account.yaml'
    config_template = './aux_data/configs/vigblob_account.template.yaml'
    infinite_check(config_file, config_template)

    config_file = './aux_data/configs/philly_vc.yaml'
    config_template = './aux_data/configs/philly_vc.template.yaml'
    infinite_check(config_file, config_template)

def parse_job_command(job_info):
    config_extra_param = job_info['meta']['extra_param']
    extra_param = decode_config_extra_param(config_extra_param)
    return extra_param

def abort_submit(partial_id, **kwargs):
    multi_param = copy.deepcopy(kwargs)
    if 'cluster' in multi_param:
        del multi_param['cluster']
    p = MultiPhillyVC(**multi_param)
    client, job_info = p.search_job_id(partial_id)
    client.attach_meta([job_info])

    submit_client = MultiPhillyVC(**kwargs)
    cmd = parse_job_command(job_info)
    submit_client = submit_client.select_client_for_submit()
    submit_client.config_param['data_folder'] = cmd['data_folder']
    num_gpu = kwargs['num_gpu'] if 'num_gpu' in kwargs else job_info['num_gpu']
    submit_client.submit_without_sync(
            cmd['command'], num_gpu)
    if job_info['status'] in ['Queued', 'Running']:
        client.abort(job_info['appID'])
    else:
        assert job_info['status'] in ['Failed', 'Pass', 'Killed']
    logging.info('Done')

def execute(task_type, **kwargs):
    if task_type in ['q', 'query']:
        if len(kwargs.get('remainders', [])) > 0:
            assert len(kwargs['remainders']) == 1
            tracking(kwargs['remainders'][0], **kwargs)
        else:
            p = MultiPhillyVC(**kwargs)
            p.print_summary()
            p.query()
    elif task_type == 'submit':
        params = kwargs['remainders']
        cmd = ' '.join(params)
        submit_without_sync(cmd=cmd, **kwargs)
        # assert len(kwargs['remainders']) == 1
        # submit_without_sync(cmd=kwargs['remainders'][0], **kwargs)
    elif task_type in ['a', 'abort']:
        p = MultiPhillyVC(**kwargs)
        for v in kwargs['remainders']:
            v = v.strip()
            p.search_job_id_and_abort(v)
    elif task_type == 'blame':
        blame(**kwargs)
    elif task_type == 'ssh':
        assert len(kwargs['remainders']) == 1
        app_id = kwargs['remainders'][0]
        p = create_multi_philly_client(**kwargs)
        client, app_info = p.search_job_id(app_id)
        cmd_run(app_info['ssh'].split(' '),
                stdin=None,
                shell=True)
    elif task_type == 'sync':
        p = create_philly_client(**kwargs)
        p.sync_code('')
    elif task_type == 'update_config':
        p = create_philly_client(**kwargs)
        p.update_config()
    elif task_type == 'init':
        ensure_init_config_files()
        p = create_philly_client(**kwargs)
        p.update_config()
        p.sync_code('')
    elif task_type == 'resubmit':
        partial_ids = kwargs['remainders']
        del kwargs['remainders']
        if len(partial_ids) == 0:
            multi_client = create_multi_philly_client(**kwargs)
            multi_client.auto_resubmit()
        else:
            for partial_id in partial_ids:
                abort_submit(partial_id, **kwargs)
    elif task_type == 'summary':
        m = create_multi_philly_client()
        m.print_summary()
    elif task_type in ['i', 'inject']:
        if not kwargs.get('with_log'):
            kwargs['with_log'] = True
        m = create_multi_philly_client(**kwargs)
        m.inject()
    else:
        assert 'Unknown {}'.format(task_type)

if __name__ == '__main__':
    os.environ['LD_LIBRARY_PATH'] = '/opt/intel/mkl/lib/intel64'
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)
