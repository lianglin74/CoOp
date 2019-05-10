#!/usr/bin/python
import os.path as op
from qd.qd_common import retry_agent, cmd_run
from qd.qd_common import ensure_directory
from qd.qd_common import load_from_yaml_file
from qd.qd_common import load_from_yaml_str
from qd.qd_common import url_to_str
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

from qd.qd_common import dump_to_yaml_str
from qd.qd_common import init_logging
from qd.qd_common import list_to_nested_dict


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
    assert len(dest_dir) > 0 and dest_dir[0] != '/' and dest_dir[0] != '\\'
    account = 'vig'
    dest_url = op.join('https://{}.blob.core.windows.net/data',
            account,
            dest_dir,
            op.basename(src_dir))

    c = create_cloud_storage(account)
    dest_url, _ = c.az_upload2(src_dir, op.join(dest_dir, op.basename(src_dir)))
    if kwargs.get('copy_to_hdfs', True):
        env = {'AZURE_STORAGE_ACCESS_KEY': c.account_key}

        sub_cmd = ['-cp', '-r', dest_url, dest_dir, 3]
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
            cmd.append('{}{}'.format(folder_prefix, dest_dir))
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
    param.update(kwargs)
    if not param.get('password'):
        param['password'] = os.environ['PHILLY_PASSWORD']
    return PhillyVC(**param)

def decode_extra_param(extraParam):
    re_result = re.match('.*python scripts/.*\.py -bp (.*)', extraParam)
    if re_result and len(re_result.groups()) == 1:
        ps = load_from_yaml_str(base64.b64decode(re_result.groups()[0]))
        return ps

def get_http_prefix(vc_type):
    if vc_type == 'ap':
        return 'http://phillyonap'
    elif vc_type == 'azure':
        return 'https://philly'
    else:
        return 'https://philly'

def attach_log_parsing_result(job_info):
    logs = job_info['latest_log']
    all_log = logs.split('\n')
    for log in reversed(all_log):
        pattern = '.*solver\.cpp:[0-9]*] Iteration [0-9]* \(.* iter\/s, ([0-9\.]*s\/100) iters, left: ([0-9\.]*h)\), loss = [0-9\.]*'
        result = re.match(pattern, log)
        if result and result.groups():
            job_info['speed'], job_info['left'] = result.groups()
            del job_info['latest_log']
            return

        # 2019-05-10T04:29:11.340Z: [1,0]<stdout>:2019-05-10 04:29:11,341.341 trainer.py:111   do_train(): eta: 3 days, 17:56:14  iter: 440  loss: 2.3267 (2.3407)  loss_retina_cls: 1.2245 (1.2373)  loss_retina_reg: 1.0626 (1.1034)  time: 0.6675 (0.9005)  data: 0.0033 (0.2286)  lr: 0.002300  max mem: 4049
        pattern = '.*: eta: (.*) iter.*'
        result = re.match(pattern, log)
        if result and result.groups():
            job_info['speed'] = None
            job_info['left'] = result.groups()[0]
            del job_info['latest_log']
            return
    del job_info['latest_log']

def print_table(a_to_bs, all_key=None):
    if len(a_to_bs) == 0:
        logging.info('no rows')
        return
    if not all_key:
        all_key = []
        for a_to_b in a_to_bs:
            all_key.extend(a_to_b.keys())
        all_key = list(set(all_key))
    all_width = [max([len(str(a_to_b.get(k, ''))) for a_to_b in a_to_bs] +
        [len(k)]) for k in all_key]
    row_format = '  '.join(['{{:{}}}'.format(w) for w in all_width])

    all_line = []
    line = row_format.format(*all_key)
    all_line.append(line.strip())
    for a_to_b in a_to_bs:
        line = row_format.format(*[str(a_to_b.get(k, '')) for k in all_key])
        all_line.append(line)
    logging.info('info\n{}'.format('\n'.join(all_line)))

class PhillyVC(object):
    def __init__(self, vc, cluster, user_name=None,
            use_blob_as_input=True, **kwargs):
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
        self.use_blob_as_input = use_blob_as_input
        self.user_name = user_name

        self.password = kwargs.get('password')

        self.src_config_path = 'src/qd/philly/config.py'
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
        self.query_with_log = kwargs.get('with_log')

    def get_data_folder_in_blob(self):
        assert self.config_param['data_folder'].startswith(self.blob_mount_point)
        result = self.config_param['data_folder'][len(self.blob_mount_point): ]
        if result.startswith('/'):
            result = result[1:]
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

    def search_job_id(self, partial_id):
        all_job_info = self.query_all_job(my_own=True)
        all_job_info = [j for j in all_job_info if
                j['appID'].endswith(partial_id)]
        assert len(all_job_info) == 1, ([j['appID'] for j in all_job_info], partial_id)
        return all_job_info[0], all_job_info[0]['appID']

    def abort(self, application_id):
        cmd = 'https://philly/api/abort?clusterId={}&jobId={}'.format(
                self.cluster, application_id)
        self.philly_rest_api(cmd)

    def query_all_job(self, my_own=True):
        cmd="{}/api/list?".format(self.get_http_prefix())
        param = ['clusterId={}'.format(self.cluster),
                'vcId={}'.format(self.vc),
                ]
        if my_own:
            param.append('userName={}'.format(self.user_name))
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

        return all_job_info

    def update_config(self):
        if not self.use_blob_as_input:
            self.upload_file(self.src_config_path, self.dest_config_folder)
        else:
            c = create_cloud_storage('vig')
            c.az_upload2(self.src_config_path,
                    op.join(self.dest_config_folder,
                        op.basename(self.src_config_path)))

    def attach_log(self, job_info):
        job_id = job_info['appID']
        job_info['latest_log'] = self.philly_job_log(job_id)

    def attach_gpu_utility(self, all_job_info):
        for job_info in all_job_info:
            if not job_info['ssh']:
                continue
            try:
                gpu_result = cmd_run('{} nvidia-smi'.format(
                    job_info['ssh']).split(' '), shell=True, return_output=True)
                from gpu_util import parse_gpu_usage_dict
                gpu_info = parse_gpu_usage_dict(gpu_result)
                import numpy as np
                gpu_info = {t: np.mean([g[t] for g in gpu_info]) for t in ['mem_used',
                        'mem_total', 'gpu_util']}
                for k in gpu_info:
                    job_info[k] = gpu_info[k]
            except Exception as e:
                logging.info(str(e))

    def query(self):
        all_job_info = self.query_all_job()
        all_job_info = [j for j in all_job_info if j['status'] != 'Pass' and j['status'] !=
            'Failed' and j['status'] != 'Killed']
        all_key = ['appID-s', 'elapsedTime', 'retries', 'preempts']
        if self.query_with_gpu:
            self.attach_gpu_utility(all_job_info)
            all_key.append('mem_used')
            all_key.append('gpu_util')
        self.attach_meta(all_job_info)
        for job_info in all_job_info:
            job_info['cluster'] = self.cluster
            job_info['vc'] = self.vc
            job_info['vc_type'] = self.vc_type
        if self.query_with_log:
            for job_info in all_job_info:
                self.attach_log(job_info)
                attach_log_parsing_result(job_info)
            all_key.extend(['speed', 'left'])
        for j in all_job_info:
            j['appID-s'] = j['appID'][-4:]
        keys = ['data', 'net', 'max_iters', 'max_iter']
        for job_info in all_job_info:
            for k in keys:
                job_info[k] = job_info['meta'].get('param',
                        {}).get(k)
        all_key.extend(keys)

        # remove the key if all value is None
        all_key = [k for k in all_key if any(k in j and j[k] is not None for j in all_job_info)]

        # find the keys whose values are the same
        def all_equal(x):
            assert len(x) > 0
            return all(y == x[0] for y in x[1:])

        if len(all_job_info) > 1:
            all_key = [k for k in all_key if not all_equal([j.get(k) for j in all_job_info])]

        print_table(all_job_info, all_key=all_key)
        return all_job_info

    def attach_meta(self, all_job_info):
        all_meta = self.query_meta_data([j['appID'] for j in all_job_info])
        self.parse_meta_data(all_meta)
        for job_info, meta in zip(all_job_info, all_meta):
            job_info['meta'] = meta

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
                ps = load_from_yaml_str(base64.b64decode(re_result.groups()[0]))
                command_parse = re.match('python .* -bp (.*)', ps['command'])
                if command_parse:
                    param = load_from_yaml_str(base64.b64decode(command_parse.groups()[0]))
                    meta['param'] = param.get('param', {})

    def submit2(self, param):
        '''
        param should be a list of string or integer or something
        '''
        if self.random_id is None:
            random_id = int(random.random() * 10000)
            self.sync_code(random_id)
            self.random_id = random_id

        random_qd = self.get_random_qd()

        extraParam = '{} python scripts/run.py {}'.format(
                random_qd, ' '.join(map(str, param)))

        return self.submit_without_sync(extraParam)

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
        if self.use_blob_as_input:
            config_file = op.join(self.blob_mount_point,
                    self.dest_config_file)
        else:
            config_file = "/hdfs/{}/{}/{}".format(self.vc,
                self.dest_config_folder, op.basename(self.src_config_path))
        logging.info('cmd:\n{} d d d {}'.format(config_file, extraParam))
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
            "CustomMPIArgs": 'env CUDA_CACHE_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO OMP_NUM_THREADS=2',
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

        blob_account  = 'vig'
        cloud_blob = create_cloud_storage(blob_account)
        blob_container = cloud_blob.container_name
        blob_key = cloud_blob.account_key

        data['volumes'] = {'blob': {'type': 'blobfuseVolume',
            'storageAccount': blob_account,
            'containerName': blob_container,
            'path': self.blob_mount_point,
            }}
        data['credentials'] = {'storageAccounts': {blob_account: {
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

        if not self.dry_run:
            result_str = cmd_run(cmd, return_output=True)
            return result_str

    def submit(self, extraParam, **submit_param):
        self.submit_without_sync(extraParam, **submit_param)

    def submit_without_sync(self, extraParam):
        '''
        use submit() because of bad naming here.
        '''
        result = self.philly_submit_v2(str(self.random_id), self.num_gpu,
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

    def track(self, job_id, random_qd=''):
        i = 0
        while True:
            status = self.track_job_once(job_id)
            logging.info('track {}-th times ({}, {})'.format(i,
                job_id,
                random_qd))
            key = 'status'
            if key in status:
                logging.info('status = {}'.format(status[key]))
                #logging.info('{}'.format(pformat(status)))
            else:
                logging.info('no satus: {}'.format(pformat(status)))
            i = i + 1
            time.sleep(60)

    def track_job_once(self, job_id):
        result = self.philly_job_log(job_id)
        logging.info(result)

        status = json.loads(self.philly_job_status(job_id))

        meta = decode_extra_param(json.loads(self.philly_job_meta(job_id))['cmd'])
        logging.info(pformat(meta))

        logging.info('satus = {}'.format(status['status']))
        self.get_ssh_command(status)
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
            logging.info(cmd)
            return cmd

    def get_http_prefix(self):
        if self.vc_type == 'ap':
            return 'http://phillyonap'
        elif self.vc_type == 'azure':
            return 'https://philly'
        else:
            return 'https://philly'

    def philly_job_log(self, job_id, logRev='latest'):
        cmd_str = \
            '{}/api/log?clusterId={}&vcId={}&jobId={}&jobType=cust&logType=stdout&logRev={}&content=partial'.format(
                self.get_http_prefix(),
                self.cluster,
                self.vc,
                job_id, logRev)
        result = self.philly_rest_api(cmd_str)

        if 'WARNING' in result and 'too big for preview' in result:
            # wget https://storage.sc2.philly.selfhost.corp.microsoft.com/input/sys/jobs/application_1544809666047_4657/stdout/1/stdout.txt
            result = url_to_str('https://storage.{}.philly.selfhost.corp.microsoft.com/input/sys/jobs/{}/stdout/1/stdout.txt'.format(
                self.cluster, job_id))

        logging.info(result)
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

    def upload_file(self, file_from, file_target):
        if self.cluster in ['sc2', 'wu1']:
            blob = False
            philly_upload_dir(file_from, file_target, self.vc,
                    self.cluster, blob=blob)
        else:
            philly_upload(file_from, file_target, self.vc, self.cluster)

    def philly_rest_api(self, CMD):
        user_name, password = self.user_name, self.password
        cmd = ['curl', '-k', '--ntlm', '--user',
                '"redmond\\{}:{}"'.format(user_name, password),
                '"{}"'.format(CMD)]
        result_str = cmd_run(cmd, shell=True, return_output=True)
        return result_str

def convert_to_philly_extra_command(param, script='scripts/tools.py'):
    logging.info(pformat(param))
    x = copy.deepcopy(param)
    from qd.qd_common import dump_to_yaml_str
    result = "python {} -bp {}".format(
            script,
            base64.b64encode(dump_to_yaml_str(x)).decode())
    return result

def submit_without_sync(cmd, **kwargs):
    isDebug = False
    all_extra_param = []
    if cmd == 'gc':
        params = {'type': 'del_intermediate_models'}
        extra_param = convert_to_philly_extra_command(params,
                script='garbage_collector')
        all_extra_param.append(extra_param)
    elif cmd == 'ssh':
        isDebug = True
        extra_param = 'ls'
        all_extra_param.append(extra_param)
    else:
        all_extra_param.append(cmd)
    kwargs.update({'isDebug': isDebug})
    logging.info(all_extra_param)
    p = create_philly_client(**kwargs)
    if kwargs.get('real_submit', True):
        list(map(lambda extra_param: p.submit_without_sync(extra_param),
                all_extra_param))

def tracking(app_id, **kwargs):
    p = create_philly_client(**kwargs)
    _, app_id = p.search_job_id(app_id)
    p.track_job_once(app_id)

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
    for status, username_to_gpus in status_to_username_to_gpus.iteritems():
        username_num_jobs_num_gpus = []
        for username, all_gpus in username_to_gpus.iteritems():
            username_num_jobs_num_gpus.append((username,
                len(all_gpus), sum(all_gpus)))
        status_to_username_num_jobs_num_gpus[status] = \
            sorted(username_num_jobs_num_gpus, key=lambda x: -x[-1])
    logging.info('\n{}'.format(pformat(status_to_username_num_jobs_num_gpus)))

    status_to_queue_to_username_to_gpus = \
        list_to_nested_dict(all_username_status_gpus_queue, [1, 3, 0])
    status_to_queue_to_username_num_jobs_num_gpus = {}
    for status, queue_to_username_to_gpus in status_to_queue_to_username_to_gpus.iteritems():
        for queue, username_to_gpus in queue_to_username_to_gpus.iteritems():
            username_num_jobs_num_gpus = []
            for username, gpus in username_to_gpus.iteritems():
                username_num_jobs_num_gpus.append(
                        (username, len(gpus), sum(gpus)))
            x = sorted(username_num_jobs_num_gpus, key=lambda x: -x[-1])
            if status not in status_to_queue_to_username_num_jobs_num_gpus:
                status_to_queue_to_username_num_jobs_num_gpus[status] = {}
            status_to_queue_to_username_num_jobs_num_gpus[status][queue] = x
    logging.info('\n{}'.format(pformat(status_to_queue_to_username_num_jobs_num_gpus)))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Philly Interface')
    parser.add_argument('task_type',
            choices=['ssh', 'query', 'abort', 'submit', 'sync',
                'update_config', 'gc', 'blame', 'init'])
    parser.add_argument('-wl', '--with_log', default=False, action='store_true')
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-c', '--cluster', default=argparse.SUPPRESS, type=str)
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

def execute(task_type, **kwargs):
    if task_type == 'query':
        if len(kwargs.get('remainders', [])) > 0:
            assert len(kwargs['remainders']) == 1
            tracking(kwargs['remainders'][0], **kwargs)
        else:
            p = create_philly_client(**kwargs)
            p.query()
    elif task_type == 'submit':
        assert len(kwargs['remainders']) == 1
        submit_without_sync(cmd=kwargs['remainders'][0], **kwargs)
    elif task_type == 'abort':
        p = create_philly_client(**kwargs)
        for v in kwargs['remainders']:
            v = v.strip()
            _, job_id = p.search_job_id(v)
            p.abort(job_id)
    elif task_type == 'blame':
        blame(**kwargs)
    elif task_type == 'ssh':
        assert len(kwargs['remainders']) == 1
        app_id = kwargs['remainders'][0]
        p = create_philly_client(**kwargs)
        app_info, app_id = p.search_job_id(app_id)
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
    else:
        assert 'Unknown {}'.format(task_type)

if __name__ == '__main__':
    os.environ['LD_LIBRARY_PATH'] = '/opt/intel/mkl/lib/intel64'
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)
