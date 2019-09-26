import os.path as op
from pprint import pformat
import logging
import os

from qd.qd_common import load_from_yaml_file, dict_update_nested_dict
from qd.qd_common import cmd_run
from qd.qd_common import decode_general_cmd
from qd.qd_common import ensure_directory
from qd.qd_common import try_once
from qd.cloud_storage import create_cloud_storage
import copy


def create_aml_client(**kwargs):
    if 'cluster' in kwargs:
        cluster = kwargs['cluster']
        path = op.join('aux_data', 'aml', '{}.yaml'.format(cluster))
        last_config = './aux_data/aml/aml.yaml'
        if not op.exists(last_config) or op.islink(last_config):
            from qd.qd_common import try_delete
            try_delete(last_config)
            # next time, we don't have to specify teh parameter of cluster
            # since this one will be the default one.
            os.symlink(op.relpath(path, op.dirname(last_config)),
                    last_config)
    else:
        path = os.environ.get('AML_CONFIG_PATH', './aux_data/aml/aml.yaml')
        kwargs['cluster'] = op.splitext(op.basename(op.realpath(path)))[0]
    param = load_from_yaml_file(path)
    dict_update_nested_dict(param, kwargs)
    return AMLClient(**param)

def update_by_run_details(info, details):
    info['start_time'] = details.get('startTimeUtc')
    info['end_time'] = details.get('endTimeUtc')
    from dateutil.parser import parse
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if info['start_time'] is not None:
        d= (now - parse(info['start_time'])).total_seconds() / 3600
        info['elapsedTime'] = round(d, 2)
    if info['end_time'] is not None:
        d= (now - parse(info['end_time'])).total_seconds() / 3600
        info['elapsedFinished'] = round(d, 2)
    if len(details['runDefinition']['arguments']) > 0:
        cmd = details['runDefinition']['arguments'][-1]
        info['cmd'] = cmd
        info['cmd_param'] = decode_general_cmd(cmd)
    info['docker_image'] = details['runDefinition']['environment']['docker']['baseImage']
    info['num_gpu'] = details['runDefinition']['mpi']['processCountPerNode'] * \
            details['runDefinition']['nodeCount']
    info['logFiles'] = details['logFiles']

def parse_run_info(run, with_details=True,
        with_log=False, log_full=True):
    info = {}
    info['status'] = run.status
    info['appID'] = run.id
    info['appID-s'] = run.id[-5:]
    info['portal_url'] = run.get_portal_url()
    if not with_details:
        return info
    details = run.get_details()
    update_by_run_details(info, details)
    if with_log:
        all_log = download_run_logs(info, full=log_full)
        info['all_log_path'] = all_log
        master_logs = [l for l in all_log if l.endswith('70_driver_log_0.txt')]
        if len(master_logs) > 0:
            logging.info('parsing the log for {}'.format(info['appID']))
            info['master_log'] = master_logs[0]
            info['latest_log'] = cmd_run(['tail', '-n', '100',
                info['master_log']],
                return_output=True)
            from qd.qd_common import attach_log_parsing_result
            attach_log_parsing_result(info)

    info['data_store'] = list(set([v['dataStoreName'] for k, v in
        details['runDefinition']['dataReferences'].items()]))
    logging.info(info['portal_url'])

    param_keys = ['data', 'net', 'expid', 'full_expid']
    for k in param_keys:
        from qd.qd_common import dict_has_path
        if dict_has_path(info, 'cmd_param$param${}'.format(k)):
            v = info['cmd_param']['param'][k]
        else:
            v = None
        info[k] = v
    return info

def create_aml_run(experiment, run_id):
    from azureml.core.script_run import ScriptRun
    run_id = run_id.strip().strip('/')
    try:
        r = ScriptRun(experiment, run_id)
    except:
        all_run = experiment.get_runs()
        matched_runs = [r for r in all_run if r.id.endswith(run_id)]
        assert len(matched_runs) == 1, ', '.join([r.id for r in
            matched_runs])
        r = matched_runs[0]
    return r

def download_run_logs(run_info, full=True):
    # Do not use r.get_all_logs(), which could be stuck. Use
    # wget instead by calling download_run_logs
    if 'logFiles' not in run_info:
        return []
    all_log_file = []
    log_folder = os.environ.get('AML_LOG_FOLDER', 'assets')
    log_status_file = op.join(log_folder, run_info['appID'],
            'log_status.yaml')
    if not full:
        if op.isfile(log_status_file):
            log_status = load_from_yaml_file(log_status_file)
            if run_info['status'] == AMLClient.status_failed and \
                    log_status['status'] != AMLClient.status_failed:
                full = True
    for k, v in run_info['logFiles'].items():
        target_file = op.join(log_folder, run_info['appID'], k)
        from qd.qd_common import url_to_file_by_wget
        from qd.qd_common import url_to_file_by_curl
        if full:
            url_to_file_by_wget(v, target_file)
        else:
            url_to_file_by_curl(v, target_file, -1024 * 100 )
        all_log_file.append(target_file)

    log_status = {'status': run_info['status'],
                  'is_full': full}
    from qd.qd_common import write_to_yaml_file
    write_to_yaml_file(log_status, log_status_file)

    return all_log_file

def get_compute_status(compute_target):
    info = {}
    info['running_node_count'] = compute_target.status.node_state_counts.running_node_count
    info['preparing_node_count'] = compute_target.status.node_state_counts.preparing_node_count
    info['max_node_count'] = compute_target.scale_settings.maximum_node_count
    return info

class AMLClient(object):
    status_running = 'Running'
    status_queued = 'Queued'
    status_failed = 'Failed'
    status_completed = 'Completed'
    status_canceled = 'Canceled'
    def __init__(self, azure_blob_config_file, config_param, docker,
            datastore_name, aml_config, use_custom_docker,
            compute_target, source_directory, entry_script,
            with_log=True,
            env=None,
            **kwargs):
        self.kwargs = kwargs
        self.cluster = kwargs.get('cluster', 'aml')
        self.use_cli_auth = kwargs.get('use_cli_auth', False)
        self.aml_config = aml_config
        self.compute_target = compute_target
        # do not change the datastore_name unless the storage account
        # information is changed
        self.datastore_name = datastore_name
        self.source_directory = source_directory
        self.entry_script = entry_script
        # it does not matter what the name is
        if 'experiment_name' not in kwargs:
            from qd.qd_common import get_user_name
            self.experiment_name = get_user_name()
        else:
            self.experiment_name = kwargs['experiment_name']
        self.gpu_set_by_client = ('num_gpu' in kwargs)
        self.num_gpu = kwargs.get('num_gpu', 4)
        self.docker = docker

        self.config_param = {p: {'azure_blob_config_file': azure_blob_config_file,
                                 'path': v,
                                 'datastore_name': self.datastore_name} if
            type(v) is str else v for p, v in config_param.items()}

        self.blob_config_to_blob = {blob_config: create_cloud_storage(config_file=blob_config) for
                blob_config in set(v['azure_blob_config_file'] for v in self.config_param.values())}
        for p, v in self.config_param.items():
            if 'storage_type' not in v:
                v['storage_type'] = 'blob'

        for p, v in self.config_param.items():
            v['cloud_blob'] = self.blob_config_to_blob[v['azure_blob_config_file']]
            if 'datastore_name' not in v or v['datastore_name'] is None:
                if v['storage_type'] == 'blob':
                    v['datastore_name'] = '{}_{}'.format(
                            v['cloud_blob'].account_name,
                            v['cloud_blob'].container_name)
                elif v['storage_type'] == 'file':
                    v['datastore_name'] = 'file_{}_{}'.format(
                            v['cloud_blob'].account_name,
                            v['file_share'])
                else:
                    raise NotImplementedError()

        import copy
        self.env = {} if env is None else copy.deepcopy(env)

        self.with_log = with_log

        from azureml.core import Workspace
        if self.use_cli_auth:
            from azureml.core.authentication import AzureCliAuthentication
            cli_auth = AzureCliAuthentication()
            self.ws = Workspace.from_config(self.aml_config, auth=cli_auth)
        else:
            self.ws = Workspace.from_config(self.aml_config)
        self.compute_target = self.ws.compute_targets[self.compute_target]

        self.attach_data_store()
        self.attach_mount_point()

        self.use_custom_docker = use_custom_docker

        from azureml.core import Experiment
        self.experiment = Experiment(self.ws, name=self.experiment_name)

    def get_compute_status(self):
        compute_status = get_compute_status(self.compute_target)

        return compute_status

    def abort(self, run_id):
        run = create_aml_run(self.experiment, run_id)
        run.cancel()

    def query(self, run_id=None, by_status=None, max_runs=None):
        if run_id is None:
            # all_run is ordered by created time, latest first
            all_run = list(self.experiment.get_runs())
            if max_runs:
                all_run = all_run[: min(max_runs, len(all_run))]
            if by_status:
                assert by_status in [self.status_failed, self.status_queued,
                        self.status_running], "Unknown status: {}".format(by_status)
                all_run = [r for r in all_run if r.status == by_status]

            def check_with_details(r):
                valid_status = [self.status_running, self.status_queued]
                if by_status:
                    valid_status.append(by_status)
                return self.with_log and r.status in valid_status
            all_info = [parse_run_info(r, with_details=check_with_details(r),
                with_log=self.with_log, log_full=False) for r in all_run]
            for info in all_info:
                # used for injecting to db
                info['cluster'] = self.cluster
            from qd.qd_common import print_job_infos
            print_job_infos(all_info)
            return all_info
        else:
            r = create_aml_run(self.experiment, run_id)
            info = parse_run_info(r,
                    with_details=True,
                    with_log=self.with_log,
                    log_full=True)
            if 'master_log' in info:
                cmd_run(['tail', '-n', '100', info['master_log']])
            if info['status'] == self.status_failed:
                detect_aml_error_message(info['appID'])
            logging.info(pformat(info))
            return [info]

    def resubmit(self, partial_id):
        run = create_aml_run(self.experiment, partial_id)
        run_info = parse_run_info(run)
        if not self.gpu_set_by_client:
            num_gpu = run_info['num_gpu']
        else:
            num_gpu = self.num_gpu
        app_id = self.submit(run_info['cmd'], num_gpu=num_gpu)
        run.cancel()
        return app_id

    def attach_data_store(self):
        from azureml.core import Datastore
        for p, v in self.config_param.items():
            try:
                ds = Datastore.get(self.ws, v['datastore_name'])
            except:
                cloud_blob = v['cloud_blob']
                if v['storage_type'] == 'blob':
                    ds = Datastore.register_azure_blob_container(workspace=self.ws,
                                                                 datastore_name=v['datastore_name'],
                                                                 container_name=cloud_blob.container_name,
                                                                 account_name=cloud_blob.account_name,
                                                                 account_key=cloud_blob.account_key)
                else:
                    assert v['storage_type'] == 'file'
                    ds = Datastore.register_azure_file_share(workspace=self.ws,
                            datastore_name=v['datastore_name'],
                            file_share_name=v['file_share'],
                            account_name=cloud_blob.account_name,
                            account_key=cloud_blob.account_key)
            v['data_store'] = ds

    def attach_mount_point(self):
        for p, v in self.config_param.items():
            ds = v['data_store']
            v['mount_point'] = ds.path(v['path']).as_mount()

    def qd_data_exists(self, fname):
        return self.config_param['data_folder']['cloud_blob'].exists(
                op.join(self.config_param['data_folder']['path'],
                fname))

    def upload_qd_data(self, d):
        from qd.tsv_io import TSVDataset
        data_root = TSVDataset(d)._data_root
        self.config_param['data_folder']['cloud_blob'].az_sync(data_root,
                op.join(self.config_param['data_folder']['path'], d))

    def sync_full_expid_from_local(self, full_expid):
        cloud = self.config_param['output_folder']['cloud_blob']
        local_folder = op.join('output', full_expid)
        remote_folder = op.join(self.config_param['output_folder']['path'],
                    full_expid)
        cloud.az_sync(local_folder, remote_folder)

    def sync_full_expid_from_local_by_exist(self, full_expid):
        cloud = self.config_param['output_folder']['cloud_blob']
        local_folder = op.join('output', full_expid)
        remote_folder = op.join(self.config_param['output_folder']['path'],
                    full_expid)
        for d, dirs, files in os.walk(local_folder):
            for f in files:
                local_file = op.join(d, f)
                assert local_file.startswith(local_folder)
                fname = local_file[len(local_folder):]
                if fname.startswith('/'):
                    fname = fname[1:]
                remote_file = op.join(remote_folder, fname)
                if not cloud.exists(remote_file) and \
                        any(remote_file.endswith(suffix)
                            for suffix in ['.pt', '.yaml']):
                    cloud.az_upload2(local_file, remote_file)

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
        target_path = op.join(self.config_param['output_folder']['path'],
                path_splits[-3], path_splits[-2], path_splits[-1])
        cloud = self.config_param['output_folder']['cloud_blob']
        if not cloud.exists(target_path):
            cloud.az_upload2(model_file, target_path)

    def submit(self, cmd, num_gpu=None):
        script_params = {'--' + p: v['mount_point'] for p, v in self.config_param.items()}
        script_params['--command'] = cmd

        from azureml.train.estimator import Estimator
        from azureml.train.dnn import PyTorch
        import azureml
        env = azureml.core.runconfig.EnvironmentDefinition()
        env.docker.enabled = True
        env.docker.base_image = self.docker['image']
        env.docker.shm_size = '1024g'
        env.python.interpreter_path = '/opt/conda/bin/python'
        env.python.user_managed_dependencies = True

        env.environment_variables['NCCL_TREE_THRESHOLD'] = '0'
        env.environment_variables['NCCL_LL_THRESHOLD'] = '0'

        for k, v in self.env.items():
            assert type(v) is str
        env.environment_variables.update(self.env)

        #env.environment_variables['NCCL_IB_DISABLE'] = '0'
        #env.environment_variables['NCCL_SOCKET_IFNAME'] = 'eth0'

        #env.environment_variables['NCCL_IB_HCA'] = 'mlx4_0:1'
        #env.environment_variables['CUDA_CACHE_DISABLE'] = '1'
        #env.environment_variables['OMP_NUM_THREADS'] = '2'

        from azureml.core.runconfig import MpiConfiguration
        mpi_config = MpiConfiguration()
        if num_gpu is None:
            num_gpu = self.num_gpu
        if num_gpu <= 4:
            mpi_config.process_count_per_node = num_gpu
            node_count = 1
        else:
            assert (num_gpu % 4) == 0
            mpi_config.process_count_per_node = 4
            node_count = num_gpu // 4

        if self.use_custom_docker:
            estimator10 = Estimator(
                    source_directory=self.source_directory,
                    compute_target=self.compute_target,
                    script_params=script_params,
                    entry_script=self.entry_script,
                    environment_definition=env,
                    node_count=node_count,
                    distributed_training=mpi_config,
                    )
        else:
            estimator10 = PyTorch(
                    source_directory=self.source_directory,
                    compute_target=self.compute_target,
                    script_params=script_params,
                    entry_script=self.entry_script,
                    environment_definition=env,
                    node_count=node_count,
                    distributed_training=mpi_config,
                    )

        r = self.experiment.submit(estimator10)
        logging.info('job id = {}, cmd = \n{}'.format(r.id, cmd))
        return r.id

    def inject(self, run_id=None):
        all_info = self.query(run_id)
        from qd.db import update_cluster_job_db
        for info in all_info:
            if 'logFiles' in info:
                del info['logFiles']
        update_cluster_job_db(all_info)

    def sync_code(self, random_id, compile_in_docker=False):
        random_qd = 'quickdetection{}'.format(random_id)
        import os.path as op
        from qd.qd_common import get_user_name
        random_abs_qd = op.join('/tmp', get_user_name(),
                '{}.zip'.format(random_qd))
        if op.isfile(random_abs_qd):
            os.remove(random_abs_qd)
        logging.info('{}'.format(random_qd))
        from qd.qd_common import zip_qd
        # zip it
        zip_qd(random_abs_qd)

        if compile_in_docker:
            from qd.qd_common import compile_by_docker
            compile_by_docker(random_abs_qd, self.docker['image'],
                    random_abs_qd)

        rel_code_path = self.config_param['code_path']['path']
        # upload it
        self.config_param['code_path']['cloud_blob'].az_upload2(random_abs_qd, rel_code_path)

    def download_latest_qdoutput(self, full_expid):
        src_path = op.join(self.config_param['output_folder']['path'],
                full_expid)
        target_folder = op.join('output', full_expid)

        self.config_param['output_folder']['cloud_blob'].blob_download_qdoutput(
                src_path,
                target_folder)

def inject_to_tensorboard(info):
    log_folder = 'output/tensorboard/aml'
    ensure_directory(log_folder)
    from torch.utils.tensorboard import SummaryWriter
    wt = SummaryWriter(log_dir=log_folder)
    for k, v in info.items():
        wt.add_scalar(tag=k, scalar_value=v)

@try_once
def detect_aml_error_message(app_id):
    import glob
    folders = glob.glob(op.join('./assets', '*{}'.format(app_id),
        'azureml-logs'))
    assert len(folders) == 1, 'not unique: {}'.format(', '.join(folders))
    folder = folders[0]
    has_nvidia_smi = False
    from qd.qd_common import read_to_buffer
    for log_file in glob.glob(op.join(folder, '70_driver_log*')):
        all_line = read_to_buffer(log_file).decode().split('\n')
        for i, line in enumerate(all_line):
            if 'Error' in line and \
                'TrackUserError:context_managers.TrackUserError' not in line and \
                'WARNING: Retrying' not in line:
                #start = max(0, i - 10)
                start = max(0, i)
                end = min(i + 1, len(all_line))
                logging.info(log_file)
                logging.info('\n'.join(all_line[start: end]))
        # check how many gpus by nvidia-smi
        import re
        num_gpu = len([line for line in all_line if re.match('.*N/A.*Default', line) is
            not None])
        has_nvidia_smi =  any(('nvidia-smi' in line for line in all_line))
        if has_nvidia_smi and num_gpu != 4:
            logging.info(log_file)
            logging.info(num_gpu)

def monitor():
    from qd.db import create_annotation_db
    c = create_annotation_db()
    cluster_to_client = {}
    dbjob_client_jobinfo = []
    for row in c.iter_general('ongoingjob'):
        if 'job_id' in row:
            appID = row['job_id']
        else:
            appID = row['appID']
        cluster = row['cluster']
        if cluster not in cluster_to_client:
            client = create_aml_client(cluster=cluster,
                    with_log=False)
            cluster_to_client[cluster] = client
        client = cluster_to_client[cluster]
        job_info = client.query(appID)[0]
        dbjob_client_jobinfo.append((row, client, job_info))

    # update status
    for row, client, job_info in dbjob_client_jobinfo:
        c.update_many('ongoingjob', {'_id': row['_id']},
                {'$set': {'status': job_info['status'],
                          'portal_url': job_info['portal_url']}})

    for row, client, job_info in dbjob_client_jobinfo:
        if job_info['status'] in [client.status_completed,
                client.status_canceled]:
            logging.info('removing _id = {}'.format(row['_id']))
            c.delete_many('ongoingjob', _id=row['_id'])
        elif job_info['status'] == client.status_failed:
            logging.info('resubmitting {}'.format(job_info['appID']))
            new_appID = client.resubmit(job_info['appID'])
            retried = row.get('retry', 0) + 1
            history = row.get('history', [])
            history.append(job_info['appID'])
            c.update_many('ongoingjob', {'_id': row['_id']},
                          {'$set': {'appID': new_appID,
                                    'status': client.status_queued,
                                    'history': history,
                                    'retried': retried},
                           '$unset': {'portal_url': ''}})

# some user might not use the mongodb and we will not crash here
@try_once
def search_partial_id_from_db(partial_id):
    from qd.db import create_annotation_db
    c = create_annotation_db()
    found = list(c.iter_phillyjob(appID={'$regex': '.*{}'.format(partial_id)}))
    if len(found) == 0 or len(found) > 1:
        return None
    else:
        return {
                'cluster': found[0]['cluster'],
                'appID': found[0]['appID'],
                }

class MultiAMLClient(object):
    def __init__(self, **kwargs):
        self.kwargs = copy.deepcopy(kwargs)
        self.cluster_to_client = {}
        self._default_client = None

    def default_client(self):
        if self._default_client is None:
            self._default_client = create_aml_client(**self.kwargs)

    def search_partial_id(self, partial_id):
        search_result = search_partial_id_from_db(partial_id)
        if search_result is None:
            client = self.default_client
            appID = partial_id
        else:
            cluster, appID = search_result['cluster'], search_result['appID']
            if cluster in self.cluster_to_client:
                client = self.cluster_to_client[cluster]
            else:
                param = copy.deepcopy(self.kwargs)
                param['cluster'] = cluster
                client = create_aml_client(**param)
                self.cluster_to_client[cluster] = client
        return client, appID

    def abort(self, partial_id):
        client, partial_id = self.search_partial_id(partial_id)
        client.abort(partial_id)

    def query(self, partial_id, **kwargs):
        client, partial_id = self.search_partial_id(partial_id)
        client.query(partial_id, **kwargs)


def execute(task_type, **kwargs):
    if task_type in ['q', 'query']:
        if len(kwargs.get('remainders', [])) > 0:
            assert len(kwargs['remainders']) == 1
            c = MultiAMLClient(**kwargs)
            c.query(partial_id=kwargs['remainders'][0])
        else:
            c = create_aml_client(**kwargs)
            c.query(max_runs=kwargs.get('max', None))
    elif task_type in ['f', 'failed', 'qf']:
        c = create_aml_client(**kwargs)
        c.query(by_status=AMLClient.status_failed)
    elif task_type in ['qq']:
        c = create_aml_client(**kwargs)
        c.query(by_status=AMLClient.status_queued)
    elif task_type in ['qr']:
        c = create_aml_client(**kwargs)
        c.query(by_status=AMLClient.status_running)
    elif task_type in ['init', 'initc']:
        c = create_aml_client(**kwargs)
        c.sync_code('')
        if task_type=='initc':
            # in this case, we first upload the raw code to unblock the job
            # submission. then we upload the compiled version to reduce the
            # overhead in aml running
            c.sync_code('', compile_in_docker=True)
    elif task_type == 'submit':
        c = create_aml_client(**kwargs)
        params = kwargs['remainders']
        cmd = ' '.join(params)
        c.submit(cmd)
    elif task_type in ['a', 'abort']:
        c = MultiAMLClient(**kwargs)
        for v in kwargs['remainders']:
            v = v.strip('/')
            c.abort(v)
    elif task_type in ['download_qdoutput', 'd']:
        c = create_aml_client(**kwargs)
        for full_expid in kwargs['remainders']:
            c.download_latest_qdoutput(full_expid)
    elif task_type == 'blame':
        raise NotImplementedError()
        blame(**kwargs)
    elif task_type == 'resubmit':
        partial_ids = kwargs['remainders']
        del kwargs['remainders']
        client = create_aml_client(**kwargs)
        if 'resubmit_to' in kwargs and \
                kwargs['resubmit_to'] != kwargs.get('cluster'):
            resubmit_to = kwargs['resubmit_to']
            if 'cluster' in kwargs:
                del kwargs['cluster']
            kwargs['cluster'] = resubmit_to
            dest_client = create_aml_client(**kwargs)
        else:
            dest_client = client
        for partial_id in partial_ids:
            run_info = client.query(partial_id)[0]
            dest_client.submit(run_info['cmd'],
                    num_gpu=run_info['num_gpu'])
            client.abort(run_info['appID'])
    elif task_type in ['s', 'summary']:
        m = create_aml_client(**kwargs)
        info = m.get_compute_status()
        logging.info(pformat(info))
        inject_to_tensorboard(info)
        from qd.db import inject_cluster_summary
        info['cluster'] = 'aml'
        inject_cluster_summary(info)
    elif task_type in ['monitor']:
        monitor()
    elif task_type in ['i', 'inject']:
        run_ids = kwargs.get('remainders', [])
        m = create_aml_client(**kwargs)
        if len(run_ids) == 0:
            m.inject()
        else:
            for run_id in run_ids:
                m.inject(run_id)
    elif task_type in ['parse']:
        detect_aml_error_message(kwargs.get('remainders')[0])
    else:
        assert 'Unknown {}'.format(task_type)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Philly Interface')
    parser.add_argument('task_type',
            choices=['ssh', 'q', 'query', 'f', 'failed', 'a', 'abort', 'submit',
                'qf', # query failed jobs
                'qq', # query queued jobs
                'qr', # query running jobs
                'd', 'download_qdoutput',
                'monitor',
                'parse',
                'init',
                'initc', # init with compile
                'blame', 'resubmit',
                's', 'summary', 'i', 'inject'])
    parser.add_argument('-wl', dest='with_log', default=True, action='store_true')
    parser.add_argument('-no-wl', dest='with_log',
            action='store_false')
    parser.add_argument('-c', '--cluster', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-rt', '--resubmit_to', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-n', '--num_gpu', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--max', default=None, type=int)
    #parser.add_argument('-wg', '--with_gpu', default=True, action='store_true')
    parser.add_argument('-no-wg', '--with_gpu', default=True, action='store_false')
    #parser.add_argument('-m', '--with_meta', default=True, action='store_true')
    parser.add_argument('-no-m', '--with_meta', default=True, action='store_false')

    parser.add_argument('-no-s', '--real_submit', default=True, action='store_false')

    parser.add_argument('remainders', nargs=argparse.REMAINDER,
            type=str)
    return parser.parse_args()

if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)
