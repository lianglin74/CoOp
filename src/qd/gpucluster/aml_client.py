import os.path as op
from pprint import pformat
import logging
import os

from deprecated import deprecated
from qd.qd_common import load_from_yaml_file, dict_update_nested_dict
from qd.qd_common import cmd_run
from qd.qd_common import decode_general_cmd
from qd.qd_common import ensure_directory
from qd.qd_common import try_once
from qd.qd_common import read_to_buffer
from qd.qd_common import get_file_size, get_url_fsize
from qd.qd_common import concat_files, try_delete
from qd.qd_common import dict_has_path
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
    if dict_has_path(details, 'runDefinition$arguments') and len(details['runDefinition']['arguments']) > 0:
        xs = [i for i, k in enumerate(details['runDefinition']['arguments']) if k == '--command']
        assert len(xs) == 1
        i = xs[0]
        cmd = details['runDefinition']['arguments'][i + 1]
        info['cmd'] = cmd
        info['cmd_param'] = decode_general_cmd(cmd)
    if dict_has_path(details, 'runDefinition$environment$docker$baseImage'):
        info['docker_image'] = details['runDefinition']['environment']['docker']['baseImage']
    if dict_has_path(details, 'runDefinition$mpi$processCountPerNode') and \
            dict_has_path(details, 'runDefinition$nodeCount'):
        info['num_gpu'] = details['runDefinition']['mpi']['processCountPerNode'] * \
                details['runDefinition']['nodeCount']
    info['logFiles'] = details['logFiles']

def create_ssh_command_from_aks_ssh_tag(aks_ssh):
    #ssh://jianfw@az-wus2-v100-32gb-infra01.westus2.cloudapp.azure.com:31951
    url, port = aks_ssh.split('//')[1].split(':')
    cmd = ['ssh', '-p', port,
           #'-o', 'StrictHostKeyChecking no',
           url]
    return cmd

def parse_run_info(run, with_details=True,
        with_log=False, log_full=True):
    info = {}
    info['status'] = run.status
    info['appID'] = run.id
    info['appID-s'] = run.id[-5:]
    info['portal_url'] = run.get_portal_url()
    if run.status == 'Queued' and \
            run.tags.get('amlk8s status') == 'running':
        # there is a bug in AML/DLTS
        info['status'] = AMLClient.status_running
    if run.status == AMLClient.status_canceled:
        if run.tags.get('amlk8s status') in ['killing', 'failed']:
            info['status'] = AMLClient.status_failed
        else:
            if 'amlk8s status' in run.tags and \
                    run.tags.get('amlk8s status') != 'Terminated':
                logging.info('unknown status {}'.format(
                    run.tags.get('amlk8s status')))
    if run.tags.get('ssh'):
        # aks cluster has this field
        info['ssh'] = create_ssh_command_from_aks_ssh_tag(run.tags['ssh'])
    if not with_details:
        return info
    details = run.get_details()
    update_by_run_details(info, details)
    if with_log:
        all_log = download_run_logs(info, full=log_full)
        if all_log is None:
            all_log = []
        info['all_log_path'] = all_log
        master_logs = [l for l in all_log if
                       l.endswith('70_driver_log_0.txt')
                       or l.endswith('00_stdout.txt')
                       or l.endswith('ps-0_stdout.txt')
                       ]
        if len(master_logs) > 0:
            logging.info('parsing the log for {}'.format(info['appID']))
            info['master_log'] = master_logs[0]
            if op.isfile(info['master_log']):
                x = cmd_run(['tail', '-c', '2097152',
                    info['master_log']],
                    return_output=True)
            else:
                x =''
            info['latest_log'] = x
            from qd.qd_common import attach_log_parsing_result
            attach_log_parsing_result(info)

    if dict_has_path(details, 'runDefinition$dataReferences'):
        info['data_store'] = list(set([v['dataStoreName'] for k, v in
            details['runDefinition']['dataReferences'].items()]))
    logging.info(info['portal_url'])

    param_keys = ['data', 'net', 'expid', 'full_expid']
    from qd.qd_common import get_all_path
    all_path = get_all_path(info)
    for k in param_keys:
        ps = [p for p in all_path if p.endswith('${}'.format(k))]
        if len(ps) == 1:
            from qd.qd_common import dict_get_path_value
            v = dict_get_path_value(info, ps[0])
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
        # sometimes, AML will return two runs with the same ID. it is quite
        # strange. Here, we deduplicate the runs by the id
        id_runs = [(r.id, r) for r in matched_runs]
        from qd.qd_common import list_to_dict
        id_to_runs = list_to_dict(id_runs, 0)
        matched_runs = [runs[0] for _, runs in id_to_runs.items()]
        assert len(matched_runs) == 1, ', '.join([r.id for r in
            matched_runs])
        r = matched_runs[0]
    return r

@try_once
def download_run_logs(run_info, full=True):
    # Do not use r.get_all_logs(), which could be stuck. Use
    # wget instead by calling download_run_logs
    if 'logFiles' not in run_info:
        return []
    all_log_file = []
    log_folder = get_log_folder()
    log_status_file = op.join(log_folder, run_info['appID'],
            'log_status.yaml')
    for k, v in run_info['logFiles'].items():
        target_file = op.join(log_folder, run_info['appID'], k)
        from qd.qd_common import url_to_file_by_curl
        try:
            url_fsize = get_url_fsize(v)
        except:
            all_log_file.append(target_file)
            continue
        if url_fsize == 0:
            continue
        if op.isfile(target_file):
            start_size = get_file_size(target_file)
            if start_size < url_fsize:
                extra_target_file = target_file + '.extra'
                if op.isfile(extra_target_file):
                    try_delete(extra_target_file)
                if full:
                    end_size = None
                else:
                    end_size = min(url_fsize, start_size + 1024 * 1024 * 10)
                try:
                    url_to_file_by_curl(v, extra_target_file, start_size,
                                        end_size)
                except:
                    pass
                if op.isfile(extra_target_file):
                    concat_files([target_file, extra_target_file], target_file)
                    try_delete(extra_target_file)
        else:
            if full:
                end_size = None
            else:
                end_size = min(url_fsize, 1024 * 100)
            try:
                url_to_file_by_curl(v, target_file, 0, end_size)
            except:
                pass
        all_log_file.append(target_file)

    log_status = {'status': run_info['status'],
                  'is_full': full}
    from qd.qd_common import write_to_yaml_file
    write_to_yaml_file(log_status, log_status_file)

    return all_log_file

def iter_active_runs(compute_target):
    from azureml._restclient.workspace_client import WorkspaceClient
    workspace_client = WorkspaceClient(compute_target.workspace.service_context)
    return workspace_client.get_runs_by_compute(compute_target.name)

def get_compute_status(compute_target, gpu_per_node=4):
    info = {}

    if hasattr(compute_target, 'status'):
        # aks-compute
        info['running_node_count'] = compute_target.status.node_state_counts.running_node_count
        info['preparing_node_count'] = compute_target.status.node_state_counts.preparing_node_count
        info['unusable_node_count'] = compute_target.status.node_state_counts.unusable_node_count
        info['max_node_count'] = compute_target.scale_settings.maximum_node_count
        info['total_gpu'] = info['max_node_count'] * gpu_per_node
        info['total_free_gpu'] = (info['max_node_count'] -
            info['unusable_node_count'] - info['running_node_count'] -
            info['preparing_node_count']) * gpu_per_node

    all_job_status = [j.status for j in iter_active_runs(compute_target)]
    info['num_running_job'] = len([j for j in all_job_status if j == 'Running'])
    info['num_queued_jobs'] = len([j for j in all_job_status if j ==
                                   AMLClient.status_queued])

    return info

def log_downloaded(appID):
    fname = op.join(get_log_folder(), appID, 'log_status.yaml')
    if not op.isfile(fname):
        return False
    status = load_from_yaml_file(fname)
    if not status['is_full']:
        return False
    return True

def print_topk_long_run_jobs(ws, topk):
    all_expname_job = []
    for name, exp in ws.experiments.items():
        running_jobs = [r for r in exp.get_runs() if r.status == AMLClient.status_running]
        running_job_infos = [parse_run_info(r) for r in running_jobs]
        all_expname_job.extend([(name, r) for r in running_job_infos])
    all_expname_job = sorted(all_expname_job, key=lambda x:-x[1]['elapsedTime'])
    logging.info(pformat(all_expname_job[:topk]))

def get_root_folder_in_curr_dir(fname):
    curr = op.abspath(os.curdir)
    fname = op.abspath(fname)
    assert fname.startswith(curr)
    fname = op.relpath(fname, curr)
    p = fname
    while True:
        d = op.dirname(p)
        if d == '':
            break
        p = d
    return p

def clean_prefix(file_or_folder):
    file_or_folder = op.normpath(file_or_folder)
    if file_or_folder.startswith('./'):
        file_or_folder = file_or_folder[2:]
    elif file_or_folder.startswith('/'):
        file_or_folder = file_or_folder[1:]
    return file_or_folder

class AMLClient(object):
    status_running = 'Running'
    status_queued = 'Queued'
    status_failed = 'Failed'
    status_completed = 'Completed'
    status_canceled = 'Canceled'
    def __init__(self, azure_blob_config_file, config_param, docker,
                 datastore_name, aml_config, use_custom_docker,
                 compute_target, source_directory=None, entry_script='aml_server.py',
                 gpu_per_node=4,
                 with_log=True,
                 env=None,
                 multi_process=True,
                 aks_compute=False,
                 sleep_if_fail=False,
                 compile_args='',
                 preemption_allowed=False,
                 aks_compute_global_dispatch=False,
                 aks_compute_global_dispatch_arg=None,
                 **kwargs):
        self.kwargs = kwargs
        self.cluster = kwargs.get('cluster', 'aml')
        self.use_cli_auth = kwargs.get('use_cli_auth', False)
        self.aml_config = aml_config
        self.compute_target_name = compute_target
        self.gpu_per_node = gpu_per_node
        # do not change the datastore_name unless the storage account
        # information is changed
        self.datastore_name = datastore_name
        if source_directory is not None:
            logging.info('no need to specify the directory')
        #self.source_directory = source_directory
        self.entry_script = entry_script
        # it does not matter what the name is
        if 'experiment_name' not in kwargs:
            from qd.qd_common import get_user_name
            self.experiment_name = get_user_name()
        else:
            self.experiment_name = kwargs['experiment_name']
        self.gpu_set_by_client = ('num_gpu' in kwargs)
        self.num_gpu = kwargs.get('num_gpu', self.gpu_per_node)
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
                    if v.get('blob_cache_timeout') is not None:
                        v['datastore_name'] += '_t{}'.format(v['blob_cache_timeout'])
                elif v['storage_type'] == 'file':
                    v['datastore_name'] = 'file_{}_{}'.format(
                            v['cloud_blob'].account_name,
                            v['file_share'])
                else:
                    raise NotImplementedError()

        import copy
        self.env = {} if env is None else copy.deepcopy(env)

        self.with_log = with_log

        self._ws = None
        self._compute_target = None
        self.use_custom_docker = use_custom_docker
        self._experiment = None
        self.multi_process = multi_process
        self.aks_compute = aks_compute
        self.sleep_if_fail = sleep_if_fail
        self.compile_args = compile_args

        self.preemption_allowed = preemption_allowed
        self.aks_compute_global_dispatch = aks_compute_global_dispatch
        self.aks_compute_global_dispatch_arg = aks_compute_global_dispatch_arg or {}

    def __repr__(self):
        return self.compute_target_name

    @property
    def source_directory(self):
        return op.dirname(__file__)

    def get_data_blob_client(self):
        self.attach_data_store()
        return self.config_param['data_folder']['cloud_blob']

    @property
    def experiment(self):
        if self._experiment is None:
            from azureml.core import Experiment
            self._experiment = Experiment(self.ws, name=self.experiment_name)
        return self._experiment

    @property
    def ws(self):
        if self._ws is None:
            from azureml.core import Workspace
            for i in range(5):
                try:
                    if self.use_cli_auth:
                        from azureml.core.authentication import AzureCliAuthentication
                        cli_auth = AzureCliAuthentication()
                        self._ws = Workspace.from_config(self.aml_config, auth=cli_auth)
                    else:
                        self._ws = Workspace.from_config(self.aml_config)
                    break
                except:
                    if i == 4:
                        raise
        return self._ws

    @property
    def compute_target(self):
        if self._compute_target is None:
            if self.aks_compute:
                from azureml.contrib.core.compute.k8scompute import AksCompute
            self._compute_target = self.ws.compute_targets[self.compute_target_name]
        return self._compute_target

    def get_cluster_status(self):
        compute_status = get_compute_status(self.compute_target,
                gpu_per_node=self.gpu_per_node)

        return compute_status

    @deprecated('use get_cluster_status')
    def get_compute_status(self):
        return self.get_cluster_status()

    def abort(self, run_id):
        run = create_aml_run(self.experiment, run_id)
        run.cancel()

    def blame(self):
        print_topk_long_run_jobs(self.ws, 5)

    def list_nodes(self):
        node_list = self.compute_target.list_nodes()
        return node_list

    def ssh(self, run_id):
        if self.aks_compute:
            info = self.query_one(run_id=run_id)
            cmd = info['ssh']
        else:
            node_list = self.compute_target.list_nodes()
            import re
            if re.match('[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*', run_id):
                target_nodes = [n for n in node_list if n.get('privateIpAddress') ==
                                run_id]
            else:
                job_info = self.query(run_id, with_log=False, with_details=False)[0]
                target_nodes = [n for n in node_list if n.get('runId') == job_info['appID']]
            user_name = self.compute_target.admin_username
            #ssh user_name@public_ip -p port
            ssh_commands = ['ssh {}@{} -p {}'.format(user_name,
                                     n['publicIpAddress'],
                                     n['port']) for n in target_nodes]
            cmd = ssh_commands[0].split(' ')
        logging.info('all commands:\n{}'.format(pformat(cmd)))
        cmd_run(cmd, stdin=None, shell=True)

    def query_one(self, run_id,
                  with_details=False, with_log=False, log_full=False,
                  detect_error_if_failed=False):
        run = create_aml_run(self.experiment, run_id)
        return self.query_run(
            run, with_details=with_details,
            with_log=with_log, log_full=log_full,
            detect_error_if_failed=detect_error_if_failed)

    def query_run(self, run, with_details=False, with_log=False, log_full=False,
                  detect_error_if_failed=False):
        info = parse_run_info(run,
                              with_details=with_details,
                              with_log=with_log,
                              log_full=log_full,
                              )
        info['cluster'] = self.cluster
        if info['status'] == self.status_failed and detect_error_if_failed:
            messages = detect_aml_error_message(info['appID'])
            if messages is not None:
                info['result'] = ','.join(messages)
        return info

    def iter_query(self, run_id=None, by_status=None, max_runs=None,
                   with_log=False, with_details=False, log_full=False,
                   detect_error_if_failed=False):
        with_log = (with_log and with_details)
        log_full = (log_full and with_log and with_details)
        if run_id is None:
            # all_run is ordered by created time, latest first
            iter_run = self.experiment.get_runs()
        elif isinstance(run_id, list):
            iter_run = [create_aml_run(self.experiment, i) for i in run_id]
        else:
            iter_run = [create_aml_run(self.experiment, run_id)]

        for i, run in enumerate(iter_run):
            if max_runs and i >= max_runs:
                break
            if by_status and run.status != by_status:
                continue
            info = self.query_run(run,
                                  with_details=with_details,
                                  with_log=with_log,
                                  log_full=log_full,
                                  detect_error_if_failed=detect_error_if_failed
                                  )
            yield info

    @deprecated('use iter_query')
    def query(self, run_id=None, by_status=None, max_runs=None,
            with_log=False, with_details=None):
        if run_id is None:
            # all_run is ordered by created time, latest first
            iter_run = self.experiment.get_runs()
            if max_runs:
                all_run = [r for _, r in zip(range(max_runs), iter_run)]
            else:
                logging.info('enumerating all runs')
                all_run = list(iter_run)
            if by_status:
                assert by_status in [self.status_failed, self.status_queued,
                        self.status_running], "Unknown status: {}".format(by_status)
                all_run = [r for r in all_run if r.status == by_status]

            def check_with_details(r):
                valid_status = [self.status_running, self.status_queued]
                if by_status:
                    valid_status.append(by_status)
                if with_details is None:
                    wd = r.status in valid_status
                parse_info = {}
                parse_info = {'with_details': wd,
                              'with_log': self.with_log or with_log,
                              'log_full': False}
                return parse_info
            all_info = [parse_run_info(r, **check_with_details(r)) for r in all_run]
            for info in all_info:
                # used for injecting to db
                info['cluster'] = self.cluster
            from qd.qd_common import print_job_infos
            print_job_infos(all_info)
            return all_info
        else:
            r = create_aml_run(self.experiment, run_id)
            if with_details is None:
                with_details = True
            info = parse_run_info(r,
                    with_details=with_details,
                    with_log=self.with_log or with_log,
                    log_full=with_log)
            if info['status'] == self.status_failed:
                messages = detect_aml_error_message(info['appID'])
                if messages is not None:
                    info['result'] = ','.join(messages)
            info['cluster'] = self.cluster
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
                    logging.info('registering blob {}'.format(v['datastore_name']))
                    ds = Datastore.register_azure_blob_container(
                        workspace=self.ws,
                        datastore_name=v['datastore_name'],
                        container_name=cloud_blob.container_name,
                        account_name=cloud_blob.account_name,
                        account_key=cloud_blob.account_key,
                        blob_cache_timeout=v.get('blob_cache_timeout'),
                        overwrite=True,
                    )
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

    @deprecated('use self.upload()')
    def upload_qd_data(self, d):
        self.attach_data_store()
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

    @deprecated('use upload_file')
    def upload_qd_output(self, model_file):
        self.upload_file(model_file)
        #assert(op.isfile(model_file))
        #rel_model_file = op.relpath(model_file, './output')
        #target_path = op.join(self.config_param['output_folder']['path'],
                #rel_model_file)
        #cloud = self.config_param['output_folder']['cloud_blob']
        #if not cloud.exists(target_path):
            #cloud.az_upload2(model_file, target_path)

    def upload_file(self, fname):
        # this function is quite general to upload anyfile for any folder as
        # long as the folder will be from blobfuse in AML
        p = get_root_folder_in_curr_dir(fname)
        target_path = op.join(self.config_param['{}_folder'.format(p)]['path'],
                              fname[len(p) + 1:])
        cloud = self.config_param['{}_folder'.format(p)]['cloud_blob']
        if not cloud.exists(target_path):
            cloud.az_upload2(fname, target_path)

    @deprecated('use upload_file')
    def upload_qd_model(self, model_file):
        self.upload_file(model_file)

    def submit(self, cmd, num_gpu=None):
        self.attach_data_store()
        self.attach_mount_point()

        script_params = {'--' + p: v['mount_point'] for p, v in self.config_param.items()}
        script_params['--command'] = cmd
        if self.sleep_if_fail:
            script_params['--sleep_if_fail'] = '1'
        if self.compile_args:
            script_params['--compile_args'] = self.compile_args

        from azureml.train.estimator import Estimator
        from azureml.train.dnn import PyTorch
        if self.aks_compute_global_dispatch:
            from azureml.core import Environment
            env = Environment('myenv')
        else:
            import azureml
            env = azureml.core.runconfig.EnvironmentDefinition()
        env.docker.enabled = True
        env.docker.base_image = self.docker['image']
        env.docker.shm_size = '1024g'
        env.python.interpreter_path = '/opt/conda/bin/python'
        env.python.user_managed_dependencies = True

        # the env should be with str. here we just convert it
        env.environment_variables.update({k: str(v) for k, v in self.env.items()})

        from azureml.core.runconfig import MpiConfiguration
        mpi_config = MpiConfiguration()
        if num_gpu is None:
            num_gpu = self.num_gpu

        # this env is only used by some code with torch.distributed.launch
        env.environment_variables.update({'WORLD_SIZE': num_gpu})

        if num_gpu <= self.gpu_per_node:
            mpi_config.process_count_per_node = num_gpu
            node_count = 1
        else:
            assert (num_gpu % self.gpu_per_node) == 0
            mpi_config.process_count_per_node = self.gpu_per_node
            node_count = num_gpu // self.gpu_per_node
        if not self.multi_process:
            mpi_config.process_count_per_node = 1

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
        if self.aks_compute:
            from azureml.contrib.core.k8srunconfig import K8sComputeConfiguration
            k8sconfig = K8sComputeConfiguration()
            k8s = dict()
            id_rsa = op.expanduser('~/.ssh/id_rsa.pub')
            if op.isfile(id_rsa):
                k8s['enable_ssh'] = False
                k8s['ssh_public_key'] = read_to_buffer(
                    id_rsa).decode()
            k8s['preemption_allowed'] = self.preemption_allowed
            #k8s['gpu_count'] = 8
            k8sconfig.configuration = k8s
            estimator10.run_config.cmk8scompute = k8sconfig

            if self.aks_compute_global_dispatch:
                from azureml.contrib.core.gjdrunconfig import GlobalJobDispatcherConfiguration
                estimator10.run_config.global_job_dispatcher = GlobalJobDispatcherConfiguration(
                    compute_type="AmlK8s",
                    **self.aks_compute_global_dispatch_arg
                    # vm_size = ["Standard_ND40rs_v2","Standard_ND40s_v2"],
                    # region = ["eastus", "westus2"],
                    # my_resource_only = False,
                    # low_priority_vm_tolerant = True,
                )

        r = self.experiment.submit(estimator10)
        logging.info('job id = {}, cmd = \n{}'.format(r.id, cmd))
        return r.id

    def inject(self, run_id=None):
        all_info = self.query(run_id, max_runs=5000)
        from qd.db import update_cluster_job_db
        for info in all_info:
            if 'logFiles' in info:
                del info['logFiles']

        collection_name = self.kwargs.get('inject_collection', 'phillyjob')
        failed_jobs = [info for info in all_info if info['status'] == self.status_failed]
        failed_job_ids = [info['appID'] for info in failed_jobs]
        from qd.db import create_annotation_db
        c = create_annotation_db()
        appID_to_failed_job = {info['appID']: info for info in failed_jobs}
        for job_in_db in c.iter_general(collection_name,
                **{'appID': {'$in': failed_job_ids}}):
            if job_in_db['status'] == self.status_failed:
                del appID_to_failed_job[job_in_db['appID']]
        for _, info in appID_to_failed_job.items():
            info.update(self.query(info['appID'], with_log=True)[0])

        update_cluster_job_db(all_info,
                collection_name=collection_name)

    def sync_code(self, random_id, compile_in_docker=False, clean=True):
        assert random_id == ''
        import os.path as op
        random_qd = op.basename(self.config_param['code_path']['path'])
        from qd.qd_common import get_user_name
        random_abs_qd = op.join('/tmp', get_user_name(),
                '{}.zip'.format(random_qd))
        if op.isfile(random_abs_qd) and clean:
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
        self.config_param['code_path']['cloud_blob'].upload(random_abs_qd, rel_code_path)

    def list(self, file_or_folder):
        file_or_folder = clean_prefix(file_or_folder)
        p = get_root_folder_in_curr_dir(file_or_folder)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        path_in_blob = op.join(self.config_param[key]['path'], file_or_folder[len(p) + 1:])
        logging.info('path in blob: {}'.format(path_in_blob))
        infos = self.config_param[key]['cloud_blob'].list_blob_info(path_in_blob)
        for i in infos:
            i['name'] = i['name'][len(path_in_blob):]
        from qd.qd_common import print_table
        print_table(infos)

    def rm(self, file_or_folder):
        file_or_folder = clean_prefix(file_or_folder)
        p = get_root_folder_in_curr_dir(file_or_folder)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        path_in_blob = op.join(self.config_param[key]['path'], file_or_folder[len(p) + 1:])
        logging.info('path in blob: {}'.format(path_in_blob))
        self.config_param[key]['cloud_blob'].rm_prefix(path_in_blob)

    def upload(self, file_or_folder, from_cluster=None):
        file_or_folder = clean_prefix(file_or_folder)
        if from_cluster is None:
            self.upload_from_local(file_or_folder)
        else:
            self.upload_from_another(file_or_folder, from_cluster)

    def upload_from_another(self, file_or_folder, from_cluster):
        p = get_root_folder_in_curr_dir(file_or_folder)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        self.config_param[key]['cloud_blob'].upload(
            op.join(from_cluster.config_param[key]['path'], file_or_folder[len(p) + 1:]),
            op.join(self.config_param[key]['path'], file_or_folder[len(p) + 1:]),
            from_blob=from_cluster.config_param[key]['cloud_blob'],
        )

    def file_exists(self, file_name):
        p = get_root_folder_in_curr_dir(file_name)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        return self.config_param[key]['cloud_blob'].file_exists(
            op.join(self.config_param[key]['path'], file_name[len(p) + 1:]),
        )

    def dir_exists(self, file_name):
        p = get_root_folder_in_curr_dir(file_name)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        return self.config_param[key]['cloud_blob'].dir_exists(
            op.join(self.config_param[key]['path'], file_name[len(p) + 1:]),
        )

    def exists(self, file_or_dir):
        p = get_root_folder_in_curr_dir(file_or_dir)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        return self.config_param[key]['cloud_blob'].exists(
            op.join(self.config_param[key]['path'], file_or_dir[len(p) + 1:]),
        )

    def query_file_info(self, file_name):
        p = get_root_folder_in_curr_dir(file_name)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        return self.config_param[key]['cloud_blob'].query_info(
            op.join(self.config_param[key]['path'], file_name[len(p) + 1:]),
        )

    def upload_from_local(self, file_or_folder):
        p = get_root_folder_in_curr_dir(file_or_folder)
        key = '{}_folder'.format(p)
        assert key in self.config_param, self.config_param.keys()
        self.config_param[key]['cloud_blob'].upload(
            file_or_folder,
            op.join(self.config_param[key]['path'], file_or_folder[len(p) + 1:]),
        )

    def download(self, file_or_folder, as_prefix=False):
        file_or_folder = clean_prefix(file_or_folder)
        p = get_root_folder_in_curr_dir(file_or_folder)
        key = '{}_folder'.format(p)
        assert key in self.config_param
        self.config_param[key]['cloud_blob'].az_download(
            op.join(self.config_param[key]['path'], file_or_folder[len(p) + 1:]),
            file_or_folder,
            tmp_first=False
        )

    @deprecated('use download')
    def download_latest_qdoutput(self, full_expid):
        src_path = op.join(self.config_param['output_folder']['path'],
                full_expid)
        target_folder = op.join('output', full_expid)

        self.config_param['output_folder']['cloud_blob'].blob_download_qdoutput(
            src_path,
            target_folder,
        )

def inject_to_tensorboard(info):
    log_folder = 'output/tensorboard/aml'
    ensure_directory(log_folder)
    from torch.utils.tensorboard import SummaryWriter
    wt = SummaryWriter(log_dir=log_folder)
    for k, v in info.items():
        wt.add_scalar(tag=k, scalar_value=v)

def get_log_folder():
    return os.environ.get('AML_LOG_FOLDER', 'assets')

def retriable_error_codes():
    return [
        'ECC',
        'init_access',
        'waiting',
        'Init',
        'before',
        'ORTE_comm',
        'init',
        'blobIO',
        'NoSpace',
    ]

@try_once
def detect_aml_error_message(app_id):
    import glob
    folders = glob.glob(op.join(get_log_folder(), '*{}'.format(app_id),
        'azureml-logs'))
    error_codes = set()
    assert len(folders) == 1, 'not unique: {}'.format(', '.join(folders))
    folder = folders[0]
    has_nvidia_smi = False
    num_waiting = 0
    for r, _, files in os.walk(folder):
        for f in files:
            log_file = op.join(r, f)
            from qd.qd_common import decode_to_str
            all_line = decode_to_str(read_to_buffer(log_file)).split('\n')
            for i, line in enumerate(all_line):
                if 'Output size is too small' in line:
                    error_codes.add('InputSmall')
                if 'RuntimeError: CUDA driver initialization failed' in line:
                    error_codes.add('Init')
                if 'FileNotFoundError: [Errno 2] No such file or directory' in line:
                    error_codes.add('FileNotFound')
                if 'Signal code: Address not mapped' in line:
                    error_codes.add('before')
                if 'cuda runtime error (3) : initialization error' in line:
                    error_codes.add('Init')
                if 'has not done sshd setup wait' in line:
                    num_waiting += 1
                    if num_waiting > 1000:
                        error_codes.add('waiting')
                if 'RuntimeError: cuda runtime error (3) : initialization error at' in line:
                    error_codes.add('init')
                if "raise RuntimeError('NaN encountered!')" in line:
                    error_codes.add('NaN')
                if 'ValueError: regression_loss is NaN' in line:
                    error_codes.add('RegNaN')
                if 'ORTE has lost communication with a remote daemon.' in line:
                    error_codes.add('ORTE_comm')
                if 'RuntimeError: NCCL error in' in line or \
                        'RuntimeError: CUDA error: misaligned address' in line or \
                        'RuntimeError: Connection reset by peer' in line or \
                        'NCCL error in' in line or \
                        'CUDA error: all CUDA-capable devices' in line:
                    if any('_default_pg.barrier()' in l for l in all_line[
                            i - 100:
                            i]):
                        error_codes.add('Init')
                if 'RuntimeError: CUDA out of memory' in line or \
                        'CUDA error: out of memory' in line:
                    if any('_default_pg.barrier()' in l for l in all_line[
                            i - 100:
                            i]):
                        error_codes.add('Init')
                    else:
                        error_codes.add('OOM')
                if 'No module named' in line:
                    error_codes.add('ModuleErr')
                if 'copy_if failed to synchronize' in line:
                    error_codes.add('copy_if')
                if 'RuntimeError: connect() timed out' in line:
                    error_codes.add('connect')
                if 'unhandled cuda error' in line:
                    error_codes.add('cuda')
                if 'ECC error' in line:
                    error_codes.add('ECC')
                if 'an illegal memory access was encountered' in line:
                    if any('barrier()' in l for l in all_line[
                            i - 100:
                            i]):
                        error_codes.add('init_access')
                    else:
                        error_codes.add('illegal_access')
                if 'CUDA error' in line:
                    error_codes.add('cuda')
                if 'Error' in line and \
                    'TrackUserError:context_managers.TrackUserError' not in line and \
                    'WARNING: Retrying' not in line:
                    #start = max(0, i - 10)
                    start = max(0, i)
                    end = min(i + 1, len(all_line))
                    logging.info(log_file)
                    logging.info('\n'.join(all_line[start: end]))
                if 'Error response from daemon' in line:
                    error_codes.add('daemon')
                if "AttributeError: module 'maskrcnn_benchmark._C'" in line:
                    error_codes.add('mask compile')
                if 'OSError: [Errno 5] Input/output error' in line:
                    error_codes.add('blobIO')
                if 'OSError: [Errno 24] Too many open files' in line:
                    error_codes.add('TooMany')
                if 'OSError: [Errno 28] No space left on device:' in line and 'tmp' in line:
                    error_codes.add('NoSpace')
            # check how many gpus by nvidia-smi
            import re
            num_gpu = len([line for line in all_line if re.match('.*N/A.*Default', line) is
                not None])
            has_nvidia_smi =  any(('nvidia-smi' in line for line in all_line))
            if has_nvidia_smi and num_gpu != 4:
                logging.info(log_file)
                logging.info(num_gpu)
    return list(error_codes)


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
        return self._default_client

    def search_partial_id(self, partial_id):
        search_result = search_partial_id_from_db(partial_id)
        if search_result is None:
            client = self.default_client()
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

    def ssh(self, partial_id):
        client, partial_id = self.search_partial_id(partial_id)
        client.ssh(partial_id)

def execute(task_type, **kwargs):
    if task_type in ['q', 'query']:
        if len(kwargs.get('remainders', [])) > 0:
            c = create_aml_client(**kwargs)
            all_info = list(c.iter_query(
                run_id=kwargs['remainders'],
                with_log=kwargs.get('with_log', True),
                log_full=kwargs.get('log_full', True),
                with_details=kwargs.get('with_details', True),
                detect_error_if_failed=True,
            ))
            from qd.qd_common import print_job_infos
            print_job_infos(all_info)
        else:
            c = create_aml_client(**kwargs)
            c.query(max_runs=kwargs.get('max', None))
    elif task_type in ['f', 'failed', 'qf']:
        c = create_aml_client(**kwargs)
        c.query(by_status=AMLClient.status_failed)
    elif task_type in ['qq']:
        c = create_aml_client(**kwargs)
        c.query(by_status=AMLClient.status_queued)
    elif task_type in ['ssh']:
        assert len(kwargs['remainders']) == 1
        c = MultiAMLClient(**kwargs)
        c.ssh(partial_id=kwargs['remainders'][0])
    elif task_type in ['qr']:
        c = create_aml_client(**kwargs)
        c.query(by_status=AMLClient.status_running)
    elif task_type in ['init', 'initc', 'initi', 'initic']:
        c = create_aml_client(**kwargs)
        clean = task_type in ['init', 'initc']
        c.sync_code('', clean=clean)
        if task_type in ['initc', 'initic']:
            # in this case, we first upload the raw code to unblock the job
            # submission. then we upload the compiled version to reduce the
            # overhead in aml running
            c.sync_code('', compile_in_docker=True, clean=clean)
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
    elif task_type in ['download', 'd']:
        c = create_aml_client(**kwargs)
        for full_expid in kwargs['remainders']:
            c.download(full_expid)
    elif task_type in ['upload', 'u']:
        if not kwargs.get('from_cluster'):
            c = create_aml_client(**kwargs)
            for data in kwargs['remainders']:
                c.upload(data)
        else:
            param = copy.deepcopy(kwargs)
            param.pop('from_cluster')
            c = create_aml_client(**param)
            param = copy.deepcopy(kwargs)
            param['cluster'] = param['from_cluster']
            param.pop('from_cluster')
            from_c = create_aml_client(**param)
            for data in kwargs['remainders']:
                c.upload(data, from_cluster=from_c)
    elif task_type in ['ls']:
        c = create_aml_client(**kwargs)
        for data in kwargs['remainders']:
            c.list(data)
    elif task_type in ['rm']:
        c = create_aml_client(**kwargs)
        for data in kwargs['remainders']:
            c.rm(data)
    elif task_type == 'blame':
        c = create_aml_client(**kwargs)
        c.blame()
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
        result = []
        for partial_id in partial_ids:
            run_info = client.query_one(partial_id,
                                        with_details=True)
            result.append(dest_client.submit(run_info['cmd'],
                    num_gpu=run_info['num_gpu']))
            client.abort(run_info['appID'])
        logging.info(pformat(result))
    elif task_type in ['s', 'summary']:
        m = create_aml_client(**kwargs)
        info = m.get_cluster_status()
        logging.info(pformat(info))
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
                'd', 'download',
                'u', 'upload',
                'monitor',
                'parse',
                'init',
                'initc', # init with compile
                'initi', # incremental init
                'initic', # incremental & compile
                'blame', 'resubmit',
                     'ls', 'rm',
                's', 'summary', 'i', 'inject'])
    parser.add_argument('-no-wl', dest='with_log', action='store_false')
    parser.add_argument('-no-dt', dest='with_details', action='store_false')
    parser.add_argument('-no-lf', dest='log_full', action='store_false')
    parser.add_argument('-hold', dest='sleep_if_fail', default=False, action='store_true')
    parser.add_argument('-c', '--cluster', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-rt', '--resubmit_to', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-f', '--from_cluster', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-ca', '--compile_args', default=argparse.SUPPRESS, type=str)
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-n', '--num_gpu', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--max', default=None, type=int)
    #parser.add_argument('-wg', '--with_gpu', default=True, action='store_true')
    parser.add_argument('-no-wg', '--with_gpu', default=True, action='store_false')
    #parser.add_argument('-m', '--with_meta', default=True, action='store_true')
    parser.add_argument('-no-m', '--with_meta', default=True, action='store_false')

    parser.add_argument('-no-s', '--real_submit', default=True, action='store_false')
    parser.add_argument('-ic', '--inject_collection', default='phillyjob', type=str)

    parser.add_argument('remainders', nargs=argparse.REMAINDER,
            type=str)
    return parser.parse_args()

if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)
