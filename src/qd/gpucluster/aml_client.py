from pprint import pformat
from collections import OrderedDict
import logging
from qd.qd_common import load_from_yaml_file, dict_update_nested_dict
from qd.qd_common import cmd_run
from qd.cloud_storage import create_cloud_storage
from qd.qd_common import decode_general_cmd

def create_aml_client(**kwargs):
    param = load_from_yaml_file('./aux_data/aml/aml.yaml')
    dict_update_nested_dict(param, kwargs)
    return AMLClient(**param)

def parse_run_info(run):
    info = {}
    info['status'] = run.status
    info['appID'] = run.id
    info['appID-s'] = run.id[-5:]
    info['cluster'] = 'aml'
    details = run.get_details()
    info['start_time'] = details.get('startTimeUtc')
    from dateutil.parser import parse
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if info['start_time'] is not None:
        d= (now - parse(info['start_time'])).total_seconds() / 3600
        info['elapsedTime'] = round(d, 2)
    if len(details['runDefinition']['arguments']) > 0:
        cmd = details['runDefinition']['arguments'][-1]
        info['cmd'] = cmd
        info['cmd_param'] = decode_general_cmd(cmd)
    info['num_gpu'] = details['runDefinition']['mpi']['processCountPerNode'] * \
            details['runDefinition']['nodeCount']
    return info

def print_run_info(run):
    info = parse_run_info(run)
    logging.info(pformat(info))

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

class AMLClient(object):
    status_running = 'Running'
    status_queued = 'Queued'
    def __init__(self, azure_blob_config_file, config_param, docker,
            datastore_name, aml_config, use_custom_docker,
            compute_target, source_directory, entry_script,
            with_log=True,
            **kwargs):
        self.kwargs = kwargs
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
        self.num_gpu = kwargs.get('num_gpu', 4)
        self.docker = docker

        self.cloud_blob = create_cloud_storage(config_file=azure_blob_config_file)
        self.config_param = config_param
        self.with_log = with_log

        from azureml.core import Workspace
        ws = Workspace.from_config(self.aml_config)
        self.ws = ws
        compute_target = ws.compute_targets[self.compute_target]
        self.compute_target = compute_target

        from azureml.core import Datastore
        ds = Datastore.register_azure_blob_container(workspace=ws,
                                                     datastore_name=self.datastore_name,
                                                     container_name=self.cloud_blob.container_name,
                                                     account_name=self.cloud_blob.account_name,
                                                     account_key=self.cloud_blob.account_key)

        self.code_path = ds.path(self.config_param['code_path']).as_mount()
        self.data_folder = ds.path(self.config_param['data_folder']).as_mount()
        self.model_folder = ds.path(self.config_param['model_folder']).as_mount()
        self.output_folder = ds.path(self.config_param['output_folder']).as_mount()

        self.use_custom_docker = use_custom_docker

        from azureml.core import Experiment
        self.experiment = Experiment(self.ws, name=self.experiment_name)

    def abort(self, run_id):
        run = create_aml_run(self.experiment, run_id)
        run.cancel()

    def query(self, run_id=None):
        if run_id is None:
            all_run = list(self.experiment.get_runs())
            all_info = [parse_run_info(r) for r in all_run]
            if self.with_log:
                for info, r in zip(all_info, all_run):
                    if r.status != AMLClient.status_running:
                        continue
                    all_log = list(r.get_all_logs())
                    info['all_log_path'] = all_log
                    if len(all_log) > 0:
                        info['latest_log'] = cmd_run(['tail', '-n', '100',
                            all_log[0]],
                            return_output=True)
                        from qd.qd_common import attach_log_parsing_result
                        attach_log_parsing_result(info)

            from qd.qd_common import print_job_infos
            param_keys = ['data', 'net', 'expid']
            for k in param_keys:
                for info in all_info:
                    if info is None:
                        import ipdb;ipdb.set_trace(context=15)
                    from qd.qd_common import dict_has_path
                    if dict_has_path(info, 'cmd_param$param${}'.format(k)):
                        v = info['cmd_param']['param'][k]
                    else:
                        v = None
                    info[k] = v
            print_job_infos(all_info)
            return all_info
        else:
            r = create_aml_run(self.experiment, run_id)
            logging.info(r.get_portal_url())

            if self.with_log:
                all_log = r.get_all_logs()
                if len(all_log) > 0:
                    cmd_run(['tail', '-n', '100', all_log[0]])
                    logging.info('log files: \n{}'.format(pformat(all_log)))
            print_run_info(r)

    def resubmit(self, partial_id):
        run = create_aml_run(self.experiment, partial_id)
        run_info = parse_run_info(run)
        self.submit(run_info['cmd'])
        run.cancel()

    def submit(self, cmd, num_gpu=None):
        script_params = {
                '--code_path': self.code_path,
                '--data_folder': self.data_folder,
                '--model_folder': self.model_folder,
                '--output_folder': self.output_folder,
                '--command': cmd}

        from azureml.train.estimator import Estimator
        from azureml.train.dnn import PyTorch
        import azureml
        env = azureml.core.runconfig.EnvironmentDefinition()
        env.docker.enabled = True
        env.docker.base_image = self.docker['image']
        env.docker.shm_size = '16g'
        env.python.interpreter_path = '/opt/conda/bin/python'
        env.python.user_managed_dependencies = True

        env.environment_variables['NCCL_TREE_THRESHOLD'] = '0'
        env.environment_variables['NCCL_LL_THRESHOLD'] = '0'

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

    def inject(self):
        all_info = self.query()
        from qd.db import update_cluster_job_db
        update_cluster_job_db(all_info)

    def sync_code(self, random_id):
        random_qd = 'quickdetection{}'.format(random_id)
        import os.path as op
        random_abs_qd = op.join('/tmp', '{}.zip'.format(random_qd))
        logging.info('{}'.format(random_qd))
        from qd.qd_common import zip_qd
        # zip it
        zip_qd(random_abs_qd)

        rel_code_path = self.config_param['code_path']
        # upload it
        self.cloud_blob.az_upload2(random_abs_qd, rel_code_path)

def execute(task_type, **kwargs):
    if task_type in ['q', 'query']:
        if len(kwargs.get('remainders', [])) > 0:
            assert len(kwargs['remainders']) == 1
            c = create_aml_client(**kwargs)
            c.query(run_id=kwargs['remainders'][0])
        else:
            c = create_aml_client(**kwargs)
            c.query()
    elif task_type == 'init':
        c = create_aml_client(**kwargs)
        c.sync_code('')
    elif task_type == 'submit':
        c = create_aml_client(**kwargs)
        params = kwargs['remainders']
        cmd = ' '.join(params)
        c.submit(cmd)
    elif task_type in ['a', 'abort']:
        c = create_aml_client(**kwargs)
        for v in kwargs['remainders']:
            v = v.strip('/')
            c.abort(v)
    elif task_type == 'blame':
        raise NotImplementedError()
        blame(**kwargs)
    elif task_type == 'resubmit':
        partial_ids = kwargs['remainders']
        del kwargs['remainders']
        client = create_aml_client(**kwargs)
        if len(partial_ids) == 0:
            client.auto_resubmit()
        else:
            for partial_id in partial_ids:
                client.resubmit(partial_id)
    elif task_type == 'summary':
        raise NotImplementedError()
        m = create_multi_philly_client()
        m.print_summary()
    elif task_type in ['i', 'inject']:
        m = create_aml_client(**kwargs)
        m.inject()
    else:
        assert 'Unknown {}'.format(task_type)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Philly Interface')
    parser.add_argument('task_type',
            choices=['ssh', 'q', 'query', 'a', 'abort', 'submit',
                'init',
                'sync',
                'update_config', 'gc', 'blame', 'resubmit',
                'summary', 'i', 'inject'])
    parser.add_argument('-wl', dest='with_log', default=True, action='store_true')
    parser.add_argument('-no-wl', dest='with_log',
            action='store_false')
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

if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)
