from pprint import pformat
from collections import OrderedDict
import logging
from qd.qd_common import load_from_yaml_file, dict_update_nested_dict
from qd.cloud_storage import create_cloud_storage

def create_aml_client(**kwargs):
    param = load_from_yaml_file('./aux_data/configs/aml.yaml')
    dict_update_nested_dict(param, kwargs)
    return AMLClient(**param)

def parse_run_info(run):
    info = {}
    info['status'] = run.status
    info['run_id'] = run.id
    details = run.get_details()
    info['start_time'] = details.get('startTimeUtc')
    from dateutil.parser import parse
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if info['start_time'] is not None:
        d= (now - parse(info['start_time'])).total_seconds() / 3600
        info['elapsed_hours'] = round(d, 2)
    cmd = details['runDefinition']['arguments'][-1]
    info['cmd'] = cmd
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
    def __init__(self, azure_blob_id, config_param, docker, **kwargs):
        self.kwargs = kwargs
        self._initialized = False
        self.aml_config = 'aux_data/aml/config.json'
        self.compute_target = 'NC24RSV3'
        # do not change the datastore_name unless the storage account
        # information is changed
        self.datastore_name = 'vig_data'
        # it does not matter what the name is
        self.experiment_name = 'none'
        self.num_gpu = kwargs.get('num_gpu', 4)
        self.azure_blob_id = azure_blob_id
        self.docker = docker

        self.cloud_blob = create_cloud_storage(self.azure_blob_id)
        self.config_param = config_param
        self.with_log = True

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

        from azureml.core import Experiment
        self.experiment = Experiment(self.ws, name=self.experiment_name)

    def abort(self, run_id):
        run = create_aml_run(self.experiment, run_id)
        run.cancel()

    def query(self, run_id=None):
        if run_id is None:
            all_run = self.experiment.get_runs()
            all_info = [parse_run_info(r) for r in all_run]
            running_runs = [r for r in all_run if r.status ==
                    AMLClient.status_running]
            for r in running_runs:
                all_log = list(r.get_all_logs())
            #all_info = [parse_run_info(r) for r in all_run
                    #if r.status in [AMLClient.status_running, AMLClient.status_queued]]
            from qd.qd_common import print_table
            all_info = [i for i in all_info if i['status'] in
                [AMLClient.status_running, AMLClient.status_queued]]
            print_table(all_info, ['status', 'run_id', 'elapsed_hours'])

        else:
            r = create_aml_run(self.experiment, run_id)

            if self.with_log:
                all_log = r.get_all_logs()
            from qd.qd_common import cmd_run
            cmd_run(['tail', '-n', '100', all_log[0]])
            logging.info(pformat(all_log))
            print_run_info(r)

    def submit(self, cmd):
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
        if self.num_gpu <= 4:
            mpi_config.process_count_per_node = self.num_gpu
            node_count = 1
        else:
            assert (self.num_gpu % 4) == 0
            mpi_config.process_count_per_node = 4
            node_count = self.num_gpu // 4

        #estimator10 = PyTorch(
                #source_directory='./src/qd/gpucluster',
                #compute_target=self.compute_target,
                #script_params=script_params,
                #entry_script='aml_server.py',
                #environment_definition=env,
                #node_count=node_count,
                #distributed_training=mpi_config,
                #)

        estimator10 = Estimator(
                source_directory='./src/qd/gpucluster',
                compute_target=self.compute_target,
                script_params=script_params,
                entry_script='aml_server.py',
                environment_definition=env,
                node_count=node_count,
                distributed_training=mpi_config,
                )

        r = self.experiment.submit(estimator10)
        logging.info('job id = {}, cmd = \n{}'.format(r.id, cmd))

def execute(task_type, **kwargs):
    if task_type in ['q', 'query']:
        if len(kwargs.get('remainders', [])) > 0:
            assert len(kwargs['remainders']) == 1
            c = create_aml_client(**kwargs)
            c.query(run_id=kwargs['remainders'][0])
        else:
            c = create_aml_client(**kwargs)
            c.query()
    elif task_type == 'submit':
        assert len(kwargs['remainders']) == 1
        submit_without_sync(cmd=kwargs['remainders'][0], **kwargs)
    elif task_type in ['a', 'abort']:
        c = create_aml_client(**kwargs)
        for v in kwargs['remainders']:
            v = v.strip('/')
            c.abort(v)
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
    elif task_type == 'inject':
        if not kwargs.get('with_log'):
            kwargs['with_log'] = True
        m = create_multi_philly_client(**kwargs)
        m.inject()
    else:
        assert 'Unknown {}'.format(task_type)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Philly Interface')
    parser.add_argument('task_type',
            choices=['ssh', 'q', 'query', 'a', 'abort', 'submit', 'sync',
                'update_config', 'gc', 'blame', 'init', 'resubmit',
                'summary', 'inject'])
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

if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)
