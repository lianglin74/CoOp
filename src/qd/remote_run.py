import sys
import os
import psutil
import time
import yaml
import os.path as op
import logging
from pprint import pformat
import multiprocessing as mp

from qd.qd_common import write_to_yaml_file, load_from_yaml_file
from qd.gpu_util import gpu_available
from qd.qd_common import is_cluster
from qd.qd_common import cmd_run
from qd.qd_common import write_to_file

def sync_qd(ssh_info, delete=False):
    target_folder = '/tmp/code/quickdetection/'
    sync(ssh_info=ssh_info, target_folder=target_folder, delete=delete)
    remote_run('cd {} && ./compile.conda.sh'.format(target_folder), ssh_info)

    remote_run('{} -m nltk.downloader all'.format(get_executable()), ssh_info)

    if is_cluster(ssh_info):
        c = []
        c.append('cd {}'.format(op.join(target_folder, 'src/CCSCaffe/python')))
        c.append('sudo python -c "import caffe; caffe.set_mode_gpu()"')
        remote_run(' && '.join(c), ssh_info)

    special_path = ['data', 'output', 'models']
    for p in special_path:
        if p in ssh_info:
            c = []
            c.append('cd {}'.format(target_folder))
            c.append('rm -rf {}'.format(p))
            c.append('ln -s {} {}'.format(ssh_info[p], op.join(target_folder,
                p)))
            remote_run(' && '.join(c), ssh_info)

def sync(ssh_info, from_folder='/home/jianfw/code/quickdetection/',
        target_folder='/tmp/code/quickdetection/', delete=False):
    remote_run('mkdir -p {}'.format(target_folder), ssh_info)
    cmd = []
    cmd.append('rsync')
    if delete:
        cmd.append('--delete')
    cmd.append('-tvrzh')
    cmd.append('--links')
    def exclude_if_exists(sub):
        if op.exists(op.join(from_folder, sub)):
            cmd.append('--exclude')
            cmd.append('/{}'.format(sub))
    exclude_if_exists('src/CCSCaffe/.build_release/')
    exclude_if_exists('src/CCSCaffe/.build_debug/')
    exclude_if_exists('src/CCSCaffe/.build/')
    exclude_if_exists('src/CCSCaffe/python/caffe/_caffe.so')
    exclude_if_exists('src/CCSCaffe/python/caffe/proto/')
    exclude_if_exists('src/CCSCaffe/Makefile.config')
    exclude_if_exists('assets')
    exclude_if_exists('output')
    exclude_if_exists('data')
    exclude_if_exists('models')
    exclude_if_exists('.git')
    exclude_if_exists('src/CCSCaffe/.git')
    exclude_if_exists('tmp_run')
    exclude_if_exists('visualization')
    cmd.append('--exclude')
    cmd.append('*.swp')
    cmd.append('--exclude')
    cmd.append('*.so')
    cmd.append('--exclude')
    cmd.append('*.swo')
    cmd.append('--exclude')
    cmd.append('*.o')
    cmd.append('--exclude')
    cmd.append('*.caffemodel')
    cmd.append('--exclude')
    cmd.append('*.solverstate')
    cmd.append('--exclude')
    cmd.append('*.pyc')
    cmd.append('{}'.format(from_folder))
    extra_ssh_info = []
    for key in ssh_info:
        if len(key) > 0 and key[0] == '-':
            extra_ssh_info.append(key)
            extra_ssh_info.append(str(ssh_info[key]))
    cmd.append('-e')
    if len(extra_ssh_info) > 0:
        cmd.append('ssh {}'.format(' '.join(extra_ssh_info)))
    else:
        cmd.append('ssh -i /home/jianfw/.ssh/id_rsa')
    cmd.append('{}@{}:{}'.format(ssh_info['username'],
        ssh_info['ip'],
        target_folder))
    cmd_run(cmd)

def scp_f(local_folder, target_file, ssh_cmd):
    logging.info('ssh info' + str(ssh_cmd))
    logging.info((local_folder, target_file))
    cmd = ['scp', '-r']
    if '-p' in ssh_cmd:
        cmd.append('-P')
        cmd.append(str(ssh_cmd['-p']))
    if '-i' in ssh_cmd:
        cmd.append('-i')
        cmd.append(ssh_cmd['-i'])
    cmd += [local_folder, '{}@{}:{}'.format(ssh_cmd['username'],
        ssh_cmd['ip'], target_file)]
    cmd_run(cmd)

def scp(local_file, target_file, ssh_cmd):
    assert op.isfile(local_file)
    logging.info('ssh info' + str(ssh_cmd))
    logging.info((local_file, target_file))
    cmd = ['scp']
    if '-p' in ssh_cmd:
        cmd.append('-P')
        cmd.append(str(ssh_cmd['-p']))
    if '-i' in ssh_cmd:
        cmd.append('-i')
        cmd.append(ssh_cmd['-i'])
    cmd += [local_file, '{}@{}:{}'.format(ssh_cmd['username'],
        ssh_cmd['ip'], target_file)]
    cmd_run(cmd)

def remote_run(str_cmd, ssh_info, return_output=False):
    cmd = ['ssh', '-t', '-t', '-o', 'StrictHostKeyChecking no']
    for key in ssh_info:
        if len(key) > 0 and key[0] == '-':
            cmd.append(key)
            cmd.append(str(ssh_info[key]))
    cmd.append('{}@{}'.format(ssh_info['username'], ssh_info['ip']))
    if is_cluster(ssh_info):
        prefix = 'source ~/.bashrc && export PATH=/usr/local/nvidia/bin:$PATH && '
    else:
        cs = []
        # don't use anaconda since caffe is slower under anaconda because of the
        # data preprocessing. not i/o
        cs.append('source ~/.bashrc')
        if 'conda' in get_executable():
            cs.append('export PATH=$HOME/anaconda3/bin:\$PATH')
            cs.append('export LD_LIBRARY_PATH=$HOME/anaconda3/lib:\$LD_LIBRARY_PATH')
        cs.append('export PATH=/usr/local/nvidia/bin:\$PATH')
        cs.append('export PYTHONPATH=/tmp/code/quickdetection/src/CCSCaffe/python:\$PYTHONPATH')
        prefix = ' && '.join(cs) + ' && '

    suffix = ' && hostname'
    ssh_command = '{}{}{}'.format(prefix, str_cmd, suffix)
    # this will use the environment variable like what you have after ssh
    ssh_command = 'bash -i -c "{}"'.format(ssh_command)
    cmd.append(ssh_command)

    return cmd_run(cmd, return_output)

def get_executable():
    return sys.executable

def remote_python_run(func, kwargs, ssh_cmd, cmd_prefix=''):
    logging.info('ssh_cmd: ' + str(ssh_cmd))
    working_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = '/tmp/code/quickdetection/scripts'
    # serialize kwargs locally
    if kwargs is None:
        str_args = ''
    else:
        str_args = yaml.dump(kwargs)
    logging.info(str_args)
    param_basename = 'remote_run_param_{}.txt'.format(hash(str_args))
    param_local_file = '/tmp/{}'.format(param_basename)
    write_to_file(str_args, param_local_file)
    param_target_file = op.join(working_dir, param_basename)
    # send it to the remote machine
    scp(param_local_file, param_target_file, ssh_cmd)
    # generate the script locally
    if func.__module__ == '__main__':
        func_module = 'run'
    else:
        func_module = func.__module__

    import inspect
    argnames, ins_varargs, ins_kwargs, ins_defaults = inspect.getargspec(func)
    if len(argnames) == 0 and \
            ins_varargs is None and \
            ins_kwargs is None and \
            ins_defaults is None:
        script = '''
import matplotlib
matplotlib.use(\'Agg\')
if __name__ == '__main__':
    from {} import {}
    {}()
    '''.format(func_module, func.__name__, func.__name__)
    else:
        script = '''
import matplotlib
matplotlib.use(\'Agg\')
if __name__ == '__main__':
    from {} import {}
    import yaml
    param = yaml.load(open(\'{}\', \'r\').read())
    {}(**param)
    '''.format(func_module, func.__name__, param_target_file,
        func.__name__)
    basename = 'remote_run_{}.py'.format(hash(script))
    local_file = '/tmp/{}'.format(basename)
    write_to_file(script, local_file)

    target_file = op.join(working_dir, basename)
    scp(local_file, target_file, ssh_cmd)
    if len(cmd_prefix) > 0 and not cmd_prefix.endswith(' '):
        cmd_prefix = cmd_prefix + ' '
    remote_run('cd {} && {}{} {}'.format(
        op.dirname(working_dir),
        cmd_prefix,
        get_executable(),
        target_file),
        ssh_cmd)

def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except:
        return False

def collect_process_info():
    result = {}
    for process in psutil.process_iter():
        result[process.pid] = {}
        result[process.pid]['username'] = process.username()
        result[process.pid]['time_spent_in_hour'] = (int(time.time()) -
                process.create_time()) / 3600.0
        result[process.pid]['cmdline'] = ' '.join(process.cmdline())
    return result

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

