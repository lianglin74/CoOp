import sys
import os
import psutil
import time
import yaml
import os.path as op
import logging
from qd.qd_common import is_cluster
from qd.qd_common import cmd_run
from qd.qd_common import write_to_file

def sync_qd(ssh_info, delete=False):
    target_folder = '/tmp/code/quickdetection/'
    sync(ssh_info=ssh_info, target_folder=target_folder, delete=delete)
    remote_run('cd {} && ./compile.sh'.format(target_folder), ssh_info)

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
        #cs.append('export PATH=$HOME/anaconda3/bin:$PATH')
        #cs.append('export LD_LIBRARY_PATH=$HOME/anaconda3/lib:$LD_LIBRARY_PATH')
        cs.append('export PATH=/usr/local/nvidia/bin:$PATH')
        cs.append('export PYTHONPATH=/tmp/code/quickdetection/src/CCSCaffe/python:$PYTHONPATH')
        prefix = ' && '.join(cs) + ' && '

    suffix = ' && hostname'
    cmd.append('{}{}{}'.format(prefix, str_cmd, suffix))

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

