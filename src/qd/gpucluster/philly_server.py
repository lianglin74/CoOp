#!/usr/bin/env python
import os
import sys
import logging
import zipfile
import subprocess as sp
import os.path as op
import base64
import yaml
from pprint import pformat

def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

def init_logging():
    import socket
    fmt = '%(asctime)s.%(msecs)03d {}-{} %(process)d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s'.format(
        socket.gethostname(), get_mpi_rank())

    log_file = '/tmp/config_log/log_rank{}.txt'.format(get_mpi_rank())
    ensure_directory(op.dirname(log_file))
    file_handle = logging.FileHandler(log_file)
    logger_fmt = logging.Formatter(fmt)
    file_handle.setFormatter(fmt=logger_fmt)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logger_fmt
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.INFO)
    root.addHandler(file_handle)
    root.addHandler(ch)

def unzip(zip_file, target_folder):
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(path=target_folder)

def cmd_run(cmd, working_directory='./', succeed=True,
        return_output=False):
    e = os.environ.copy()
    e['PYTHONPATH'] = '/app/caffe/python:{}'.format(e.get('PYTHONPATH', ''))
    # in the maskrcnn, it will download the init model to TORCH_HOME. By
    # default, it is /root/.torch, which is different among diferent nodes.
    # However, teh implementation assumes that folder is a share folder. Thus
    # only rank 0 do the data downloading. Here, we assume the output folder is
    # shared, which is the case in AML.
    e['TORCH_HOME'] = './output/torch_home'
    ensure_directory(e['TORCH_HOME'])
    logging.info('start to cmd run: {}'.format(' '.join(map(str, cmd))))
    for c in cmd:
        logging.info(c)
    if not return_output:
        try:
            p = sp.Popen(cmd, stdin=sp.PIPE,
                    cwd=working_directory,
                    env=e)
            p.communicate()
            if succeed:
                assert p.returncode == 0
        except:
            if succeed:
                logging.info('raising exception')
                raise
    else:
        return sp.check_output(cmd)

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise
        # we should always check if it succeeds.
        assert op.isdir(path), 'failed'

def compile_qd(folder):
    path = os.environ['PATH']
    cmd_run(['sudo', 'env', 'PATH={}'.format(path), 'pip', 'install', '--no-index',
        '--find-links', '/var/storage/shared/input/jianfw/pipwheels',
        '-r', 'requirements.txt'],
        working_directory=folder,
        succeed=False)
    cmd_run(['sudo', 'env', 'PATH={}'.format(path), 'pip', 'install',
        '-r', 'requirements.txt'],
        working_directory=folder,
        succeed=False)

    compile_file = 'compile.philly.sh'
    cmd_run(['chmod', '+x', op.join(folder, compile_file)])
    #cmd_run(['sudo', 'env', 'PATH={}'.format(path), './{}'.format(compile_file)],
            #working_directory=folder, succeed=False)

    # in the context of sudo -i, the current folder will be changed to /root.
    # so we need to use the full file path of compile_file
    cmd_run(['./{}'.format(compile_file)],
            working_directory=folder, succeed=False)

def update_ssh():
    cmd_run(['cp', op.expanduser('~/.ssh/id_rsa'), op.expanduser('~/')])
    cmd_run(['sudo', 'chmod', '777', op.expanduser('~/id_rsa')])

def get_mpi_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

import fcntl

def acquireLock():
    ''' acquire exclusive lock file access '''
    logging.info('start to acqure the lock')
    locked_file_descriptor = open('/tmp/lockfile.LOCK', 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    logging.info('acqured')
    return locked_file_descriptor

def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()

def wrap_all(code_zip, code_root,
        data_folder, model_folder, output_folder, command):

    if get_mpi_rank() == 0:
        p = launch_monitoring_process()

    lock_fd = acquireLock()
    logging.info('got the lock')
    if not op.isdir(code_root):
        logging.info('setup the code')
        cmd_run(['grep', 'Port', '/etc/ssh/sshd_config'])
        cmd_run(['ifconfig'])
        cmd_run(['df', '-h'])
        cmd_run(['nvidia-smi'])

        ensure_directory(code_root)

        # set up the code, models, output under qd
        logging.info('unzipping {}'.format(code_zip))
        unzip(code_zip, code_root)
        cmd_run(['rm', '-rf', 'data'], code_root)
        cmd_run(['rm', '-rf', 'models'], code_root, succeed=False)
        cmd_run(['rm', '-rf', 'output'], code_root)
        cmd_run(['ln', '-s',
            data_folder,
            op.join(code_root, 'data')])
        cmd_run(['ln',
            '-s',
            model_folder,
            op.join(code_root, 'models')
            ])

        cmd_run(['mkdir', '-p', output_folder])

        cmd_run(['sudo', 'chmod', 'a+rw',
            output_folder], succeed=False)

        cmd_run(['ln', '-s', output_folder, op.join(code_root, 'output')])

        # compile the source code
        compile_qd(code_root)
    else:
        logging.info('skip to setup the code')
    releaseLock(lock_fd)

    if type(command) is str:
        command = list(command.split(' '))

    if len(command) > 0:
        cmd_run(command, working_directory=code_root)

    if get_mpi_rank() == 0:
        terminate_monitoring_process(p)

def terminate_monitoring_process(p):
    p.terminate()
    p.join()

def parse_gpu_usage_dict(result):
    import re
    used = []
    p = '^\|.* ([0-9]*)MiB \/ *([0-9]*)MiB *\| *([0-9]*)\%.*Default \|$'
    for line in result.split('\n'):
        line = line.strip()
        r = re.match(p, line)
        if r != None:
            u = [int(g) for g in r.groups()]
            names = ['mem_used', 'mem_total', 'gpu_util']
            used.append({n: v for n, v in zip(names, u)})
    return used

def monitor():
    while True:
        cmd_result = cmd_run(['nvidia-smi'], return_output=True).decode()
        gpu_result = parse_gpu_usage_dict(cmd_result)
        logging.info('{}'.format(gpu_result))
        import time
        time.sleep(60 * 30) # every 30 minutes

def launch_monitoring_process():
    from multiprocessing import Process
    p = Process(target=monitor)
    p.start()
    return p

def get_mpi_local_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))

def get_mpi_local_size():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))

def link_nltk():
    target_folder = op.expanduser('~/nltk_data')
    if op.islink(target_folder) or op.exists(target_folder):
        return
    nltk_folder = '/var/storage/shared/input/jianfw/nltk_data'
    if op.isdir(nltk_folder):
        os.symlink(nltk_folder, op.expanduser('~/nltk_data'))

def load_from_yaml_str(s):
    return yaml.load(s)

def run_in_philly():
    logging.info(sys.executable)
    extra_param = sys.argv[4]
    dict_param = load_from_yaml_str(base64.b64decode(extra_param))

    logging.info('start')
    logging.info(pformat(dict_param))

    for k in os.environ:
        logging.info('{} = {}'.format(k, os.environ[k]))

    qd_root = op.join('/tmp', 'code', 'quickdetection')

    # usefull only in philly@ap. no hurt for other philly
    if get_mpi_rank() == 0:
        update_ssh()
        link_nltk()

    wrap_all(dict_param['code_path'], qd_root,
            dict_param['data_folder'], dict_param['model_folder'],
            dict_param['output_folder'], dict_param['command'])

    # the permission should be changed because the output is there, but the
    # permission is for the docker job only and teh philly-fs cannot delete or
    # change it
    #if get_mpi_rank() == 0:
        #cmd_run(['sudo', 'chmod', '777',
            #dict_param['output_folder'],
            #'-R'], succeed=False)

if __name__ == '__main__':
    init_logging()

    run_in_philly()
