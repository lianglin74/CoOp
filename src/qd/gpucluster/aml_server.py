#!/usr/bin/python
import os
import sys
import logging
import zipfile
import subprocess as sp
import os.path as op
import base64
import yaml

def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

def init_logging():
    import socket
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s.%(msecs)03d {} %(process)d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s'.format(
            socket.gethostname()),
        datefmt='%m-%d %H:%M:%S',
    )

def unzip(zip_file, target_folder):
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(path=target_folder)

def cmd_run(cmd, working_directory='./', succeed=False,
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
        try:
            if not os.path.exists(path) and not op.islink(path):
                os.makedirs(path)
        except:
            pass

def compile_qd(folder):
    path = os.environ['PATH']
    cmd_run(['env', 'PATH={}'.format(path), 'pip', 'install', '--no-index',
        '--find-links', '/var/storage/shared/input/jianfw/pipwheels',
        '-r', 'requirements.txt'],
        working_directory=folder,
        succeed=False)
    cmd_run(['env', 'PATH={}'.format(path), 'pip', 'install',
        '-r', 'requirements.txt'],
        working_directory=folder,
        succeed=False)

    compile_file = 'compile.aml.sh'
    cmd_run(['chmod', '+x', op.join(folder, compile_file)])
    cmd_run(['env', 'PATH={}'.format(path), './{}'.format(compile_file)],
            working_directory=folder, succeed=False)

def update_ssh():
    cmd_run(['cp', op.expanduser('~/.ssh/id_rsa'), op.expanduser('~/')])
    cmd_run(['chmod', '777', op.expanduser('~/id_rsa')])

def get_mpi_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

import fcntl

def acquireLock():
    ''' acquire exclusive lock file access '''
    locked_file_descriptor = open('/tmp/lockfile.LOCK', 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor

def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()

def ensure_ssh_server_running():
    ssh_folder = op.expanduser('~/.ssh')
    cmd_run(['ls', ssh_folder], succeed=False)

    y = cmd_run(['service', 'ssh', 'status'],
            succeed=True, return_output=True)
    y = y.decode()
    if 'sshd is not running' in y:
        cmd_run(['service', 'ssh', 'restart'],
                succeed=True)
    elif 'active (running)' in y or 'sshd is running' in y:
        logging.info('ssh is running. ignore to start')
    else:
        logging.info('unknown ssh server satus: \n{}'.format(y))

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

def terminate_monitoring_process(p):
    p.terminate()
    p.join()

def launch_monitoring_process():
    from multiprocessing import Process
    p = Process(target=monitor)
    p.start()
    return p

def wrap_all(code_zip, code_root,
        data_folder, model_folder, output_folder, command):
    cmd_run(['ibstatus'])
    cmd_run(['grep', 'Port', '/etc/ssh/sshd_config'])
    cmd_run(['nvidia-smi'])
    cmd_run(['ifconfig'])
    cmd_run(['df', '-h'])
    cmd_run(['ls', '/dev'])

    if get_mpi_rank() == 0:
        p = launch_monitoring_process()

    lock_fd = acquireLock()
    logging.info('got the lock')
    # start the ssh server
    if not op.isdir(code_root):
        ensure_directory(code_root)
        # set up the code, models, output under qd
        logging.info('unzipping {}'.format(code_zip))
        unzip(code_zip, code_root)
        cmd_run(['rm', 'data'], code_root)
        cmd_run(['rm', 'models'], code_root)
        cmd_run(['rm', 'output'], code_root)
        cmd_run(['ln', '-s',
            data_folder,
            op.join(code_root, 'data')])
        cmd_run(['ln',
            '-s',
            model_folder,
            op.join(code_root, 'models')
            ])

        cmd_run(['mkdir', '-p', output_folder])

        cmd_run(['chmod', 'a+rw',
            output_folder], succeed=False)

        cmd_run(['ln', '-s', output_folder, op.join(code_root, 'output')])

        # compile the source code
        compile_qd(code_root)
    ensure_ssh_server_running()
    releaseLock(lock_fd)

    # after the code is compiled, let's check the lib version
    cmd_run(['pip', 'freeze'])
    logging.info(command)
    if type(command) is str:
        command = list(command.split(' '))

    if len(command) > 0:
        cmd_run(command, working_directory=code_root,
                succeed=True)

    if get_mpi_rank() == 0:
        terminate_monitoring_process(p)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--model_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--command', type=str)
    args = parser.parse_args()
    return args

def run():
    from pprint import pformat
    logging.info(pformat(sys.argv))
    dict_param = vars(parse_args())

    logging.info('start')

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

if __name__ == '__main__':
    init_logging()
    #ensure_ssh_server_running()
    run()
