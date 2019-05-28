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

def cmd_run(cmd, working_directory='./', succeed=False):
    e = os.environ.copy()
    e['PYTHONPATH'] = '/app/caffe/python:{}'.format(e.get('PYTHONPATH', ''))
    logging.info('start to cmd run: {}'.format(' '.join(map(str, cmd))))
    for c in cmd:
        logging.info(c)
    try:
        p = sp.Popen(cmd, stdin=sp.PIPE,
                cwd=working_directory,
                env=e)
        p.communicate()
        if succeed:
            assert p.returncode == 0
    except:
        if succeed:
            raise
            assert p.returncode == 0

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        if not os.path.exists(path) and not op.islink(path):
            os.makedirs(path)

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

def wrap_all(code_zip, code_root,
        data_folder, model_folder, output_folder, command):
    lock_fd = acquireLock()
    if not op.isdir(code_root):
        cmd_run(['grep', 'Port', '/etc/ssh/sshd_config'])
        cmd_run(['ifconfig'])
        cmd_run(['df', '-h'])
        cmd_run(['nvidia-smi'])

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
    releaseLock(lock_fd)

    if type(command) is str:
        command = list(command.split(' '))

    if len(command) > 0:
        cmd_run(command, working_directory=code_root)

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

def run_in_philly():
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

    # the permission should be changed because the output is there, but the
    # permission is for the docker job only and teh philly-fs cannot delete or
    # change it
    if get_mpi_rank() == 0:
        cmd_run(['chmod', '777',
            dict_param['output_folder'],
            '-R'], succeed=False)

if __name__ == '__main__':
    init_logging()

    run_in_philly()