#!/usr/bin/python
import os
import sys
import logging
import zipfile
import subprocess as sp
import os.path as op
import torch.distributed as dist
import time
import torch

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

def cmd_run(cmd, working_directory='./', succeed=True):
    e = os.environ.copy()
    e['PYTHONPATH'] = '/app/caffe/python:{}'.format(e.get('PYTHONPATH', ''))
    logging.info('start to cmd run: {}'.format(' '.join(map(str, cmd))))
    for c in cmd:
        logging.info(c)
    p = sp.Popen(cmd, stdin=sp.PIPE,
            cwd=working_directory,
            env=e)
    p.communicate()
    if succeed:
        assert p.returncode == 0

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        if not os.path.exists(path) and not op.islink(path):
            os.makedirs(path)

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
    cmd_run(['sudo', 'env', 'PATH={}'.format(path), './{}'.format(compile_file)],
            working_directory=folder, succeed=False)

def update_ssh():
    cmd_run(['cp', op.expanduser('~/.ssh/id_rsa'), op.expanduser('~/')])
    cmd_run(['sudo', 'chmod', '777', op.expanduser('~/id_rsa')])

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

def wrap_all(code_zip,
        code_root,
        input_folder,
        command,
        output_folder):

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
            op.join(input_folder, 'data', 'qd_data'),
            op.join(code_root, 'data')])
        cmd_run(['ln',
            '-s',
            op.join(input_folder, 'work', 'qd_models'),
            op.join(code_root, 'models')
            ])
        qd_output = op.join(input_folder, 'work', 'qd_output')

        cmd_run(['mkdir', '-p', qd_output])

        cmd_run(['sudo', 'chmod', 'a+rw',
            qd_output], succeed=False)

        cmd_run(['ln', '-s', qd_output, op.join(code_root, 'output')])

        # compile the source code
        compile_qd(code_root)
    releaseLock(lock_fd)

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

def run_in_philly():
    '''
    extra_params: command
    '''
    logging.info('start')
    _, input_folder, log_folder, model_folder = sys.argv[:4]
    extra_params = sys.argv[4:]
    logging.info("input data = {}".format(input_folder))
    logging.info('log folder = {}'.format(log_folder))
    logging.info('model folder = {}'.format(model_folder))
    logging.info('extra param = {}'.format(' '.join(extra_params)))

    for k in os.environ:
        logging.info('{} = {}'.format(k, os.environ[k]))

    qd_root = op.join('/tmp', 'code', 'quickdetection')

    # usefull only in philly@ap. no hurt for other philly
    update_ssh()
    link_nltk()

    if len(extra_params) > 0 and extra_params[0].startswith('quickdetection'):
        qd_zip = op.join(input_folder, 'code',
                '{}.zip'.format(extra_params[0]))
        command = extra_params[1:]
    else:
        qd_zip = op.join(input_folder, 'code', 'quickdetection.zip')
        command = extra_params

    wrap_all(qd_zip,
            qd_root,
            input_folder,
            command,
            model_folder)

    # the permission should be changed because the output is there, but the
    # permission is for the docker job only and teh philly-fs cannot delete or
    # change it
    if get_mpi_rank():
        cmd_run(['sudo', 'chmod', '777',
            op.join(input_folder, 'work', 'qd_output'),
            '-R'], succeed=False)

def run_in_local():
    qd_zip = 'quickdetection.zip'
    qd_root = op.expanduser('~/code/tmp')
    input_folder = op.expanduser('~')
    output_folder = op.expanduser('~/tmp')

    wrap_all(qd_zip,
            qd_root,
            input_folder,
            ['ls'],
            output_folder)

if __name__ == '__main__':
    init_logging()

    run_in_philly()
    #run_in_local()

