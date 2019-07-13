from datetime import datetime
import numpy as np
import os
import os.path as op
import random

import torch
import torch.distributed as dist

from qd.qd_common import ensure_directory, load_list_file, write_to_yaml_file
from qd.qd_pytorch import get_master_node_ip, get_philly_mpi_hosts, get_aml_mpi_host_names, init_random_seed, save_parameters

def get_dist_url(init_method_type, dist_url_tcp_port=23456):
    if init_method_type == 'file':
        dist_file = op.abspath(op.join('output', 'dist_sync'))
        if not op.isfile(dist_file):
            ensure_directory(op.dirname(dist_file))
            open(dist_file, 'a').close()
        dist_url = 'file://' + dist_file
    elif init_method_type == 'tcp':
        dist_url = 'tcp://{}:{}'.format(get_master_node_ip(),
                dist_url_tcp_port)
    elif init_method_type == 'env':
        dist_url = 'env://'
    else:
        raise ValueError('unknown init_method_type = {}'.format(init_method_type))
    return dist_url

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
