import torch
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.qd_common import is_hvd_initialized
import torch.distributed as dist
import os
from qd.tsv_io import load_list_file
import os.path as op
from qd.qd_common import get_mpi_local_rank, get_mpi_local_size


def describe_tensor(t):
    return 'min/max/mean={:.2f}/{:.2f}/{:.2f}+-{:.2f}'.format(t.min(),
                                       t.max(),
                                       t.mean(),
                                       t.std())

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if get_mpi_size() == 1:
        return tensor
    if not is_hvd_initialized():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(get_mpi_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        import horovod as hvd
        output = hvd.torch.allgather(tensor)
        return output

def get_aml_mpi_host_names():
    return os.environ['AZ_BATCH_HOST_LIST'].split(',')

def get_philly_mpi_hosts():
    return load_list_file(op.expanduser('~/mpi-hosts'))

def get_master_node_ip():
    if 'AZ_BATCH_HOST_LIST' in os.environ:
        return get_aml_mpi_host_names()[0]
    elif 'AZ_BATCHAI_JOB_MASTER_NODE_IP' in os.environ:
        return os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
    else:
        return get_philly_mpi_hosts()[0]

def ensure_init_process_group(device_id=None, port=12345):
    if get_mpi_size() == 1:
        return
    if not dist.is_initialized():
        dist_url = 'tcp://{}:{}'.format(get_master_node_ip(),
                port)
        init_param = {'backend': 'nccl',
                'init_method': dist_url,
                'rank': get_mpi_rank(),
                'world_size': get_mpi_size()}
        if device_id is None:
            device_id = get_mpi_local_rank()
        torch.cuda.set_device(device_id)
        dist.init_process_group(**init_param)

def calc_num_node_in_grad_fn(grad_fn):
    result = 0
    if grad_fn is not None:
        result += 1
        if hasattr(grad_fn, 'next_functions'):
            for f in grad_fn.next_functions:
                result += calc_num_node_in_grad_fn(f)
    return result

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    use_hvd = is_hvd_initialized()
    if not use_hvd:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        dist.barrier()
    else:
        from qd.qd_common import get_mpi_size
        if get_mpi_size() > 1:
            import horovod.torch as hvd
            hvd.allreduce(torch.tensor(0), name='barrier')

def all_gather_grad_curr(x):
    if get_mpi_size() == 1:
        return x
    else:
        with torch.no_grad():
            all_x = [torch.zeros_like(x) for _ in range(get_mpi_size())]
            # note, all_rep should be treated as constent, which means no grad
            # will be propagated back through all_rep
            torch.distributed.all_gather(all_x, x)
        all_x[get_mpi_rank()] = x
        return torch.cat(all_x, dim=0)


