import torch
from torch import nn
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.qd_common import is_hvd_initialized
import torch.distributed as dist
import os
from qd.tsv_io import load_list_file
import os.path as op
from torch import nn
import logging
from qd.qd_common import get_mpi_local_rank, get_mpi_local_size
from qd.qd_common import ensure_directory


def boxlist_to_list_dict(box_list,
                         label_id_to_label,
                         extra=0):
    # box_list here is the maskrcnn-benchmark style of BoxList
    # use to_rects, which handles the case where 'labels' not in the field
    box_list = box_list.convert("xyxy")
    if len(box_list) == 0:
        return []
    scores = box_list.get_field("scores").tolist()
    labels = box_list.get_field("labels").tolist()
    extra_key_values = [(k, v) for k, v in box_list.extra_fields.items()
            if k not in ['scores', 'labels']]
    boxes = box_list.bbox
    rects = []
    for i, (box, score, label_id) in enumerate(zip(boxes, scores, labels)):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        r = [top_left[0],
             top_left[1],
             bottom_right[0] + extra,
             bottom_right[1] + extra,
             ]
        rect = {'class': label_id_to_label[label_id], 'conf': score, 'rect': r}

        for k, v in extra_key_values:
            f = v[i]
            if isinstance(f, torch.Tensor):
                f = f.squeeze()
                if len(f.shape) == 1:
                    # just to make it a list of float
                    rect[k] = f.tolist()
                elif len(f.shape) == 0:
                    rect[k] = float(f)
                else:
                    raise ValueError('unknown Tensor {}'.format(
                        ','.join(map(str, f.shape))))
            else:
                raise ValueError('unknown {}'.format(type(f)))
        rects.append(rect)
    return rects

def freeze_parameters(modules):
    if isinstance(modules, nn.Module):
        modules = [modules]
    for m in modules:
        for n, p in m.named_parameters():
            p.requires_grad = False
            if hasattr(m, 'name_from_root'):
                logging.info('freeze param: {}.{}'.format(
                    m.name_from_root,
                    n))
            else:
                logging.info('freeze param: {}'.format(
                    n))
        m.eval()

def get_torch_version_info():
    return {
        'version': torch.__version__,
        'cuda': torch.version.cuda,
        'nccl': torch.cuda.nccl.version(),
        'cudnn': torch.backends.cudnn.version(),
    }

def update_bn_momentum(model, bn_momentum):
    from collections import defaultdict
    type_to_count = defaultdict(int)
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = bn_momentum
            type_to_count[module.__class__.__name__] += 1
    from pprint import pformat
    logging.info(pformat(dict(type_to_count)))

def freeze_bn_(model):
    num_freezed = 0
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            num_freezed += 1
            logging.info('freeze {}'.format(n))
            m.eval()
    logging.info('#freeze = {}'.format(num_freezed))

def attach_module_name_(model):
    for n, m in model.named_modules():
        m.name_from_root = n

def query_modules_by_name(module, part_name):
    return [m for n, m in module.named_modules() if n.endswith(part_name)]

def replace_module_by_name(module, module_part_name, creator_func):
    attach_module_name_(module)
    return replace_module(module,
                   lambda m: m.name_from_root.endswith(module_part_name),
                   creator_func)

def replace_module(module, condition_func, creator_func):
    module_output = module
    if condition_func(module):
        module_output = creator_func(module)
    for name, child in module.named_children():
        child = replace_module(child, condition_func, creator_func)
        module_output.add_module(name, child)
    del module
    return module_output

def torch_save(t, f):
    ensure_directory(op.dirname(f))
    tmp_f = f + '.tmp'
    torch.save(t, tmp_f)
    os.rename(tmp_f, f)

def torch_load(filename):
    return torch.load(filename, map_location=lambda storage, loc: storage)

def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    elif isinstance(x, dict):
        x = {k: recursive_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        x = [recursive_to_device(y, device) for y in x]
    return x

def distributed_sinkhorn(sim, eps, niters):
    # sim: B * K. different gpus has different rows.
    world_size = get_mpi_size()

    sim = sim / eps
    max_sim = sim.max()
    max_reduce_(max_sim)
    sim -= max_sim

    Q = sim.exp()
    sum_q = Q.sum()
    sum_reduce_(sum_q)
    Q /= sum_q

    #Q = (sim / eps).exp().T
    #Q /= sum(Q)

    B, K = Q.shape
    device = sim.device
    c = torch.ones(K, device=device) / K
    r = torch.ones(B, device=device) / (B * world_size)
    #cap = 1e-5
    for _ in range(niters):
        u = Q.sum(dim=0)
        sum_reduce_(u)
        #u[u < cap & u > 0] = cap
        #u[u > -cap & u < 0] = -cap
        Q *= (c / (u + 1e-5)).unsqueeze(0)
        Q *= (r / (Q.sum(dim=1) + 1e-5)).unsqueeze(1)
    return (Q / (Q.sum(dim=1, keepdim=True) + 1e-5))

def sinkhorn_correct_sum(sim, eps, niters):
    # the function is used to solve the following problem. max (sim * Q).sum()
    # + eps * entropy(Q) with niters iterations. The constraint is uniform
    # distribution

    # here it is sim / eps rather than sim / -eps, since the cost matrix is
    # -sim, so based on the paper of lightspeed, it is -sim / -eps = sim /
    # eps, which is consistent with SwAV paper. The rotation of T is useless,
    # it looks like, but it is ok since the returned matrix is also transposed.

    # reduce the chance of being small here
    sim = sim / eps
    max_sim = sim.max()
    sim -= max_sim
    Q = sim.exp()
    Q /= Q.sum()

    #Q = (sim / eps).exp().T
    #Q /= sum(Q)

    B, K = Q.shape
    device = sim.device
    c, r = torch.ones(K, device=device) / K, torch.ones(B, device=device) / B
    #cap = 1e-5
    for _ in range(niters):
        u = Q.sum(dim=0)
        #u[u < cap & u > 0] = cap
        #u[u > -cap & u < 0] = -cap
        Q *= (c / (u + 1e-5)).unsqueeze(0)
        Q *= (r / (Q.sum(dim=1) + 1e-5)).unsqueeze(1)
    return (Q / (Q.sum(dim=1, keepdim=True) + 1e-5))

def sinkhorn2(sim, eps, niters):
    # this function fixes the Q/Q.sum() issue. all others are the same with
    # sinkhorn. This function is only used for parity check of
    # sinkhorn_correct_sum, where no transpose is performed.
    # use sinkhorn. The difference is to remove transpose at the beginning and
    # the end.
    # the function is used to solve the following problem. max (sim * Q).sum()
    # + eps * entropy(Q) with niters iterations. The constraint is uniform
    # distribution

    # here it is sim / eps rather than sim / -eps, since the cost matrix is
    # -sim, so based on the paper of lightspeed, it is -sim / -eps = sim /
    # eps, which is consistent with SwAV paper. The rotation of T is useless,
    # it looks like, but it is ok since the returned matrix is also transposed.

    # reduce the chance of being small here
    sim = sim / eps
    max_sim = sim.max()
    sim -= max_sim
    Q = sim.exp().T
    Q /= Q.sum()

    #Q = (sim / eps).exp().T
    #Q /= sum(Q)

    K, B = Q.shape
    device = sim.device
    r, c = torch.ones(K, device=device) / K, torch.ones(B, device=device) / B
    #cap = 1e-5
    for _ in range(niters):
        u = Q.sum(dim=1)
        #u[u < cap & u > 0] = cap
        #u[u > -cap & u < 0] = -cap
        Q *= (r / (u + 1e-5)).unsqueeze(1)
        Q *= (c / (Q.sum(dim=0) + 1e-5)).unsqueeze(0)
    return (Q / (Q.sum(dim=0, keepdim=True) + 1e-5)).T

def sinkhorn(sim, eps, niters):
    # this function has some issues, where we should use Q/Q.sum() but we used
    # Q/=sum(Q)
    # the function is used to solve the following problem. max (sim * Q).sum()
    # + eps * entropy(Q) with niters iterations. The constraint is uniform
    # distribution

    # here it is sim / eps rather than sim / -eps, since the cost matrix is
    # -sim, so based on the paper of lightspeed, it is -sim / -eps = sim /
    # eps, which is consistent with SwAV paper. The rotation of T is useless,
    # it looks like, but it is ok since the returned matrix is also transposed.

    # reduce the chance of being small here
    sim = sim / eps
    max_sim = sim.max()
    sim -= max_sim
    Q = sim.exp().T
    Q /= sum(Q)

    #Q = (sim / eps).exp().T
    #Q /= sum(Q)

    K, B = Q.shape
    device = sim.device
    r, c = torch.ones(K, device=device) / K, torch.ones(B, device=device) / B
    #cap = 1e-5
    for _ in range(niters):
        u = Q.sum(dim=1)
        #u[u < cap & u > 0] = cap
        #u[u > -cap & u < 0] = -cap
        Q *= (r / (u + 1e-5)).unsqueeze(1)
        Q *= (c / (Q.sum(dim=0) + 1e-5)).unsqueeze(0)
    return (Q / (Q.sum(dim=0, keepdim=True) + 1e-5)).T

def describe_tensor(t, num_dec=2):
    t = t.float()
    if t.numel() == 1:
        return 'value={:.2f}'.format(float(t))
    format_str = 'min/max/mean={{:.{0}f}}/{{:.{0}f}}/{{:.{0}f}}+-{{:.{0}f}}'.format(
        num_dec)
    return format_str.format(t.min(),
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
    elif 'MASTER_IP' in os.environ:
        return os.environ['MASTER_IP']
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

def sum_reduce_(x):
    if get_mpi_size() > 1:
        torch.distributed.all_reduce(x)

def max_reduce_(x):
    if get_mpi_size() > 1:
        torch.distributed.all_reduce(x, torch.distributed.ReduceOp.MAX)



