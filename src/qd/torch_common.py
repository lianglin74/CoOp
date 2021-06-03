import json
from pprint import pformat
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
import math


def module_ual(m1, m2):
    name_to_m1 = dict(m1.named_modules())
    name_to_m2 = dict(m2.named_modules())
    if len(name_to_m1) != len(name_to_m2):
        return False


def hash_tensor_prime_simple(x):
    assert x.dim() == 2
    assert x.dtype == torch.long
    prime = 19
    idx = torch.arange(x.shape[1], device=x.device)
    coef = prime ** idx
    with torch.no_grad():
        x = x * coef[None, ]
        x = x.sum(dim=1)
    return x

def set_sigmoid_prob_prior_bias(bias, prior_prob):
    assert prior_prob > 0 and prior_prob != 1
    if prior_prob > 1:
        prior_prob = 1. / prior_prob
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    torch.nn.init.constant_(bias, bias_value)

def convert_single_label_to_one_hot_label(single_labels, label_size):
    if single_labels.dim() == 0:
        single_labels = single_labels[None]
    hot = torch.ones((len(single_labels), label_size),
                     dtype=single_labels.dtype) * -1
    bool_pos = single_labels != -1
    num_pos = bool_pos.sum()
    if num_pos > 0:
        pos_label = single_labels[bool_pos]
        pos_hot = torch.zeros((num_pos, label_size), dtype=single_labels.dtype)
        pos_hot.scatter_(1, pos_label.long()[:, None], 1)
        hot[bool_pos] = pos_hot
    return hot

def init_random_seed(random_seed):
    import random
    random.seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def boxlist_to_list_dict(box_list,
                         label_id_to_label,
                         extra=0,
                         encode_np_fields=[],
                         ):
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
                f = f.cpu()
                if k in encode_np_fields:
                    # this is for vlp, based on maskrcnn master branch
                    from qd.qd_common import encode_np
                    rect[k] = encode_np(f.numpy()).decode()
                elif len(f.shape) == 1:
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
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
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

def torch_save(t, f, **kwargs):
    ensure_directory(op.dirname(f))
    tmp_f = f + '.tmp'
    torch.save(t, tmp_f, **kwargs)
    os.rename(tmp_f, f)

def torch_load(filename):
    from qd.qd_common import get_user_name
    user_name = get_user_name()
    from qd.qd_common import acquireLock, releaseLock
    from qd.qd_common import hash_sha1
    lock_fd = acquireLock(op.join('/tmp',
        '{}_lock_{}'.format(user_name, hash_sha1(filename))))

    result = torch.load(filename, map_location=lambda storage, loc: storage)
    releaseLock(lock_fd)
    return result

def recursive_to_device(d, device, **kwargs):
    if isinstance(d, tuple) or isinstance(d, list):
        return [recursive_to_device(x, device, **kwargs) for x in d]
    elif isinstance(d, dict):
        return dict((k, recursive_to_device(v, device)) for k, v in d.items())
    elif isinstance(d, torch.Tensor) or hasattr(d, 'to'):
        #return d.to(device, non_blocking=True)
        return d.to(device, **kwargs)
    else:
        return d

# use recrusive_to_device, this naming is too short
def to(d, device):
    return recursive_to_device(d, device)

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

@torch.no_grad()
def describe_tensor(t, num_dec=2):
    t = t.float()
    if t.numel() == 1:
        return 'value={:.2f}'.format(float(t))
    format_str = \
        '{{}}, min/max/mean={{:.{0}e}}/{{:.{0}e}}/{{:.{0}e}}+-{{:.{0}e}}'.format(
            num_dec,
        )
    return format_str.format(
        t.shape,
        t.min(),
                             t.max(),
                             t.mean(),
                             t.std(),
                             )

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
        mpi_hosts = op.expanduser('~/mpi-hosts')
        if op.isfile(mpi_hosts):
            return load_list_file(mpi_hosts)[0]
        else:
            return 'localhost'

def ensure_init_process_group(device_id=None, port=12345):
    if not dist.is_initialized():
        dist_url = 'tcp://{}:{}'.format(get_master_node_ip(),
                port)
        from datetime import timedelta
        init_param = {
            'backend': 'nccl',
            'init_method': dist_url,
            'rank': get_mpi_rank(),
            'world_size': get_mpi_size(),
            'timeout': timedelta(days=10),
        }
        if device_id is None:
            device_id = get_mpi_local_rank()
        torch.cuda.set_device(device_id)
        logging.info(init_param)
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
    #use_hvd = is_hvd_initialized()
    use_hvd = False
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

def all_gather_curr_first_grad(x):
    if get_mpi_size() == 1:
        return x
    else:
        with torch.no_grad():
            all_x = [torch.zeros_like(x) for _ in range(get_mpi_size())]
            # note, all_rep should be treated as constent, which means no grad
            # will be propagated back through all_rep
            torch.distributed.all_gather(all_x, x)
        rank = get_mpi_rank()
        if rank == 0:
            all_x[0] = x
        else:
            all_x[rank] = all_x[0]
            all_x[0] = x
        return torch.cat(all_x, dim=0)

def sum_reduce_(x):
    if get_mpi_size() > 1:
        torch.distributed.all_reduce(x)

def sum_single_reduce_(x, dst):
    if get_mpi_size() > 1:
        torch.distributed.reduce(x, dst)

def max_reduce_(x):
    if get_mpi_size() > 1:
        torch.distributed.all_reduce(x, torch.distributed.ReduceOp.MAX)

class FlopCountModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, args, kwargs):
        y = self.model(*args, **kwargs)
        return y

def count_flops(model, *args, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    model = FlopCountModel(model)
    from fvcore.nn import flop_count
    y = flop_count(model, (args, kwargs,))
    return y

class SparseTensor(object):
    # torch.sparse.FloatTensor does not support pin_memory, which could be
    # required in dataloader with pin_memory=True
    def __init__(self, idx, value, size):
        assert idx.dtype == torch.long
        if idx.dim() == 1:
            idx = idx[None,]
        assert idx.dim() == 2
        assert idx.shape[0] == len(size)
        self.idx = idx
        self.value = value
        self.size = size
        assert isinstance(size, (tuple, list))

    def to_dense(self):
        return torch.sparse.FloatTensor(self.idx, self.value, self.size).to_dense()

    def to(self, *args, **kwargs):
        self.idx = self.idx.to(*args, **kwargs)
        self.value = self.value.to(*args, **kwargs)
        return self

    def __repr__(self):
        return 'SparseTensor(idx.shape={}, value.shape={}, size={})'.format(
            self.idx.shape,
            self.value.shape,
            self.size,
        )

    @property
    def shape(self):
        return self.size

    def pin_memory(self):
        return SparseTensor(self.idx.pin_memory(),
                            self.value.pin_memory(),
                            self.size)
    @staticmethod
    def stack(ts):
        return stack_sparse_tensor(ts)

def stack_sparse_tensor(ts):
    ts = list(ts)
    assert len(ts) > 0
    assert all(t.size == ts[0].size for t in ts[1:])
    assert all(isinstance(t, SparseTensor) for t in ts)
    all_idx = []
    for i, t in enumerate(ts):
        extra = torch.full((1, t.idx.shape[1]), i,
                           dtype=t.idx.dtype,
                           device=t.idx.device)
        cat = torch.cat((extra, t.idx))
        all_idx.append(cat)
    idx = torch.cat(all_idx, dim=1)
    value = torch.cat([t.value for t in ts])
    size = (len(ts),) + ts[0].size
    return SparseTensor(idx, value, size)

class IgnoreLastDimSparseTensor(object):
    # this is used to represent the target matrix for binary cross entropy.
    # Normally there might be lots of all -1 entries to ignore certain samples.
    # The positive entries are a few only and the matrix is more like a sparse
    # matrix. ignore_index here means the index array for the first dim to the
    # second last dim. For example of 2-d tensor, ignore_index should be a 1-d
    # array. If one element of ignore_index is 1, it means self[1, :] equals
    # -1.
    def __init__(self, idx, value, size, ignore_index):
        self.sparse = SparseTensor(idx, value, size)
        if len(size) == 2:
            if ignore_index.dim() == 1:
                ignore_index = ignore_index[None, :]
        assert ignore_index.dim() == 2
        assert ignore_index.shape[0] == len(size) - 1
        self.ignore_index = ignore_index

    def to_dense(self):
        mat = self.sparse.to_dense()
        if mat.dim() == 3:
            mat[self.ignore_index[0], self.ignore_index[1], :] = -1
        elif mat.dim() == 2:
            mat[self.ignore_index[0], :] = -1
        else:
            raise NotImplementedError(mat.dim())
        return mat

    def remove_ignore_2d_to_dense(self):
        # remove the ignored index and return the dense mat
        # note, we should not create a full matrix and then index it, in which
        # case there is large amount of memory footprint.
        assert len(self.sparse.shape) == 2
        relation = self.ignore_index[0].view((-1, 1)) < self.sparse.idx[0, :].view((1, -1))
        small_size = relation.sum(dim=0)
        idx = self.sparse.idx.clone()
        idx[0, :] = self.sparse.idx[0, :] - small_size
        size = self.sparse.shape
        size = (size[0] - len(self.ignore_index[0]), size[1])
        return SparseTensor(idx, self.sparse.value, size).to_dense()

    def reshape(self, size):
        # the last dimension should be kept
        assert size[-1] == self.sparse.shape[-1]
        assert len(size) == 2 and size[0] == -1, 'not supported'
        origin_shape = self.sparse.shape
        assert size[-1] == origin_shape[-1]

        idx_multiplier = origin_shape[1:-1] + (1,)
        idx_multiplier = torch.cumprod(torch.tensor(idx_multiplier[::-1],
                                                    device=self.ignore_index.device), dim=0)
        idx_multiplier = torch.flip(idx_multiplier, dims=(0,))[:, None]
        idx = (self.sparse.idx[:-1] * idx_multiplier).sum(dim=0, keepdim=True)
        idx = torch.cat((idx, self.sparse.idx[-1].view((1, -1))))

        ignore_index = (self.ignore_index * idx_multiplier).sum(
            dim=0, keepdim=True)
        return IgnoreLastDimSparseTensor(
            idx,
            self.sparse.value,
            (torch.prod(torch.tensor(origin_shape[:-1])).item(), origin_shape[-1]),
            ignore_index,
        )


    def keep_first_sum(self):
        # ignore the rows in ignore_index; sum up the value along the first
        # dimension
        idx = self.sparse.idx[0, :][None]
        value = self.sparse.value
        size = (self.sparse.shape[0],)
        return torch.sparse.FloatTensor(idx, value, size).to_dense()

    def to(self, *args, **kwargs):
        self.sparse = self.sparse.to(*args, **kwargs)
        self.ignore_index = self.ignore_index.to(*args, **kwargs)
        return self

    def __repr__(self):
        return ('IgnoreLastDimSparseTensor(idx.shape={}, '
                'value.shape={}, size={}, ignore_index.shape={})').format(
                    self.sparse.idx.shape,
                    self.sparse.value.shape,
                    self.sparse.size,
                    self.ignore_index.shape,
                )

    @property
    def shape(self):
        return self.sparse.size

    def pin_memory(self):
        return IgnoreLastDimSparseTensor(
            self.sparse.idx.pin_memory(),
            self.sparse.value.pin_memory(),
            self.sparse.size,
            self.ignore_index.pin_memory(),
        )

    @staticmethod
    def stack(ts):
        return stack_ignore_last_dim_sparse_tensor(ts)

def stack_ignore_last_dim_sparse_tensor(ts):
    sparse_tensors = [t.sparse for t in ts]
    stacked_sparse = stack_sparse_tensor(sparse_tensors)
    all_ignore = []
    for i, t in enumerate(ts):
        extra = torch.full((1, t.ignore_index.shape[1]), i,
                           dtype=t.ignore_index.dtype,
                           device=t.ignore_index.device)
        ignore = torch.cat((extra, t.ignore_index))
        all_ignore.append(ignore)
    ignore = torch.cat(all_ignore, dim=1)
    return IgnoreLastDimSparseTensor(
        stacked_sparse.idx,
        stacked_sparse.value,
        stacked_sparse.size,
        ignore,
    )

def set_seed(seed, n_gpu):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class InputAsDict(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
    def forward(self, data_dict):
        if isinstance(data_dict, torch.Tensor):
            im = data_dict
        else:
            im = data_dict['image']
        return self.module(im)

class BoxList(object):
    # compared with the boxlist in maskrcnn, we remove the image size and
    # To_REMOVE
    def __init__(self, bbox, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh", 'cxywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.mode = mode
        self.extra_fields = {}

    def to_list_of_dict(self):
        assert 'rect' not in self.extra_fields
        result = [{'rect': b} for b in self.bbox.tolist()]
        for k, v in self.extra_fields.items():
            if hasattr(v, 'tolist'):
                v = v.tolist()
            for r, sub_v in zip(result, v):
                r[k] = sub_v
        return result

    @staticmethod
    def create_empty_list(mode='xyxy'):
        bbox = torch.zeros((0, 4))
        return BoxList(bbox, mode)

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh", 'cxywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, mode=mode)
        elif mode == 'xywh':
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin, ymax - ymin), dim=-1
            )
            bbox = BoxList(bbox, mode=mode)
        elif mode == 'cxywh':
            bbox = torch.cat(
                ((xmin + xmax) / 2., (ymin + ymax) / 2, xmax - xmin, ymax - ymin), dim=-1
            )
            bbox = BoxList(bbox, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + w,
                ymin + h,
            )
        elif self.mode == 'cxywh':
            cx, cy, w, h = self.bbox.split(1, dim=-1)
            return (
                cx - w / 2.,
                cy - h / 2.,
                cx + w / 2.,
                cy + h / 2.,
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, ratios, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(ratio, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(ratios, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, size, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """

        image_width, image_height = size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        import PIL
        if method == PIL.Image.FLIP_LEFT_RIGHT:
            transposed_xmin = image_width - xmax
            transposed_xmax = image_width - xmin
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == PIL.Image.FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def to(self, device, **kwargs):
        bbox = BoxList(self.bbox.to(device, **kwargs), self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, size, remove_empty=True):
        xs = self.bbox.index_select(1, torch.tensor([0, 2], dtype=torch.int64, device=self.bbox.device))\
                 .clamp(min=0, max=size[0])
        ys = self.bbox.index_select(1, torch.tensor([1, 3], dtype=torch.int64, device=self.bbox.device))\
                 .clamp(min=0, max=size[1])
        self.bbox = torch.stack([xs, ys], dim=2).view(-1, 4)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def height(self):
        box = self.bbox
        if self.mode == "xyxy":
            height = box[:, 3] - box[:, 1]
        elif self.mode == "xywh":
            height = box[:, 3]
        else:
            raise RuntimeError("Unknown box mode")
        return height

    def width(self):
        box = self.bbox
        if self.mode == "xyxy":
            width = box[:, 2] - box[:, 0]
        elif self.mode == "xywh":
            width = box[:, 2]
        else:
            raise RuntimeError("Unknown box mode")
        return width

    def copy_with_fields(self, fields, skip_missing=True):
        bbox = BoxList(self.bbox, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "mode={})\n".format(self.mode)
        s += 'extra_fields={}'.format(self.extra_fields)
        return s

def load_model_state_ignore_mismatch(model, init_dict):
    real_init_dict = {}
    name_to_param = dict(model.named_parameters())
    name_to_param.update(dict(model.named_buffers()))

    def same_shape(a, b):
        return len(a.shape) == len(b.shape) and \
                all(x == y for x, y in zip(a.shape, b.shape))

    num_ignored = 0
    unique_key_in_init_dict = []
    keys_shape_mismatch = []
    for k in init_dict:
        if k in name_to_param:
            if same_shape(init_dict[k], name_to_param[k]):
                real_init_dict[k] = init_dict[k]
            else:
                logging.info('{} shape is not consistent, expected: {}; got '
                             '{}'.format(k, name_to_param[k].shape, init_dict[k].shape))
                keys_shape_mismatch.append(k)
        else:
            unique_key_in_init_dict.append(k)
            num_ignored = num_ignored + 1

    logging.info('unique keys in init dict = {}; total = {}'.format(
        pformat(unique_key_in_init_dict), len(unique_key_in_init_dict),
    ))
    result = model.load_state_dict(real_init_dict, strict=False)
    logging.info('unique key (not initialized) in current model = {}'.format(
        pformat(result.missing_keys),
    ))

    #logging.info('loaded key = {}'.format(
        #pformat(list(real_init_dict.keys()))))

class L2NormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter = 0

    def forward(self, x):
        verbose = (self.iter % 100) == 0
        self.iter += 1
        if verbose:
            from qd.torch_common import describe_tensor
            logging.info(describe_tensor(x))
        return torch.nn.functional.normalize(x)

def replace_fc_with_mlp_(model, num=1, mlp_bn=False,
                         with_l2=True):
    while not hasattr(model, 'fc'):
        model = model.module
    module_list = []
    dim_mlp = model.fc.weight.shape[1]
    for i in range(num):
        module_list.append(nn.Linear(dim_mlp, dim_mlp))
        if mlp_bn:
            module_list.append(nn.BatchNorm1d(dim_mlp))
        module_list.append(nn.ReLU())
    module_list.append(model.fc)
    if with_l2:
        module_list.append(L2NormModule())
    model.fc = nn.Sequential(*module_list)

def evaluate_topk(iter_pred_tsv, iter_label_tsv, key_name='class'):
    correct = 0
    total = 0
    for (key, str_rects), (key_pred, str_pred) in zip(iter_label_tsv, iter_pred_tsv):
        total = total + 1
        assert key == key_pred
        curr_predict = json.loads(str_pred)
        if len(curr_predict) == 0:
            continue
        curr_gt_rects = json.loads(str_rects)
        if type(curr_gt_rects) is int:
            # imagenet data
            curr_gt_rects = [{key_name: str(curr_gt_rects)}]
        curr_pred_best = curr_predict[max(range(len(curr_predict)), key=lambda
                                          i: curr_predict[i]['conf'])][key_name]
        if any(g[key_name] == str(curr_pred_best) for g in curr_gt_rects):
            correct = correct + 1
    return 1. * correct / total

def remove_prefix(model, prefix):
    out = {}
    for k, v in model.items():
        while k.startswith(prefix):
            k = k[len(prefix): ]
        out[k] = v
    return out

def adapt_position_encoding(model_file, before, patch_size, after, out_file,
                            suffix='image_encoder.module.pos_embed'):
    model = torch_load(model_file)
    if 'model' in model:
        model = model['model']
    keys = [k for k in model if k.endswith(suffix)]
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = before // patch_size
    assert (before % patch_size) == 0
    grid_after = after // patch_size
    assert (after % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]
    assert origin_pos_embed.shape[1] == grid_before * grid_before + 1

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size, mode='bicubic')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after * grid_after + 1, embed_dim)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed
    torch_save(model, out_file)

