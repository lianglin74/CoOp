import torch
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.qd_common import is_hvd_initialized


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    use_hvd = is_hvd_initialized()
    import torch.distributed as dist
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


