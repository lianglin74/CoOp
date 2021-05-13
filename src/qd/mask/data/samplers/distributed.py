# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed.
# FIXME remove this once c10d fixes the bug it has
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        fixed_split: whether we re-shuffle after one epoch. The case when
        we set it as True is for large dataset training in AML, where the
        harddisk size is limitted. In this case, we make sure each GPU sees
        the same data so that every time
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
            length_divisible=1, fixed_split=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            from qd.qd_common import get_mpi_size
            num_replicas = get_mpi_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            from qd.qd_common import get_mpi_rank
            rank = get_mpi_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        if length_divisible > 1:
            import logging
            logging.info('before making divisible = {}'.format(self.num_samples))
            self.num_samples = ((self.num_samples + length_divisible - 1) //
                    length_divisible) * length_divisible
            logging.info('adjust to = {}'.format(self.num_samples))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.fixed_split = fixed_split

    def __iter__(self):
        dataset_len = len(self.dataset)
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(dataset_len, generator=g).tolist()
        else:
            indices = torch.arange(dataset_len).tolist()

        offset = self.num_samples * self.rank
        for i in range(self.num_samples):
            index = offset + i
            while index >= dataset_len:
                index -= dataset_len
            yield indices[index]

        # add extra samples to make it evenly divisible
        #assert (self.total_size - len(indices)) <= len(indices), 'not implemented'
        #indices += indices[: (self.total_size - len(indices))]
        #assert len(indices) == self.total_size

        ## subsample
        #offset = self.num_samples * self.rank
        #indices = indices[offset : offset + self.num_samples]
        #assert len(indices) == self.num_samples

        #return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        if not self.fixed_split:
            self.epoch = epoch
