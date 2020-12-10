import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler


class CompositeRankAwareSampler(Sampler):
    def __init__(self, dataset):
        from qd.qd_common import get_mpi_size
        from qd.qd_common import get_mpi_rank
        # the returned value should be a list of integer
        source_list = dataset.get_composite_source_idx()

        num_source = max(source_list) + 1
        self.world_size = get_mpi_size()
        self.rank = get_mpi_rank()
        assert (num_source % self.world_size) == 0

        num_source_each = num_source // self.world_size
        start = self.rank * num_source_each
        end = start + num_source_each
        self.all_idx = [i for i, idx_s in enumerate(source_list)
                        if idx_s >= start and idx_s < end]
        self.curr_idx = 0

    def __iter__(self):
        while True:
            if self.curr_idx >= len(self.all_idx):
                self.curr_idx -= len(self.all_idx)
            yield self.all_idx[self.curr_idx]
            self.curr_idx += 1

    def __len__(self):
        raise ValueError('should not be called')

class InfiniteSampler(Sampler):
    def __init__(self, sample_size, shuffle_at_init=True):
        from qd.qd_common import get_mpi_size
        from qd.qd_common import get_mpi_rank

        self.sample_size = sample_size
        self.world_size = get_mpi_size()
        self.rank = get_mpi_rank()
        self.idx = self.rank
        self.shuffle_at_init = shuffle_at_init

    def __iter__(self):
        if self.shuffle_at_init:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(6)
            indices = torch.randperm(self.sample_size, generator=g).tolist()
        else:
            indices = torch.arange(self.sample_size).tolist()
        while True:
            # consider the case where self.sample_size < self.world_size
            yield indices[self.idx % self.sample_size]
            self.idx += self.world_size
            if self.idx >= self.sample_size:
                self.idx -= self.sample_size
            # be careful to re-shuffle the list since not all ranks will be at
            # this point.

    def __len__(self):
        raise ValueError('should not be called')

    def set_iter(self, i):
        self.idx = i * self.world_size + self.rank

    def set_epoch(self, epoch):
        raise ValueError('the order will be fixed')

class InfiniteBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        raise ValueError('infinite')

class MaxIterBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        self.batch_sampler.sampler.set_iter(
            start_iter * self.batch_sampler.batch_size)

    def __iter__(self):
        iteration = self.start_iter
        for batch in self.batch_sampler:
            iteration += 1
            if iteration > self.num_iterations:
                break
            yield batch

    def __len__(self):
        return self.num_iterations

