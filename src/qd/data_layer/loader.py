import torch
from qd.tsv_io import TSVFile
from qd.data_layer.transform import FeatureDecoder
from qd.data_layer.dataset import DatasetPlusTransform
from qd.torch_common import recursive_to_device
from qd.data_layer.samplers import OrderedSplitSampler
import os, sys
from threading import Thread
from queue import Queue

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def create_feature_loader(fname, batch_size=16):
    tsv = TSVFile(fname)
    transform = FeatureDecoder()
    dataset = DatasetPlusTransform(tsv, transform)
    sampler = OrderedSplitSampler(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4,
        sampler=sampler)
    return loader

def create_data_loader(dataset,
                       transform,
                       batch_size=16,
                       num_workers=8):
    dataset = DatasetPlusTransform(dataset, transform)
    sampler = OrderedSplitSampler(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
    )
    return loader

class AsynchronousLoader(object):
    def __init__(self, dataloader, device='cuda', queue_size=2):
        self.device = device
        self.queue_size = queue_size

        # Use PyTorch's DataLoader for collating samples and stuff since it's nicely written and parallelrised
        self.dataloader = dataloader

        assert dataloader.pin_memory

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize = self.queue_size)

        self.idx = 0

    def __getattr__(self, key):
        return getattr(self.dataloader, key)

    def load_loop(self): # The loop that will load into the queue in the background
        for i, sample in enumerate(self.dataloader):
            self.queue.put(self.load_instance(sample))

    def load_instance(self, sample): # Recursive loading for each instance based on torch.utils.data.default_collate
        with torch.cuda.stream(self.load_stream):
            return recursive_to_device(sample, self.device, non_blocking=True)

    def __iter__(self):
        assert self.idx == 0, 'idx must be 0 at the beginning of __iter__. Are you trying to run the same instance more than once in parallel?'
        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        #self.worker.setDaemon(True)
        self.worker.start()
        return self

    def __next__(self):
        # If we've reached the number of batches to return or the queue is empty and the worker is dead then exit
        if (not self.worker.is_alive() and self.queue.empty()) or self.idx >= len(self.dataloader):
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        else: # Otherwise return the next batch
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out


    def __len__(self):
        return len(self.dataloader)

