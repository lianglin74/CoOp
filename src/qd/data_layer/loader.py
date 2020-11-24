import torch
from qd.tsv_io import TSVFile
from qd.data_layer.transform import FeatureDecoder
from qd.data_layer.dataset import DatasetPlusTransform
from qd.data_layer.samplers import OrderedSplitSampler


def create_feature_loader(fname, batch_size=16):
    tsv = TSVFile(fname)
    transform = FeatureDecoder()
    dataset = DatasetPlusTransform(tsv, transform)
    sampler = OrderedSplitSampler(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4,
        sampler=sampler)
    return loader

def create_data_loader(dataset, transform, batch_size=16):
    dataset = DatasetPlusTransform(dataset, transform)
    sampler = OrderedSplitSampler(len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8,
        sampler=sampler)
    return loader
