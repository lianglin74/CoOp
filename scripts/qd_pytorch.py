from tqdm import tqdm
from datetime import datetime
from qd_common import write_to_yaml_file
from deteval import deteval_iter
from tsv_io import tsv_writer, tsv_reader
import simplejson as json
from qd_common import ensure_directory
import os.path as op
import copy
from pprint import pformat
from qd_common import init_logging
import logging
import os
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tsv_io import TSVFile
from qd_common import img_from_base64, load_from_yaml_file
from tsv_io import TSVDataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch import nn
try:
    from itertools import izip as zip
except:
    # python 3
    pass

import json
from yoloinit import ncc2_train2, calc_epsilon, EPS
import time
import re
import glob
from qd_common import worth_create
from process_tsv import load_key_rects
from qd_common import read_to_buffer
from qd_common import write_to_file
from qd_common import load_list_file


class ListCollator(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        return list(zip(*batch))

class OrderedSampler(torch.utils.data.Sampler):
    def __init__(self, idx):
        super(OrderedSampler, self).__init__(None)
        self.idx = list(idx)

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


class CompositeTSVFile():
    def __init__(self, list_file, seq_file, cache_policy=False):
        self.seq_file = seq_file
        self.list_file = list_file
        self.cache_policy = cache_policy
        self.initialized = False
        self.initialize()

    def __getitem__(self, index):
        idx_source, idx_row = self.seq[index]
        return self.tsvs[idx_source].seek(idx_row)

    def __len__(self):
        return len(self.seq)

    def initialize(self):
        '''
        this function has to be called in init function if cache_policy is
        enabled. Thus, let's always call it in init funciton to make it simple.
        '''
        if self.initialized:
            return
        self.seq = [(int(idx_source), int(idx_row)) for idx_source, idx_row in
                tsv_reader(self.seq_file)]
        self.tsvs = [TSVFile(f, self.cache_policy) for f in load_list_file(self.list_file)]
        self.initialized = True

class TSVSplit(Dataset):
    def __init__(self, data, split, version=0, cache_policy=None):
        dataset = TSVDataset(data)
        self.is_composite = False
        if op.isfile(dataset.get_data(split)):
            self.tsv = TSVFile(dataset.get_data(split),
                    cache_policy)
            self.label_tsv = TSVFile(dataset.get_data(split, 'label',
                version=version), cache_policy)
        else:
            splitX = split + 'X'
            list_file = dataset.get_data(splitX)
            seq_file = dataset.get_shuffle_file(split)
            self.tsv = CompositeTSVFile(list_file, seq_file, cache_policy)
            list_file = dataset.get_data(splitX, 'label')
            self.label_tsv = CompositeTSVFile(list_file, seq_file,
                    cache_policy)
            assert version == 0

    def __getitem__(self, index):
        _, __, str_image = self.tsv[index]
        key, str_label = self.label_tsv[index]
        return key, str_label, str_image

    def __len__(self):
        return len(self.tsv)

class TSVSplitImage(TSVSplit):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None):
        super(TSVSplitImage, self).__init__(data, split, version, 
                cache_policy)
        self.transform = transform
        # load the label map
        dataset = TSVDataset(data)
        if labelmap is None:
            labelmap = load_list_file(dataset.get_labelmap_file())
        elif type(labelmap) is str:
            labelmap = load_list_file(labelmap)
        assert type(labelmap) is list
        self.label_to_idx = {l: i for i, l in enumerate(labelmap)}

    def __getitem__(self, index):
        key, str_label, str_im = super(TSVSplitImage, self).__getitem__(index)
        img = img_from_base64(str_im)
        if self.transform is not None:
            img = self.transform(img)
        label = self._tsvcol_to_label(str_label)
        # before it returns 2 elements. now it is 3. some code might be broken,
        # adjust it accoordingly. The reason is that we want to save the
        # prediction result for each row. The key is required to identify which
        # image it is. 
        return img, label, key

    def get_num_pos_labels(self):
        return len(self.label_to_idx)

    def _tsvcol_to_label(self, col):
        try:
            idx = int(col)
        except:
            info = json.loads(col)
            assert len(info) == 1
            idx = self.label_to_idx[info[0]['class']]
        label = torch.from_numpy(np.array(idx, dtype=np.int))
        return label

class TSVSplitImageMultiLabel(TSVSplitImage):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None):
        super(TSVSplitImageMultiLabel, self).__init__(data, split, version,
                transform, cache_policy, labelmap)

    def _tsvcol_to_label(self, col):
        rects = json.loads(col)
        label = torch.zeros(len(self.label_to_idx))
        if type(rects) is int:
            label[rects] = 1
        else:
            all_cls = set(r['class'] for r in rects)
            # if it starts with -, it means negative
            all_idx = [self.label_to_idx[c] for c in all_cls if 
                    not c.startswith('-') and c in self.label_to_idx]
            label[all_idx] = 1
        return label

class TSVSplitImageMultiLabelNeg(TSVSplitImage):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None):
        super(TSVSplitImageMultiLabelNeg, self).__init__(data, split, version,
                transform, cache_policy, labelmap)

    def _tsvcol_to_label(self, col):
        rects = json.loads(col)
        label = torch.zeros(len(self.label_to_idx))
        label[:] = -1
        if type(rects) is int:
            label[rects] = 1
        else:
            all_cls = set(r['class'] for r in rects)
            # if it starts with -, it means negative
            all_idx = [self.label_to_idx[c] for c in all_cls if 
                    not c.startswith('-')]
            label[all_idx] = 1
            all_idx = [self.label_to_idx[c[1:]] for c in all_cls if 
                    c.startswith('-')]
            label[all_idx] = 0
        return label

class TorchTSVDataset(Dataset):
    def __init__(self, tsv_file, transform=None, labelmap=None,
            tsv_label_file=None):
        '''
        deprecated. using TSVSplit
        '''
        self.tsv = TSVFile(tsv_file)
        if tsv_label_file:
            self.label_tsv = TSVFile(tsv_label_file)
        else:
            self.label_tsv = None
        self.transform = transform
        self.label_to_idx = {l: i for i, l in enumerate(labelmap)}

    def __getitem__(self, index):
        row = self.tsv.seek(index)
        img = img_from_base64(row[-1])
        if self.transform is not None:
            img = self.transform(img)
        if self.label_tsv:
            row = self.label_tsv.seek(index)
        label = self._tsvcol_to_label(row[1])
        # before it returns 2 elements. now it is 3. some code might be broken,
        # adjust it accoordingly. The reason is that we want to save the
        # prediction result for each row. The key is required to identify which
        # image it is. 
        return img, label, row[0]

    def _tsvcol_to_label(self, col):
        try:
            idx = int(col)
        except:
            info = json.loads(col)
            assert len(info) == 1
            idx = self.label_to_idx[info[0]['class']]
        label = torch.from_numpy(np.array(idx, dtype=np.int))
        return label

    def __len__(self):
        logging.info('checking the number of images')
        return self.tsv.num_rows()

class TorchMultiLabelTSVDataset(TorchTSVDataset):
    def __init__(self, tsv_file, transform=None, labelmap=None,
            tsv_label_file=None):
        '''
        deprecated. using TSVSplitImageMultiLabel
        '''
        super(TorchMultiLabelTSVDataset, self).__init__(tsv_file, transform,
                labelmap, tsv_label_file)

    def _tsvcol_to_label(self, col):
        rects = json.loads(col)
        label = torch.zeros(len(self.label_to_idx))
        if type(rects) is int:
            label[rects] = 1
        else:
            all_cls = set(r['class'] for r in rects)
            # if it starts with -, it means negative
            all_idx = [self.label_to_idx[c] for c in all_cls if 
                    not c.startswith('-') and c in self.label_to_idx]
            label[all_idx] = 1
        return label

class FeatureExtractionModel(nn.Module):
    def __init__(self, original_model, target_idx):
        super(FeatureExtractionModel, self).__init__()
        self.feature = nn.Sequential(*list(original_model.children())[:target_idx])

    def forward(self, x):
        return self.feature(x)

def ncc2_train2_torch(X,Y, avgnorm2, cmax=None):
    '''
    X: num_sample x num_feature
    '''
    X = X.cpu().numpy().astype(float)
    Y = Y.cpu().numpy().astype(int).squeeze()

    tic = time.time()
    W, B = ncc2_train2(X, Y, avgnorm2.numpy(), cmax=cmax)
    logging.info('ncc2 takes: {}s'.format(time.time() - tic))

    return torch.from_numpy(W).cuda(), torch.from_numpy(B).cuda()

def calculate_ap_by_true_list(corrects, total):
    if total == 0:
        return 0
    corrects = corrects.cpu()
    total = total.cpu()
    precision = (1. * corrects.cumsum(dim=0)) / torch.range(1, len(corrects))
    return torch.sum(precision * corrects) / total 

class EfficientTagMetric(object):
    def __init__(self, num_bucket=100):
        self.num_bucket = 100
        self.count = None
        self.correct = None
        self.num_tag = None

    def update(self, pred, gt):
        '''
        pred: sample x number_of_tag: probability
        gt: sample x number_of_tag. 1/0
        '''
        pred = pred.cpu()
        gt = gt.cpu()
        if self.count is None:
            assert self.correct is None
            assert self.num_tag is None
            num_tag = gt.shape[1]
            self.count = torch.zeros((self.num_bucket, num_tag))
            self.correct = torch.zeros((self.num_bucket, num_tag))
            self.num_tag = num_tag
        idx = self.num_bucket - 1 - (pred * (self.num_bucket - 1)).type(torch.LongTensor)
        for t in range(self.num_tag):
            for i, bucket_idx in enumerate(idx[:, t]):
                self.count[bucket_idx, t] += 1
                self.correct[bucket_idx, t] += gt[i, t]

    def summarize(self):
        if self.num_tag is None:
            logging.info('no data available')
            return 0
        cum_correct = self.correct.cumsum(dim=0)
        cum_count = self.count.cumsum(dim=0)

        aps = torch.zeros(self.num_tag)
        for t in range(self.num_tag):
            all_p = torch.zeros(self.num_bucket)
            for i, (correct, count) in enumerate(zip(
                cum_correct[:, t], cum_count[:, t])):
                if count != 0:
                    all_p[i] = 1. * correct / count
            total = torch.sum(self.correct[:, t])
            if total == 0:
                ap = 0
            else:
                ap = torch.sum(all_p * self.correct[:, t]) / total
            aps[t] = ap
        return aps

class TagMetric(object):
    def __init__(self):
        self.all_pred = []
        self.all_gt = []

    def update(self, pred, gt):
        '''
        pred: sample x number_of_tag: probability
        gt: sample x number_of_tag. 1/0
        '''
        self.all_pred.append(pred)
        self.all_gt.append(gt)

    def summarize(self):
        pred = torch.cat(self.all_pred)
        gt = torch.cat(self.all_gt)
        _, idx = torch.sort(pred, 0, descending=True)
        num_tag = gt.shape[-1]
        aps = torch.zeros(num_tag)
        for t in range(num_tag):
            is_correct = gt[idx[:, t], t]
            ap = calculate_ap_by_true_list(is_correct.cpu(), torch.sum(is_correct))
            aps[t] = ap
        return aps

def create_transform(train=False):
    if not train:
        return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


def evaluate_tag(dataloader, model):
    m = EfficientTagMetric()
    #m = TagMetric()
    softmax_func = torch.nn.Softmax(dim=1)
    scanned_images = 0
    max_images = 100
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        output = model(inputs)
        output = softmax_func(output)
        m.update(output, labels)
        scanned_images += len(inputs)
        if scanned_images > max_images:
            break
    aps = m.summarize()
    logging.info('mean ap = {}'.format(torch.mean(aps)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def extract_feature(dataloader, model):
    all_X = []
    all_Y = []
    enough = False
    counts = None
    max_images = 100
    scanned_images = 0
    data_loader_time = AverageMeter()
    start_time = time.time()
    for i, (inputs, labels) in enumerate(dataloader):
        data_loader_time.update(time.time() - start_time)
        inputs = inputs.cuda()
        labels = labels.cuda()
        output = model(inputs)
        output = torch.squeeze(output)
        for o, l in zip(output, labels):
            for idx in torch.nonzero(l):
                all_X.append(o.data.cpu())
                all_Y.append(idx.data.cpu())
                if counts is None:
                    counts = torch.zeros(labels.shape[1])
                counts[idx] = counts[idx] + 1
        scanned_images += len(inputs)
        if (i % 100) == 0:
            logging.info('data loader time: {}s'.format(data_loader_time.avg))
        if scanned_images > max_images:
            logging.info('scanned maximum images')
            break
        if torch.min(counts) > 20:
            logging.info('number of images for each category is enough')
            break
        start_time = time.time()
    X = torch.stack(all_X, 0)
    Y = torch.stack(all_Y, 0)
    return X, Y

def transfer_learning():
    #m = Tmp()
    #m.children()
    #import ipdb;ipdb.set_trace(context=15)
    x = torch.zeros(2)
    import ipdb;ipdb.set_trace(context=15)
    torch.sum(x)

    data = 'Tag2KCoco'

    data_transform = create_transform()

    dataset = TSVDataset(data)
    labelmap = dataset.load_labelmap()
    label_to_idx = {l: i for i, l in enumerate(labelmap)}

    image_dataset = TorchMultiLabelTSVDataset(dataset.get_data('train'),
            label_to_idx, data_transform)

    dataloader = torch.utils.data.DataLoader(image_dataset, 
            batch_size=4,
            num_workers=4)
    
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    original_model = model

    base_w = model.fc.weight.data
    base_avgnorm2 = torch.mean(torch.sum(base_w * base_w, 1))

    model = FeatureExtractionModel(model, -1)
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    
    X, Y = extract_feature(dataloader, model)

    W, B = ncc2_train2_torch(X, Y, base_avgnorm2, len(label_to_idx))
    
    original_model.fc.weight = torch.nn.Parameter(
            W.type_as(original_model.fc.weight))
    original_model.fc.bias = torch.nn.Parameter(
            B.type_as(original_model.fc.bias))
    original_model = original_model.cuda()

    # test the accuracy on the training set
    evaluate_tag(dataloader, original_model)

def get_test_transform():
    normalize = get_data_normalize()
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])

def get_train_transform():
    normalize = get_data_normalize()
    totensor = transforms.ToTensor()
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    data_augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        color_jitter,
        transforms.RandomHorizontalFlip(),
        totensor,
        normalize, ])
    return data_augmentation

def get_data_normalize():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return normalize

class ModelLoss(nn.Module):
    def __init__(self, model, criterion):
        super(ModelLoss, self).__init__()
        self.module = model
        self.criterion = criterion

    def forward(self, data, target):
        out = self.module(data)
        loss = self.criterion(out, target)
        # make sure the length of the loss is the same with the dim 0. The
        # loss should be the sum of the result.
        N = data.shape[0]
        return loss.expand(N) / N

class MultiHotCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiHotCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        return multi_hot_cross_entropy(input, target)

class BCEWithLogitsNegLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsNegLoss, self).__init__()

    def forward(self, feature, target):
        return bce_with_logits_neg_loss(feature, target)

def bce_with_logits_neg_loss(feature, target):
    weight = torch.ones_like(target)
    weight[target == -1] = 0
    criterion = nn.BCEWithLogitsLoss(weight, reduce=False).cuda()
    loss = criterion(feature, target)
    return torch.sum(loss) / (torch.sum(weight) + 1)

def multi_hot_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    
    loss = torch.sum(-soft_targets * logsoftmax(pred)) / (torch.sum(soft_targets)+0.000001)

    return loss

def load_latest_parameters(folder):
    yaml_pattern = op.join(folder,
            'parameters_*.yaml')
    yaml_files = glob.glob(yaml_pattern)
    if len(yaml_files) == 0:
        return {}
    def parse_time(f):
        m = re.search('.*parameters_(.*)\.yaml', f) 
        t = datetime.strptime(m.group(1), '%Y_%m_%d_%H_%M_%S')
        return t
    times = [parse_time(f) for f in yaml_files]
    fts = [(f, t) for f, t in zip(yaml_files, times)]
    fts.sort(key=lambda x: x[1], reverse=True)
    yaml_file = fts[0][0]
    logging.info('using {}'.format(yaml_file))
    param = load_from_yaml_file(yaml_file)
    return param

def parse_epoch(s):
    s = op.basename(s)
    results = re.match('^model_iter_(.*)e\.pth.tar$', s)
    return int(results.groups()[0])

def save_parameters(param, folder):
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    write_to_yaml_file(param, op.join(folder, 
        'parameters_{}.yaml'.format(time_str)))
    # save the env parameters
    write_to_yaml_file(os.environ, op.join(folder, 
        'env_{}.yaml'.format(time_str)))

def torch_save(t, f):
    ensure_directory(op.dirname(f))
    torch.save(t, f)

def torch_load(filename):
    return torch.load(filename)

def get_philly_mpi_hosts():
    return load_list_file(op.expanduser('~/mpi-hosts'))

def ensure_create_evaluate_meta_file(evaluate_file):
    result = None
    simple_file = evaluate_file + '.map.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['map'] = \
                        result[size_type][thresh]['map']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

    simple_file = evaluate_file + '.class_ap.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            eval_result= read_to_buffer(evaluate_file)
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['class_ap'] = \
                        result[size_type][thresh]['class_ap']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)
    
    simple_file = '{}.prec.threshold.tsv'.format(evaluate_file)
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        _, max_key = max([(float(k), k) for k in result['overall']], 
                key=lambda x: x[0])
        class_thresh = result['overall'][max_key]['class_thresh']
        precision_ths = None
        for l in class_thresh:
            precision_ths = class_thresh[l].keys()
            break
        if precision_ths:
            for precision_th in precision_ths:
                sub_simple_file = '{}.{}.prec{}.threshold.tsv'.format(
                        evaluate_file, max_key, precision_th)
                def gen_rows():
                    for l in class_thresh:
                        th_recall = class_thresh[l].get(precision_th, [1, 0])
                        yield l, th_recall[0], th_recall[1]
                tsv_writer(gen_rows(), sub_simple_file)
        from_file = '{}.{}.prec{}.threshold.tsv'.format(evaluate_file, max_key, 0.5)
        if op.isfile(from_file) and worth_create(from_file, simple_file):
            from shutil import copyfile
            copyfile(from_file, simple_file)

class TorchTrain(object):
    def __init__(self, **kwargs):
        if 'load_parameter' in kwargs and kwargs['load_parameter']:
            assert 'full_expid' in kwargs
            kwargs_f = load_latest_parameters(op.join('output',
                kwargs['full_expid']))
            for k in kwargs_f:
                if k not in kwargs:
                    # we can overwrite the parameter in the parameter file
                    kwargs[k] = kwargs_f[k]

        self.kwargs = kwargs

        self._default = {'dist_backend': 'nccl', 
                'init_method_type': 'tcp',
                'log_step': 100,
                'evaluate_method': 'map',
                'test_split': 'test',
                'num_workers': 0,
                'test_normalize_module': 'softmax',
                'restore_snapshot_iter': -1,
                'ovthresh': [-1],
                'step_lr': 30,
                'base_lr': 0.1,
                'dataset_type': 'single',
                'batch_size': 256,
                'pretrained': False,
                'dist_url_tcp_port': 23456}

        self.data = kwargs.get('data', 'Unknown')
        self.net = kwargs.get('net', 'Unknown')
        self.expid = kwargs.get('expid', 'Unknown')

        self.full_expid = kwargs.get('full_expid', 
                '_'.join([self.data, self.net, self.expid]))
        self.output_folder = op.join('output', self.full_expid)
        self.test_data = kwargs.get('test_data', self.data)
        self.test_batch_size = kwargs.get('test_batch_size',
                self.batch_size)

        from qd_common import get_mpi_rank, get_mpi_size
        from qd_common import get_mpi_local_rank, get_mpi_local_size
        self.mpi_rank = get_mpi_rank()
        self.mpi_size= get_mpi_size()
        self.mpi_local_rank = get_mpi_local_rank()
        self.mpi_local_size = get_mpi_local_size()
        
        # the following two environements are used in init_dist_process if the
        # method is by env. we make sure the world is the same here
        if 'WORLD_SIZE' in os.environ:
            assert int(os.environ['WORLD_SIZE']) == self.mpi_size
        if 'RANK' in os.environ:
            assert int(os.environ['RANK']) == self.mpi_rank

        if self.mpi_size > 1:
            self.distributed = True
        else:
            self.distributed = False
            self.is_master = True
        # adapt the batch size based on the mpi_size
        self.is_master = self.mpi_rank == 0
        assert (self.batch_size % self.mpi_size) == 0
        self.batch_size = self.batch_size / self.mpi_size
        assert (self.test_batch_size % self.mpi_size) == 0
        self.test_batch_size = self.test_batch_size / self.mpi_size
        self.initialized = False

    def __getattr__(self, key):
        if key in self.kwargs:
            return self.kwargs[key]
        elif key in self._default:
            return self._default[key]
    
    def _ensure_initialized(self):
        if self._initialized:
            return
        
        self._setup_logging()
        # in torch 0.4, torch.randperm only supports cpu. if we set it as
        # cuda.Float by default, it will crash there
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #torch.set_default_tensor_type('torch.FloatTensor')

        if self.distributed:
            dist_file = op.abspath(op.join('output', 'dist_sync'))
            if not op.isfile(dist_file):
                ensure_directory(op.dirname(dist_file))
                open(dist_file, 'a').close()
            init_method_type = self.init_method_type
            if init_method_type == 'file':
                dist_url = 'file://' + dist_file
            elif init_method_type == 'tcp':
                dist_url = 'tcp://{}:{}'.format(get_philly_mpi_hosts()[0],
                        self.dist_url_tcp_port)
            elif init_method_type == 'env':
                dist_url = 'env://'
            else:
                raise ValueError('unknown init_method_type = {}'.format(init_method_type))
            from process_tsv import hash_sha1
            init_param = {'backend': self.dist_backend,
                    'init_method': dist_url,
                    'rank': self.mpi_rank,
                    'world_size': self.mpi_size}
            if init_method_type != 'env':
                init_param['group_name'] = hash_sha1(self.kwargs)
            # always set the device at the very beginning
            torch.cuda.set_device(self.mpi_local_rank)
            logging.info('init param: \n{}'.format(pformat(init_param)))
            from maskrcnn_benchmark.utils.comm import synchronize
            dist.init_process_group(**init_param)
            # we need to synchronise before exit here so that all workers can
            # finish init_process_group(). If not, worker A might exit the
            # whole program first, but worker B still needs to talk with A. In
            # that case, worker B will never return and will hang there
            synchronize()
            if self.mpi_rank != 0:
                # only print the fatal one
                logging.getLogger().setLevel(50)
        self._initialized = True

    def _get_dataset(self, data, split, stage, labelmap):
        if stage == 'train':
            transform = get_train_transform()
        else:
            assert stage == 'test'
            transform = get_test_transform()

        if self.dataset_type == 'single':
            return TSVSplitImage(data, split, 
                    version=0, 
                    transform=transform, 
                    labelmap=labelmap,
                    cache_policy=self.cache_policy)
        elif self.dataset_type == 'multi_hot':
            return TSVSplitImageMultiLabel(data, split, 
                    version=0, 
                    transform=transform, 
                    cache_policy=self.cache_policy, 
                    labelmap=labelmap)
        elif self.dataset_type == 'multi_hot_neg':
            return TSVSplitImageMultiLabelNeg(data, split, version=0,
                    transform=transform, 
                    cache_policy=self.cache_policy, 
                    labelmap=labelmap)
        else:
            raise ValueError('unknown {}'.format(self.dataset_type))

    def _get_criterion(self):
        if self.dataset_type == 'single':
            criterion = nn.CrossEntropyLoss().cuda()
        elif self.dataset_type == 'multi_hot':
            if self.loss_type == 'BCEWithLogitsLoss':
                criterion = nn.BCEWithLogitsLoss().cuda()
            elif self.loss_type == 'MultiHotCrossEntropyLoss':
                criterion = MultiHotCrossEntropyLoss()
            else:
                raise Exception('not support value {}'.format(self.loss_type))
        elif self.dataset_type == 'multi_hot_neg':
            assert self.loss_type == 'BCEWithLogitsNegLoss'
            criterion = BCEWithLogitsNegLoss()
        return criterion

    def _get_checkpoint_file(self, epoch=None, iteration=None):
        if epoch is None and iteration is None:
            assert self.max_epoch is not None or self.max_iter is not None
            return self._get_checkpoint_file(self.max_epoch, self.max_iter)

        if iteration is not None:
            assert epoch is None
            return op.join(self.output_folder, 'snapshot',
                'model_iter_{}.pth.tar'.format(iteration))
        else:
            assert epoch is not None
            return op.join(self.output_folder, 'snapshot',
                    'model_iter_{}e.pth.tar'.format(epoch))
    
    def _get_model(self, pretrained, num_class):
        model = models.__dict__[self.net](pretrained=pretrained)
        if model.fc.weight.shape[0] != num_class:
            model.fc = nn.Linear(model.fc.weight.shape[1], num_class)
            torch.nn.init.xavier_uniform_(model.fc.weight)

        return model

    def _data_parallel_wrap(self, model):
        if self.distributed:
            model.cuda()
            if self.mpi_local_size > 1:
                model = torch.nn.parallel.DistributedDataParallel(model, 
                        device_ids=[self.mpi_local_rank])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model).cuda()
        return model

    def _get_train_data_loader(self):
        train_dataset = self._get_dataset(self.data, 'train', 
                stage='train', labelmap=self.train_dataset.load_labelmap())
        
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            self.train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=(self.train_sampler==None),
            num_workers=self.num_workers, pin_memory=True,
            sampler=self.train_sampler)

        return train_dataset, train_loader

    def _get_test_data_loader(self, test_data, test_split, labelmap):
        test_dataset = self._get_dataset(test_data, test_split, stage='test',
                labelmap=labelmap)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)

        return test_dataset, test_loader

    def _save_parameters(self):
        save_parameters(self.kwargs, self.output_folder)
    
    def _setup_logging(self):
        if self.mpi_rank == 0:
            log_file = op.join(self.output_folder,
                'log_{}.txt'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
            ensure_directory(op.dirname(log_file))
            file_handle = logging.FileHandler(log_file)
            logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
            file_handle.setFormatter(fmt=logger_fmt)
            
            import sys
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logger_fmt
            ch.setFormatter(formatter)

            root = logging.getLogger()
            root.handlers = []
            root.setLevel(logging.INFO)
            root.addHandler(file_handle)
            root.addHandler(ch)
    
    def get_latest_checkpoint(self):
        all_snapshot = glob.glob(op.join(self.output_folder, 'snapshot',
            'model_iter_*e.pth.tar'))
        if len(all_snapshot) == 0:
            return
        snapshot_epochs = [(s, parse_epoch(s)) for s in all_snapshot]
        s, _ = max(snapshot_epochs, key=lambda x: x[1])
        return s

    def ensure_train(self):
        self._ensure_initialized()
        
        # we need this folder, normally. just create a folder here
        ensure_directory(op.join(self.output_folder, 'snapshot'))

        last_model_file = self._get_checkpoint_file(self.max_epoch, self.max_iter)
        if op.isfile(last_model_file) and not self.force_train:
            logging.info('skip to train')
            return 

        self._save_parameters()
        ensure_directory(self.output_folder)

        logging.info(pformat(self.kwargs))
        logging.info('torch version = {}'.format(torch.__version__))

        return self.train()

    def init_model(self, model):
        if self.init_from:
            if self.init_from['type'] == 'best_model':
                c = TorchTrain(full_expid=self.init_from['full_expid'],
                        load_parameter=True)
                model_file = c._get_checkpoint_file()
                logging.info('loading from {}'.format(model_file))
                checkpoint = torch_load(model_file)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError('unknown init type = {}'.format(self.init_from['type']))
    
    def train(self):
        train_dataset, train_loader = self._get_train_data_loader()
        num_class = train_dataset.get_num_pos_labels()
        
        model = self._get_model(self.pretrained, num_class)
        parallel_criterion = self.parallel_criterion
        if parallel_criterion:
            criterion = self._get_criterion()
            model = ModelLoss(model, criterion)
        model = self._data_parallel_wrap(model)

        self.init_model(model)
        
        optimizer = torch.optim.SGD(model.parameters(), self.base_lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)

        start_epoch = 0
        if self.restore_snapshot_iter == -1:
            last_checkpoint = self.get_latest_checkpoint()
            if last_checkpoint:
                checkpoint = torch.load(last_checkpoint)
                if parallel_criterion:
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']

        cudnn.benchmark = True
        max_iter_each_epoch = self.max_iter_each_epoch
        logging.info('start to train')
        for epoch in range(start_epoch, self.max_epoch):
            if self.distributed:
                # this is optional. need experiment to verify if it helps
                self.train_sampler.set_epoch(epoch)

            self._adjust_learning_rate(optimizer, epoch)

            self.train_epoch(train_loader, model, optimizer, epoch,
                    max_iter_each_epoch)
            
            if self.is_master:
                torch_save({
                    'epoch': epoch + 1,
                    'net': self.net,
                    'state_dict': model.state_dict() if not parallel_criterion else model.module.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, self._get_checkpoint_file(epoch + 1))

    def train_epoch(self, train_loader, model, optimizer, epoch,
            max_iter=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        # switch to train mode
        model.train()
        iter_start = time.time()
        log_start = iter_start
        log_step = self.log_step
        parallel_criterion = self.parallel_criterion
        if not parallel_criterion:
            criterion = self._get_criterion()
        for i, (input, target, _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - iter_start)
    
            target = target.cuda(non_blocking=True)
    
            # compute output
            if parallel_criterion:
                loss = model(input, target)
                loss = torch.mean(loss)
            else:
                output = model(input)
                loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))
    
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # measure elapsed time
            batch_time.update(time.time() - iter_start)
    
            if i % log_step == 0:
                speed = self.mpi_size * log_step * target.shape[0] / (time.time() - log_start)
                logging.info('Epoch: [{0}][{1}/{2}], ' \
                      'Speed: {speed:.2f} samples/sec, ' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}).'.format(
                       epoch, i, len(train_loader), speed=speed, 
                       batch_time=batch_time,
                       data_time=data_time, 
                       loss=losses))
                log_start = time.time()
                if max_iter and i > max_iter:
                    logging.info('break')
                    break
            iter_start = time.time()
    
    def _adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (0.1 ** (epoch // self.step_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _get_predict_file(self, model_file=None):
        if model_file is None:
            model_file = self._get_checkpoint_file(self.max_epoch,
                    self.max_iter)
        cc = [model_file, self.test_data, self.test_split]
        self.append_predict_param(cc)
        cc.append('predict')
        cc.append('tsv')
        return '.'.join(cc)

    def append_predict_param(self, cc):
        if self.test_normalize_module != 'softmax':
            cc.append('NormBy{}'.format(self.test_normalize_module))
        if self.predict_extract:
            cc.append('Extract{}'.format(self.predict_extract))

    def monitor_train(self):
        for e in range(self.max_epoch):
            predict_result_file = self.ensure_predict(e + 1)
            self.ensure_evaluate(predict_result_file)

    def ensure_predict(self, epoch=None, iteration=None):
        self._ensure_initialized()
        if epoch is None:
            epoch = self.max_epoch
        model_file = self._get_checkpoint_file(epoch=epoch, iteration=iteration)
        predict_result_file = self._get_predict_file(model_file)

        if not worth_create(model_file, predict_result_file) and not self.force_predict:
            logging.info('ignore to do prediction {}'.format(predict_result_file))
            return predict_result_file
        
        self.predict(model_file, predict_result_file)
        return predict_result_file

    def _get_test_normalize_module(self):
        if self.test_normalize_module == 'softmax':
            func = torch.nn.Softmax(dim=1)
        elif self.test_normalize_module == 'sigmoid':
            func = torch.nn.Sigmoid()
        else:
            raise Exception('unknown {}'.format(self.test_normalize_module))
        return func

    def predict(self, model_file, predict_result_file):
        train_dataset = TSVDataset(self.data)
        labelmap = train_dataset.load_labelmap()

        model = self._get_model(pretrained=False, num_class=len(labelmap))
        model = self._data_parallel_wrap(model)

        checkpoint = torch_load(model_file)

        model.load_state_dict(checkpoint['state_dict'])

        test_dataset, dataloader = self._get_test_data_loader(self.test_data,
                self.test_split, labelmap)
        
        softmax_func = self._get_test_normalize_module()

        # save top 50
        topk = 50
        model.eval()
        def gen_rows():
            pbar = tqdm(total=len(test_dataset))
            for i, (inputs, labels, keys) in enumerate(dataloader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                output = model(inputs)
                output = softmax_func(output)
                
                all_tops, all_top_indexes = output.topk(topk, dim=1, 
                        largest=True, sorted=False)

                for key, tops, top_indexes in zip(keys, all_tops, all_top_indexes):
                    all_tag = [{'class': labelmap[i], 'conf': float(t)} for t, i in 
                            zip(tops, top_indexes)]
                    yield key, json.dumps(all_tag)
                pbar.update(len(keys))

        tsv_writer(gen_rows(), predict_result_file)
        return predict_result_file
    
    def _get_evaluate_file(self, predict_file=None):
        if predict_file is None:
            predict_file = self._get_predict_file()
        assert predict_file.endswith('.tsv')
        cc = [op.splitext(predict_file)[0]]
        if self.evaluate_method != 'map':
            if self.evaluate_method is None:
                return
            cc.append(self.evaluate_method)
        if self.test_version:
            if self.test_version == -1:
                latest_version = TSVDataset(self.test_data).get_latest_version(
                        self.test_split, 'label')
                self.test_version = latest_version
                logging.info('inferred the latest version is {}'.format(
                    latest_version))
            cc.append('v{}'.format(self.test_version))
        cc.append('report')
        return '.'.join(cc)

    def ensure_evaluate(self, predict_file=None):
        if self.mpi_rank != 0:
            logging.info('skip because the rank {} != 0'.format(self.mpi_rank))
            return
        self._ensure_initialized()
        if not predict_file:
            model_file = self._get_checkpoint_file(self.max_epoch,
                    self.max_iter)
            predict_file = self._get_predict_file(model_file)
        evaluate_file = self._get_evaluate_file(predict_file)
        if evaluate_file is None:
            return

        if not worth_create(predict_file, evaluate_file) and not self.force_evaluate:
            logging.info('ignore {}'.format(evaluate_file))
        else:
            self.evaluate(predict_file, evaluate_file)

        # create index
        self.ensure_create_evaluate_meta_file(evaluate_file)
        return evaluate_file

    def ensure_create_evaluate_meta_file(self, evaluate_file):
        if self.evaluate_method == 'map':
            ensure_create_evaluate_meta_file(evaluate_file)

    def evaluate(self, predict_file, evaluate_file):
        dataset = TSVDataset(self.test_data)
        
        if self.evaluate_method == 'map':
            deteval_iter(
                    dataset.iter_data(self.test_split, 'label',
                        version=self.test_version),
                    predict_file,
                    report_file=evaluate_file,
                    ovthresh=self.ovthresh,
                    **self.kwargs)
        elif self.evaluate_method == 'coco_box':
            from cocoeval import convert_gt_to_cocoformat
            from cocoeval import convert_to_cocoformat
            from cocoeval import coco_eval_json
            gt_json = dataset.get_data(self.test_split, 'label.cocoformat',
                    version=self.test_version) + '.json'
            gt_iter = dataset.iter_data(self.test_split, 'label',
                        version=self.test_version)

            if not op.isfile(gt_json) or self.force_evaluate:
                convert_gt_to_cocoformat(gt_iter, gt_json)

            predict_json = predict_file + '.cocoformat.json'
            if worth_create(predict_file, predict_json) or self.force_evaluate:
                convert_to_cocoformat(predict_file, predict_json)

            result = coco_eval_json(predict_json, gt_json)

            write_to_yaml_file(result, evaluate_file)
        elif self.evaluate_method == 'top1':
            label_tsv_file = dataset.get_data(self.test_split, 'label',
                    self.test_version)
            top1 = evaluate_topk(predict_file, label_tsv_file)
            logging.info('top1 = {}'.format(top1))
            write_to_yaml_file({'top1': top1}, evaluate_file)
        else:
            logging.info('unknown evaluate method = {}'.format(self.evaluate_method))

class ModelPipeline(TorchTrain):
    pass

def evaluate_topk(predict_file, label_tsv_file):
    predicts = load_key_rects(tsv_reader(predict_file))
    predicts = {key: rects for key, rects in predicts}
    correct = 0
    total = 0
    for key, str_rects in tsv_reader(label_tsv_file):
        total = total + 1
        curr_predict = predicts.get(key, [])
        if len(curr_predict) == 0:
            continue
        curr_gt_rects = json.loads(str_rects)
        if type(curr_gt_rects) is int:
            # imagenet data
            curr_gt_rects = [{'class': str(curr_gt_rects)}]
        curr_pred_best = curr_predict[max(range(len(curr_predict)), key=lambda i: curr_predict[i]['conf'])]['class']
        if any(g['class'] == curr_pred_best for g in curr_gt_rects):
            correct = correct + 1
    return 1. * correct / total

def model_train(**kwargs):
    init_logging()
    
    t = TorchTrain(**kwargs)
    if kwargs.get('monitor_train_only'):
        t.monitor_train()
    else:
        t.ensure_train()
        t.ensure_predict()
        t.ensure_evaluate()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
    
if __name__ == '__main__':
    init_logging()

    from qd_common import parse_general_args
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
