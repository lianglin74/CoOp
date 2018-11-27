import argparse
import os
import shutil
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.models as models

from ..utils.data import get_testdata_loader
from ..utils.averagemeter import AverageMeter
from ..lib.qd_common import tsv_writer

import base64

parser = argparse.ArgumentParser(description='PyTorch feature extraction')
# necessary inputs
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--test-size', default=224, type=int,
                    help='test crop size (default: 224)')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to save prediction result')

def load_model(args):
    checkpoint = torch.load(args.model)

    # create model
    arch = checkpoint['arch']
    num_classes = checkpoint['state_dict']['module.fc.weight'].size(0)
    model = models.__dict__[arch](num_classes=num_classes)

    # load model weights
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print "=> loaded checkpoint '{}' (epoch {})".format(args.model, checkpoint['epoch'])

    # remove last layer
    model = torch.nn.Sequential(*list(model.module.children())[:-1])
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # switch to evaluate mode
    model.eval()

    return model

def main(args):
    if isinstance(args, dict):
        args = argparse.Namespace()
    elif isinstance(args, list) or isinstance(args, str):
        args = parser.parse_args(args)

    # Data loading code
    val_loader = get_testdata_loader(args)

    # Load model
    model = load_model(args)

    def gen_tsv():
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        tic = time.time()
        with torch.no_grad():
            end = time.time()
            for i, (input, lines) in enumerate(val_loader):
                data_time.update(time.time() - end)

                # compute output
                output = model(input)
                output = output.cpu()

                for f, cols in zip(output, lines):
                    s = base64.b64encode(f.view(-1).numpy())
                    cols.append(s)
                    yield cols

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    speed = 10 * args.batch_size / (time.time() - tic)
                    info_str = 'Test: [{0}/{1}]\t' \
                                'Speed: {speed:.2f} samples/sec\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                i, len(val_loader), speed=speed, batch_time=batch_time,
                                data_time=data_time)
                    print info_str
                    tic = time.time()

    tsv_writer(gen_tsv(), args.output)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
