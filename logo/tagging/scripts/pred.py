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
import torchvision.models as models

from ..utils.data import get_testdata_loader
from ..utils.averagemeter import AverageMeter


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# necessary inputs
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--labelmap', default='', type=str, metavar='PATH',
                    help='path to labelmap')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-k', '--topk', default=10, type=int,
                    metavar='K', help='top k result (default: 10)')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to save prediction result')


def load_labelmap(labelmap):
    with open(labelmap, 'r') as fp:
        return [line.strip().split('\t')[0] for line in fp]

def load_model(args):
    checkpoint = torch.load(args.model)
    arch = checkpoint['arch']
    model = models.__dict__[arch](num_classes=checkpoint['num_classes'])

    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(checkpoint['state_dict'])
    print "=> loaded checkpoint '{}' (epoch {})".format(args.model, checkpoint['epoch'])

    cudnn.benchmark = True

    # switch to evaluate mode
    model.eval()

    # load labelmap
    if args.labelmap:
        labelmap = load_labelmap(args.labelmap)
    elif 'labelmap' in checkpoint:
        labelmap = model['labelmap']
    else:
        labelmap = [str(i) for i in range(checkpoint['num_classes'])]

    return model, labelmap

def main(args):
    if isinstance(args, dict):
        args = argparse.Namespace()
    elif isinstance(args, list) or isinstance(args, str):
        args = parser.parse_args(args)

    # Data loading code
    val_loader = get_testdata_loader(args)

    # Load model and labelmap
    model, labelmap = load_model(args)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    tic = time.time()
    with torch.no_grad(), open(args.output, 'w') as fout:
        end = time.time()
        for i, (input, cols) in enumerate(val_loader):
            data_time.update(time.time() - end)

            # compute output
            output = model(input)
            output = output.cpu()

            _, pred_topk = output.topk(args.topk, dim=1, largest=True)

            for n in range(input.size(0)):
                pred = [(labelmap[k], output[n, k].item()) for k in pred_topk[n,:args.topk]]
                result = ';'.join(['{0}:{1:.5f}'.format(p[0], p[1]) for p in pred])
                fout.write('{0}\t{1}\n'.format('\t'.join([str(_) for _ in cols[n]]), result))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                speed = 100 * args.batch_size / (time.time() - tic)
                info_str = 'Test: [{0}/{1}]\t' \
                            'Speed: {speed:.2f} samples/sec\t' \
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            i, len(val_loader), speed=speed, batch_time=batch_time,
                            data_time=data_time)
                print info_str
                tic = time.time()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
