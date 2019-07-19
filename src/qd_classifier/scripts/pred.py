import argparse
import collections
import json
import logging
import numpy as np
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.models as models

from qd.deteval import deteval_iter
from qd.tsv_io import tsv_reader, tsv_writer
from qd.qd_common import json_dump, read_to_buffer
from qd_classifier.utils.data import get_testdata_loader
from qd_classifier.utils.averagemeter import AverageMeter
from qd_classifier.utils.accuracy import get_accuracy_calculator
from qd_classifier.utils.save_model import load_from_checkpoint


def _predict(model, output_file, test_dataloader, labelmap, topk=5, evaluate=False):
    # switch to evaluate mode
    model.eval()

    # Calculate accuracy
    if evaluate:
        accuracy = get_accuracy_calculator(multi_label=False)
        label2idx = {l: i for i, l in enumerate(labelmap)}
    else:
        accuracy = None

    compute_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    tic = time.time()
    with torch.no_grad(), open(output_file, 'w') as fout:
        end = time.time()
        for i, (input, cols) in enumerate(test_dataloader):
            num_samples = input.size(0)
            data_time.update((time.time() - end) / num_samples * 1000)

            # compute output
            output = model(input)
            output = output.cpu()

            # measure elapsed time
            compute_time.update((time.time() - end) / num_samples * 1000)
            end = time.time()

            if accuracy:
                target = torch.from_numpy(np.array([[label2idx[json.loads(cols[n][1])["class"]]] for n in range(num_samples)], dtype=np.int))
                accuracy.calc(output, target)

            _, pred_topk = output.topk(topk, dim=1, largest=True)
            prob = F.softmax(output, dim=1)

            for n in range(num_samples):
                pred = [(labelmap[k], prob[n, k].item()) for k in pred_topk[n,:topk]]
                result = ';'.join(['{0}:{1:.5f}'.format(p[0], p[1]) for p in pred])
                fout.write('{0}\t{1}\n'.format('\t'.join([str(_) for _ in cols[n]]), result))

            if i % 100 == 0:
                speed = (i+1) * num_samples / (time.time() - tic)
                info_str = 'Test: [{0}/{1}]\t' \
                            'Speed: {speed:.2f} samples/sec\t' \
                            'Model Computing Time {compute_time.val:.3f} ({compute_time.avg:.3f})\t' \
                            'Data Loading Time {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            i, len(test_dataloader), speed=speed, compute_time=compute_time,
                            data_time=data_time)
                logging.info(info_str)
                tic = time.time()
    if accuracy:
        logging.info(accuracy.result_str())
        return accuracy.prec()
    else:
        return None

def _evaluate(tag_pred_file, dets_file, report_file, top_k=(1, 5)):
    """ Convert tag prediction result to regular format, i.e., image_key\tlist_of_bboxes
    Args:
        tag_pred_file: tsv file of imgkey, bbox, classification results (term:conf;)
        dets_file: tsv file of imgkey, bbox list
        report_file: evaluation report created by deteval
    """
    pred_dict = collections.defaultdict(list) # key: a list of bbox
    gt_dict = collections.defaultdict(list)
    correct_counts = None
    num_samples = 0
    for cols in tsv_reader(tag_pred_file):
        assert(len(cols) == 3)
        num_samples += 1
        key = cols[0]
        bbox = json.loads(cols[1])
        label_conf_pairs = [p.rsplit(':', 1) for p in cols[2].split(';')]
        gt_dict[key].append(bbox)

        if correct_counts is None:
            correct_counts = [0] * len(label_conf_pairs)
        gt_label = bbox["class"]
        for i, pair in enumerate(label_conf_pairs):
            if pair[0] == gt_label:
                correct_counts[i] += 1
                break

        # use top1 classification prediction as label
        bbox["class"] = label_conf_pairs[0][0]
        # use classification score as conf score
        bbox["conf"] = float(label_conf_pairs[0][1])
        pred_dict[key].append(bbox)

    tsv_writer([[k, json_dump(pred_dict[k])] for k in pred_dict], dets_file)

    if len(correct_counts) > 1:
        for i in range(1, len(correct_counts)):
            correct_counts[i] += correct_counts[i-1]
    eval_dict = {"top_{}_acc".format(k): float(correct_counts[i-1])/num_samples for k in top_k}

    deteval_iter(truth_iter=[[k, json.dumps(v)] for k, v in gt_dict.items()], dets=dets_file, report_file=report_file, force_evaluate=True)
    eval_res = json.loads(read_to_buffer(report_file))
    eval_dict["mAP"] = eval_res["overall"][str(0.5)]["map"]

    return eval_dict


def main(args):
    if isinstance(args, dict):
        args = argparse.Namespace()
    elif isinstance(args, list) or isinstance(args, str):
        args = parser.parse_args(args)

    # Data loading code
    test_dataloader = get_testdata_loader(args)

    # Load model and labelmap
    model, labelmap = load_from_checkpoint(args.model)
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()

    return _predict(model, args.output, test_dataloader, labelmap, evaluate=args.evaluate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # necessary inputs
    parser.add_argument('test_data', help='path to dataset yaml config')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-k', '--topk', default=10, type=int,
                        metavar='K', help='top k result (default: 10)')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to save prediction result')
    parser.add_argument('--input_size', default=224, type=int,
                        help='input image size')
    parser.add_argument('--enlarge_bbox', default=1.0, type=float,
                        help='make bbox larger (factor*width, factor*height)')
    parser.add_argument('--opencv', action='store_true',
                        help='use OpenCV transform to process image input')
    parser.add_argument('--evaluate', action='store_true',
                        help='calculate top k accuracy')
    parser.add_argument('--cache_policy', default=None, type=str,
                            help='use cache policy in TSVFile ')

    args = parser.parse_args()
    main(args)
