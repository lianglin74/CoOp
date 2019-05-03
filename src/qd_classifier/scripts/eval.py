import argparse
import os
import shutil
import time
import argparse
import oyaml as yaml

parser = argparse.ArgumentParser(description='Tagging evaluation')
# necessary inputs
parser.add_argument('--gt', metavar='PATH', default='groundtruth', type=str,
                    help='path to ground truth folder or yaml setting file')
parser.add_argument('--set', metavar='STR', default='', type=str,
                    help='dataset name')
parser.add_argument('--result', metavar='PATH', default='', type=str,
                    help='path to latest checkpoint (default: none)')


def tsv_reader(tsv_file_name):
    with open(tsv_file_name, 'r') as fp:
        for line in fp:
            yield [x.strip() for x in line.split('\t')]


def load_gt(gt_file):
    gt = {}
    for cols in tsv_reader(gt_file):
        imgkey, tags = cols[0:2]
        assert imgkey not in gt
        gt[imgkey] = set([tag for tag in tags.split(';')])
    return gt


def load_result(result_file):
    result = {}
    for cols in tsv_reader(result_file):
        imgkey, tags = cols[0:2]
        assert imgkey not in result
        result[imgkey] = set([tag.split(':')[0] for tag in tags.split(';')])
    return result


def eval_result(gt, result):
    ''' evaluate result for one tagger
    '''
    num_gt_tags = sum([len(tags) for _, tags in gt.iteritems()])
    num_result_tags = sum([len(tags) for _, tags in result.iteritems()])

    num_correct_tags = 0
    for murl, tags in result.iteritems():
        correct_tags = tags.intersection(gt[murl])
        num_correct_tags += len(correct_tags)

    precision = float(num_correct_tags) / num_result_tags * 100
    recall = float(num_correct_tags) / num_gt_tags * 100

    return num_gt_tags, num_result_tags, num_correct_tags, precision, recall


def eval_dataset(gt_root, dataset_name, dataset, new_result):
    print('==============================================================================')
    print('{:10}\t{}\t{}\t{}\t{}\t{}'.format(
        dataset_name,
        '#_gt_tags', '#_result_tags', '#_correct_tags',
        'prec', 'recall'))
    print('==============================================================================')

    gt_file = os.path.join(gt_root, dataset['groundtruth'])
    gt = load_gt(gt_file)

    # evaluate existing baselines
    for baseline, baseline_file in dataset['baselines'].iteritems():
        baseline_file = os.path.join(gt_root, baseline_file)
        result = load_result(baseline_file)
        num_gt_tags, num_result_tags, num_correct_tags, precision, recall = eval_result(gt, result)
        print('{:10}\t{}\t\t{}\t\t{}\t\t{:.2f}\t{:.2f}'.format(
            baseline,
            num_gt_tags, num_result_tags, num_correct_tags,
            precision, recall))

    # evaluate new result if provided
    if new_result:
        result = load_result(new_result)
        num_gt_tags, num_result_tags, num_correct_tags, precision, recall = eval_result(gt, result)
        print('{:10}\t{}\t\t{}\t\t{}\t\t{:.2f}\t{:.2f}'.format(
            'new result',
            num_gt_tags, num_result_tags, num_correct_tags,
            precision, recall))


def main(args):
    if isinstance(args, dict):
        args = argparse.Namespace()

    # provide an option to use other yaml cfg than the default one.
    if not args.gt.endswith('yaml'):
        args.gt = os.path.join(args.gt, 'groundtruth.yaml')

    gt_root = os.path.split(args.gt)[0]

    with open(args.gt, 'r') as fp:
        gt_cfg = yaml.load(fp)

    for dataset_name, dataset in gt_cfg.iteritems():
        if len(args.set) == 0 or args.set.lower() == dataset_name.lower():
            eval_dataset(gt_root, dataset_name, dataset, args.result)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
