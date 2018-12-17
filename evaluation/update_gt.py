import argparse
import logging
import os

import _init_paths
from evaluation.analyze_task import analyze_verify_box_task
from evaluation.eval_utils import GroundTruthConfig, merge_gt, process_prediction_to_verify, tune_threshold_for_target, add_config_baseline
from evaluation.generate_task import generate_task_files
from evaluation.uhrs import UhrsTaskManager
from evaluation.utils import write_to_file, list_files_in_dir, ensure_dir_empty
from scripts.qd_common import init_logging

parser = argparse.ArgumentParser(description='Update ground truth for new detection')
parser.add_argument('source', type=str,
                    help='the baseline name to be verified')
parser.add_argument('task', choices=['VerifyImage', 'VerifyBox'],
                    help='choose from VerifyBox (for detection results) and VerifyImage (for tagging results)')
parser.add_argument('--datasets', default=None, nargs='+',
                    help='datasets to be evaluated, default is all datasets in config')
parser.add_argument('--config', default='./groundtruth/config.yaml', type=str,
                    help='''path to yaml config file of dataset ground truth and baselines,
                    default is ./prediction/config.yaml''')
parser.add_argument('--num_judges', default=5, type=int,
                    help='''number of judges required for each question''')
parser.add_argument('--iou_threshold', default=0.5, type=float,
                    help='IoU threshold for bounding boxes matching, default is 0.5')
parser.add_argument('--displayname', default='', type=str,
                    help='path to display name file')
parser.add_argument('--labelmap', default='', type=str,
                    help='''path to labelmap file, only classes included in the labelmap
                    will be evaluated. Default is None, all classes will be evaluated''')
parser.add_argument('--taskdir', default='./tasks/', type=str,
                    help='''path to working directory, used for uploading,
                    downloading, logging, etc''')
parser.add_argument('--honeypot', default='./honeypot/voc20_easy_gt.txt',
                    help='''path to the honey pot label file, each line is a
                    json dict of a positive ground truth label, including keys:
                    objects_to_find, image_url, bboxes''')
parser.add_argument('--add_baseline', action="store_true",
                    help='add the verified result to baselines')
parser.add_argument('--tune_threshold', default='', type=str,
                    help='''path to TSV file of classes(col 0) and target thresholds(col 1)''')


def update_gt(args):
    # parse args
    if isinstance(args, list) or isinstance(args, str):
        args = parser.parse_args(args)
    if args.datasets is None:
        cfg = GroundTruthConfig(args.config)
        args.datasets = cfg.datasets()

    NEG_IOU = 0.95  # IoU threshold with wrong box to be treated as wrong
    MERGE_IOU = 0.8   # IoU threshold to merge into existing ground truth

    # TODO: configure task group basing on task type and judge resource (crowdsource, vendor, internal)
    task_group = "vendor_verify_box"
    source = args.source
    hp_file = args.honeypot
    task_type = args.task

    for dataset_name in args.datasets:
        task_dir = os.path.join(args.taskdir, source, dataset_name)
        label_file = os.path.join(task_dir, "eval_label.tsv")
        res_file = os.path.join(task_dir, "eval_result.tsv")
        task_upload_dir = os.path.join(task_dir, "upload")
        task_download_dir = os.path.join(task_dir, "download")
        task_id_name_log = os.path.join(task_dir, "task_id_name_log")

        uhrs_client = UhrsTaskManager(task_id_name_log)
        ensure_dir_empty(task_dir)
        logging.info("Updating for dataset: {}".format(dataset_name))
        num_to_verify = process_prediction_to_verify(args.config, dataset_name, source, task_type,
                                    label_file, args.iou_threshold, NEG_IOU,
                                    include_labelmap=args.labelmap)
        if num_to_verify == 0:
            continue
        ensure_dir_empty(task_upload_dir)
        generate_task_files(task_type, label_file, hp_file,
                            os.path.join(task_upload_dir, source))
        uhrs_client.upload_tasks_from_folder(task_group, task_upload_dir,
                                             prefix=source, num_judges=args.num_judges)
        ensure_dir_empty(task_download_dir)

        round_count = 0
        while True:
            round_count += 1
            uhrs_client.wait_until_task_finish(task_group)
            uhrs_client.download_tasks_to_folder(task_group, task_download_dir)
            download_files = list_files_in_dir(task_download_dir)
            rejudge_filename = "rejudge_{}.tsv".format(round_count)
            num_rejudge = analyze_verify_box_task(
                download_files, "uhrs", res_file,
                os.path.join(task_upload_dir, rejudge_filename),
                worker_quality_file=os.path.join(task_dir, 'all_workers.tsv'),
                min_num_judges_per_hit=args.num_judges-1)
            if num_rejudge > 5 and round_count < 8:
                uhrs_client.upload_tasks_from_folder(
                    task_group, task_upload_dir, prefix=rejudge_filename,
                    num_judges=1)
            else:
                break

        merge_iou = 0 if task_type=="VerifyImage" else MERGE_IOU
        merge_gt(dataset_name, args.config, [res_file], merge_iou)


def main(args):
    update_gt(args)

    if args.tune_threshold:
        thres_dict = {}
        for dataset in args.datasets:
            res_file = os.path.join(args.res_folder, "{}.{}.tsv".format(dataset.lower(), args.source))
            tmp = tune_threshold_for_target(args.config, dataset, res_file, args.iou_threshold, args.tune_threshold)
            # choose the max threshold
            for label in tmp:
                if label not in thres_dict or thres_dict[label] < tmp[label]:
                    thres_dict[label] = tmp[label]
        outfile = os.path.join(os.path.split(args.config)[0], "threshold.{}.tsv".format(args.source))
        write_to_file([[k, thres_dict[k]] for k in thres_dict], outfile)

    if args.add_baseline:
        for dataset in args.datasets:
            fdict = {"result": os.path.join(args.res_folder, "{}.{}.tsv".format(dataset.lower(), args.source))}
            thres_file = os.path.join(os.path.split(args.config)[0], "threshold.{}.tsv".format(args.source))
            if os.path.isfile(thres_file):
                fdict["threshold"] = thres_file
            add_config_baseline(args.config, dataset, args.source, fdict)


if __name__ == "__main__":
    init_logging()
    args = parser.parse_args()
    main(args)
