import argparse
import os

from analyze_task import analyze_verify_box_task
from eval_utils import merge_gt, process_prediction_to_verify, tune_threshold_for_target, add_config_baseline
from generate_task import generate_verify_box_task
from uhrs import UhrsTaskManager
from utils import write_to_file
import _init_paths
from qd_common import init_logging

parser = argparse.ArgumentParser(description='Update ground truth for new detection')
parser.add_argument('source', type=str,
                    help='the detection source name')
parser.add_argument('res_folder', type=str,
                    help='''path to the prediciton reulsts folder, result files
                    should be named as [dataset].[source].tsv''')
parser.add_argument('--dataset', default=["MIT1K", "Instagram"], nargs='+',
                    help='datasets to be evaluated, default is MIT1K and Instagram')
parser.add_argument('--gt', default='./groundtruth/config.yaml', type=str,
                    help='''path to yaml config file in the ground truth folder,
                    default is ./groundtruth/config.yaml''')
parser.add_argument('--iou_threshold', default=0.5, type=float,
                    help='IoU threshold for bounding boxes, default is 0.5')
parser.add_argument('--conf_threshold', default=0.5, type=float,
                    help='''confidence threshold for prediction results,
                    default is 0.5''')
parser.add_argument('--displayname', default='', type=str,
                    help='path to display name file')
parser.add_argument('--labelmap', default='', type=str,
                    help='''path to labelmap file, only classes included in the labelmap
                    will be evaluated. Default is None, all classes will be evaluated''')
parser.add_argument('--task', default='./tasks/', type=str,
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


def list_files_in_dir(dirpath):
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath)
            if os.path.isfile(os.path.join(dirpath, f))]


def ensure_dir_empty(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    else:
        if len(os.listdir(dirpath)) > 0:
            raise Exception("{} is not empty".format(dirpath))


def update_gt(args):
    task_hitapp = "verify_box_group"
    source = args.source
    hp_file = args.honeypot
    task_dir = os.path.join(args.task, source)
    label_file = os.path.join(task_dir, "eval_label.tsv")
    task_upload_dir = os.path.join(task_dir, "upload")
    task_download_dir = os.path.join(task_dir, "download")
    task_id_name_log = os.path.join(task_dir, "task_id_name_log")

    uhrs_client = UhrsTaskManager(task_id_name_log)

    ensure_dir_empty(task_dir)
    files_to_verify = []
    for dataset in args.dataset:
        files_to_verify.append({
            "dataset": dataset, "source": source,
            "result": "{}.{}.tsv".format(dataset.lower(), source),
            "display": args.displayname,
            "conf_threshold": args.conf_threshold
        })
    num_to_verify = process_prediction_to_verify(args.gt, args.res_folder, files_to_verify,
                            label_file, args.iou_threshold, 0.97, include_labelmap=args.labelmap)
    if num_to_verify == 0:
        return

    ensure_dir_empty(task_upload_dir)
    generate_verify_box_task(label_file, hp_file,
                             os.path.join(task_upload_dir, source))
    uhrs_client.upload_tasks_from_folder(task_hitapp, task_upload_dir,
                                         prefix=source)
    ensure_dir_empty(task_download_dir)

    res_file = os.path.join(task_dir, "eval_result.tsv")

    round_count = 0
    while True:
        round_count += 1
        uhrs_client.wait_until_task_finish(task_hitapp)
        uhrs_client.download_tasks_to_folder(task_hitapp, task_download_dir)
        download_files = list_files_in_dir(task_download_dir)
        rejudge_filename = "rejudge_{}.tsv".format(round_count)
        num_rejudge = analyze_verify_box_task(
            download_files, "uhrs", res_file,
            os.path.join(task_upload_dir, rejudge_filename),
            os.path.join(task_dir, 'all_workers.tsv'))
        if num_rejudge > 5 and round_count < 8:
            uhrs_client.upload_tasks_from_folder(
                task_hitapp, task_upload_dir, prefix=rejudge_filename,
                num_judges=1)
        else:
            break

    merge_gt(args.gt, [res_file], args.iou_threshold)


def main(args):
    update_gt(args)

    if args.tune_threshold:
        thres_dict = {}
        for dataset in args.dataset:
            res_file = os.path.join(args.res_folder, "{}.{}.tsv".format(dataset.lower(), args.source))
            tmp = tune_threshold_for_target(args.gt, dataset, res_file, args.iou_threshold, args.tune_threshold)
            # choose the max threshold
            for label in tmp:
                if label not in thres_dict or thres_dict[label] < tmp[label]:
                    thres_dict[label] = tmp[label]
        outfile = os.path.join(os.path.split(args.gt)[0], "threshold.{}.tsv".format(args.source))
        write_to_file([[k, thres_dict[k]] for k in thres_dict], outfile)

    if args.add_baseline:
        for dataset in args.dataset:
            fdict = {"result": os.path.join(args.res_folder, "{}.{}.tsv".format(dataset.lower(), args.source))}
            thres_file = os.path.join(os.path.split(args.gt)[0], "threshold.{}.tsv".format(args.source))
            if os.path.isfile(thres_file):
                fdict["threshold"] = thres_file
            add_config_baseline(args.gt, dataset, args.source, fdict)


if __name__ == "__main__":
    init_logging()
    args = parser.parse_args()
    main(args)
