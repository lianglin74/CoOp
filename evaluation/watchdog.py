import argparse
import logging
import os
import re
import sys
import time

import _init_paths
from evaluation.eval_utils import GroundTruthConfig
from evaluation.update_gt import update_gt
from scripts.qd_common import load_from_yaml_file, init_logging

class TaskStatus(object):
    def __init__(self, fpath):
        self.TASK_STATUS = ["new", "running", "completed", "fail"]
        self.cur_path = fpath
        self.rootpath = os.path.dirname(os.path.dirname(fpath))
        for status in self.TASK_STATUS:
            s = os.path.join(self.rootpath, status)
            if not os.path.isdir(s):
                os.mkdir(s)
        self._config_logging()

    def start(self):
        self._change_status("running")

    def complete(self):
        self._change_status("completed")

    def fail(self):
        self._change_status("fail")

    def _config_logging(self):
        log_dir = os.path.join(self.rootpath, "log")
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        fname = os.path.split(self.cur_path)[1]
        fname = fname.rsplit('.', 1)[0] + ".log"
        logpath = os.path.join(log_dir, fname)
        init_logging()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logpath, 'w')
        logger.addHandler(fh)
        # ch = logging.StreamHandler(sys.stdout)
        # logger.addHandler(ch)

    def _change_status(self, target_status):
        assert(target_status in self.TASK_STATUS)
        fdir, fname = os.path.split(self.cur_path)
        cur_status = os.path.split(fdir)[1]
        assert(cur_status in self.TASK_STATUS)
        if cur_status != target_status:
            logging.info("task status moves to: {}".format(target_status))
            dest_path = os.path.join(self.rootpath, target_status, fname)
            if os.path.isfile(dest_path):
                fname = str(time.time()) + '-' + fname
                dest_path = os.path.join(self.rootpath, target_status, fname)
            os.rename(self.cur_path, dest_path)
            self.cur_path = dest_path


def pick_task(task_dir):
    all_tasks = []
    for f in os.listdir(task_dir):
        fpath = os.path.join(task_dir, f)
        if f.endswith(".yaml") and os.path.isfile(fpath):
            try:
                task_config = load_from_yaml_file(fpath)
                score = task_config.get("priority", sys.maxint)
                all_tasks.append((score, fpath))
            except Exception as e:
                logging.info("invalid task file: {}".format(fpath))

    if len(all_tasks) == 0:
        return None
    all_tasks = sorted(all_tasks, key=lambda t: t[0])

    return all_tasks[0][1]

def main():
    rootpath = "//vigdgx02/raid_data/uhrs/"
    task_dir = "//vigdgx02/raid_data/uhrs/status/new/"

    # pick up one task from task_dir
    task_yaml = pick_task(task_dir)
    if task_yaml is None:
        return None

    task_status = TaskStatus(task_yaml)
    task_config = load_from_yaml_file(task_yaml)
    gt_config_file = task_config["gt_config"]
    if not os.path.isabs(gt_config_file):
        gt_config_file = os.path.join(rootpath, gt_config_file)
    model_name = task_config["model_name"]

    # add the new model to baselines
    gt_cfg = GroundTruthConfig(gt_config_file)
    dataset_list = []
    for pred_file in task_config["pred_files"]:
        cur_dataset = pred_file["dataset"]
        if cur_dataset in dataset_list:
            task_status.fail()
            # terminate
            raise ValueError("Do not submit multiple files for one dataset: {}. Split them into different tasks".format(cur_dataset))
        dataset_list.append(cur_dataset)
        if model_name not in gt_cfg.baselines(cur_dataset):
            gt_cfg.add_baseline(cur_dataset, model_name, pred_file["result"], pred_file["conf_threshold"])

    task_status.start()
    task_root = os.path.dirname(os.path.dirname(gt_config_file))

    def get_task_name(task_yaml):
        task_name = os.path.split(task_yaml)[1]
        task_name = task_name.rsplit('.', 1)[0]
        return re.sub('[^0-9a-zA-Z_]+', '', task_name)

    uhrs_task_dir = os.path.join(task_root, "tasks", get_task_name(task_yaml))
    hp_dir = os.path.join(task_root, "honeypot")
    hp_files = [f for f in os.listdir(hp_dir) if f.endswith(".txt")]
    hp_file = os.path.join(hp_dir, hp_files[0])

    # update_gt ongoing
    try:
        for dataset in dataset_list:
            args = [model_name, task_config["task_type"],
                    "--datasets", dataset,
                    "--config", gt_config_file,
                    "--taskdir", uhrs_task_dir, "--honeypot", hp_file,
                    "--priority", repr(2000)]  # HACK: make higher priority to skip the long waiting queue
            logging.info("update ground truth with arguments: {}".format(str(args)))
            update_gt(args)
    except Exception as e:
        task_status.fail()
        raise e

    # finished updating gt
    task_status.complete()
    return task_status


def run_forever():
    while True:
        task_status = None
        try:
            task_status = main()
        except Exception as e:
            if task_status is None:
                print("task does not exist")
            else:
                print("error in task: {}".format(task_status.cur_path))
                task_status.fail()
        finally:
            time.sleep(300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', action="store_true", help="use this flag to run forever")
    args = parser.parse_args()
    if args.repeat:
        run_forever()
    else:
        main()
