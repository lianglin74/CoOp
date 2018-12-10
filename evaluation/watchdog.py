import logging
import os
import sys
import time

import _init_paths
from evaluation.eval_utils import GroundTruthConfig
from evaluation.update_gt import update_gt
from scripts.qd_common import load_from_yaml_file, init_logging

class TaskStatus(object):
    def __init__(self, fpath):
        self.TASK_STATUS = ["running", "completed", "fail"]
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
        ch = logging.StreamHandler(sys.stdout)
        logger.addHandler(ch)

    def _change_status(self, target_status):
        assert(target_status in self.TASK_STATUS)
        logging.info("task status moves to: {}".format(target_status))
        fname = os.path.split(self.cur_path)[1]
        dest_path = os.path.join(self.rootpath, target_status, fname)
        if os.path.isfile(dest_path):
            fname = str(time.time()) + '-' + fname
            dest_path = os.path.join(self.rootpath, target_status, fname)
        os.rename(self.cur_path, dest_path)
        self.cur_path = dest_path


def main():
    rootpath = "//vigdgx02/raid_data/uhrs/"
    task_dir = "//vigdgx02/raid_data/uhrs/status/new/"

    while True:
        # sweep new task config files
        task_yaml_list = [os.path.join(task_dir, f) for f in os.listdir(task_dir)
            if f.endswith(".yaml") and os.path.isfile(os.path.join(task_dir, f))]

        if len(task_yaml_list) == 0:
            time.sleep(100)
        for task_yaml in task_yaml_list:
            task_status = TaskStatus(task_yaml)
            task_config = load_from_yaml_file(task_yaml)
            gt_config_file = os.path.join(rootpath, task_config["gt_config"])
            model_name = task_config["model_name"]

            task_status.start()
            # add the new model to baselines
            gt_cfg = GroundTruthConfig(gt_config_file)
            dataset_list = []
            for pred_file in task_config["pred_files"]:
                cur_dataset = pred_file["dataset"]
                assert(cur_dataset not in dataset_list)
                dataset_list.append(cur_dataset)
                gt_cfg.add_baseline(cur_dataset, model_name, pred_file["result"], pred_file["conf_threshold"])

            task_root = os.path.dirname(os.path.dirname(gt_config_file))
            task_dir = os.path.join(task_root, "tasks")
            hp_dir = os.path.join(task_root, "honeypot")
            hp_files = [f for f in os.listdir(hp_dir) if f.endswith(".txt")]
            hp_file = os.path.join(hp_dir, hp_files[0])

            # update_gt ongoing
            for dataset in dataset_list:
                args = [model_name, task_config["task_type"],
                        "--datasets", dataset,
                        "--config", gt_config_file,
                        "--taskdir", task_dir, "--honeypot", hp_file]
                logging.info("update ground truth with arguments: {}".format(str(args)))
                update_gt(args)

            # finished updating gt
            task_status.complete()


if __name__ == "__main__":
    main()
