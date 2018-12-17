from enum import Enum
import logging
import os
import subprocess
import time
from tqdm import tqdm


class State(Enum):
    IDLE = 1
    UPLOAD_START = 2
    UPLOAD_FINISH = 3
    WAIT_FOR_TASK = 4
    WAIT_FINISH = 5
    DOWNLOAD_START = 6


class UhrsTaskManager():
    '''Python wrapper of UHRS API
    '''
    def __init__(self, task_log):
        self._task_log = task_log  # tsv file to store task id and name
        rootpath = os.path.dirname(os.path.realpath(__file__))
        self._uhrs_exe_path = os.path.join(rootpath, "./UHRSDataCollection/bin/Release/UHRSDataCollection.exe")
        if task_log and os.path.isfile(task_log):
            logging.info("tasks already uploaded at {}".format(task_log))
            self.state = State.UPLOAD_FINISH
        elif task_log and os.path.isdir(task_log):
            raise Exception("task log should be a file")
        else:
            self.state = State.IDLE

    def block_worker(self, worker_id):
        args = [self._uhrs_exe_path, "BlockSingleJudge",
                "-judgeId", repr(int(worker_id))]
        subprocess.check_call(args)

    def block_workers(self, worker_id_file):
        if not os.path.isfile(worker_id_file):
            raise Exception("file does not exist: {}".format(worker_id_file))
        args = [self._uhrs_exe_path, "BlockJudges",
                "-filepath", worker_id_file]
        subprocess.check_call(args)

    def upload_tasks_from_folder(self, task_group, dirpath, prefix="",
                                 consensus_thresh=0.0, num_judges=5):
        """Uploads task files in dirpath with prefix to UHRS, each hit will be
        judged by num_judges workers
        """
        if self.state != State.IDLE:
            raise Exception("cannot upload from state {}".format(self.state))
        self.state = State.UPLOAD_START
        task_group_id = self._get_task_group_id(task_group)
        args = [self._uhrs_exe_path, "UploadTasksFromFolder",
                "-taskGroupId", repr(task_group_id),
                "-folderPath", dirpath,
                "-taskIdNameFile", self._task_log,
                "-consensusThreshold", repr(consensus_thresh),
                "-numJudgment", repr(num_judges)]
        if prefix:
            args.extend(["-filePrefix", prefix])
        subprocess.check_call(args)
        self.state = State.UPLOAD_FINISH

    def download_tasks_to_folder(self, task_group, dirpath):
        if self.state != State.WAIT_FINISH:
            raise Exception("cannot download from state {}".format(self.state))
        self.state = State.DOWNLOAD_START
        task_group_id = self._get_task_group_id(task_group)
        args = [self._uhrs_exe_path, "DownloadTasksToFolder",
                "-taskGroupId", repr(task_group_id),
                "-folderPath", dirpath,
                "-taskIdFile", self._task_log]
        subprocess.check_call(args)
        self.state = State.IDLE

    def wait_until_task_finish(self, task_group):
        if self.state != State.UPLOAD_FINISH:
            raise Exception("cannot wait from state {}".format(self.state))
        self.state = State.WAIT_FOR_TASK
        num_done, num_total = self._count_task_progress(
            task_group, self._task_log)
        with tqdm(total=num_total) as pbar:
            while not self._is_task_finished(task_group, self._task_log):
                num_done, _ = self._count_task_progress(
                    task_group, self._task_log)
                pbar.update(num_done - pbar.n)
                time.sleep(60)
            num_done, _ = self._count_task_progress(
                    task_group, self._task_log)
            pbar.update(num_done - pbar.n)
        self.state = State.WAIT_FINISH

    def _is_task_finished(self, task_group, logfile):
        task_ids = self._read_task_ids_from_log(logfile)
        for i in task_ids:
            state = self._get_task_state(task_group, i)[0]
            if state == 1:
                return False
            if state == 0 or state == 2:
                raise Exception("task id {} is not active now".format(i))
        return True

    def _count_task_progress(self, task_group, logfile):
        task_ids = self._read_task_ids_from_log(logfile)
        num_done = 0
        num_total = 0
        for i in task_ids:
            d, t = self._get_task_state(task_group, i)[1:3]
            num_done += d
            num_total += t
        return num_done, num_total

    def _read_task_ids_from_log(self, logfile):
        task_ids = []
        with open(logfile, 'r') as fp:
            for line in fp:
                task_ids.append(int(line.split('\t')[0]))
        return task_ids

    def _get_task_state(self, task_group, task_id):
        """ Gets task state, judgments done, judgments total
        The state of the task: 0 - Disabled, 1 - Active, 2 - ManualCompleted,
        3 - Completed
        """
        task_group_id = self._get_task_group_id(task_group)
        ret = subprocess.check_output([self._uhrs_exe_path, "GetTaskState",
                                       "-taskGroupId", repr(task_group_id),
                                       "-taskId", repr(task_id)])
        return [int(r) for r in ret.decode().split('\r\n', 1)[0].split(' ')]

    def _get_task_group_id(self, task_group):
        if task_group == "verify_box":
            return 86314
        elif task_group == "verify_cover":
            return 86329
        elif task_group == "crowdsource_verify_box":
            return 91381
        elif task_group == "test":
            return 88209
        elif task_group == "internal_verify_box":
            return 111012
        elif task_group == "vendor_verify_box":
            return 113795
        else:
            raise Exception("Unknown task: {}".format(task_group))


def test():
    rootpath = "//ivm-server2/IRIS/OD/eval/tasks/test/"
    uhrs_client = UhrsTaskManager(os.path.join(rootpath, "log.txt"))
    w_id = 388605
    w_file = os.path.join(rootpath, "bad_worker.txt")
    task_group = "test"
    with open(w_file, 'w') as fp:
        fp.write(str(w_id) + '\n')
    uhrs_client.block_worker(w_id)
    uhrs_client.block_workers(w_file)
    upload_dir = os.path.join(rootpath, "upload")
    download_dir = os.path.join(rootpath, "download")
    uhrs_client.upload_tasks_from_folder(task_group, upload_dir, num_judges=1)
    # check active task at https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitappid=35295
    uhrs_client.wait_until_task_finish(task_group)
    uhrs_client.download_tasks_to_folder(task_group, download_dir)
