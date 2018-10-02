from enum import Enum
import os
import subprocess
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
        self._uhrs_exe_path = os.path.join(rootpath, "./UHRSDataCollection/UHRSDataCollection/bin/Debug/UHRSDataCollection.exe")
        if task_log and os.path.isfile(task_log):
            self.state = State.UPLOAD_FINISH
        elif task_log and os.path.isdir(task_log):
            raise Exception("task log should be a file")
        else:
            self.state = State.IDLE

    def block_worker(self, worker_id):
        args = [self._uhrs_exe_path, "block_judge", repr(worker_id)]
        subprocess.check_call(args)

    def block_workers(self, worker_id_file):
        if not os.path.isfile(worker_id_file):
            raise Exception("file does not exist: {}".format(worker_id_file))
        args = [self._uhrs_exe_path, "block_judges", worker_id_file]
        subprocess.check_call(args)

    def upload_tasks_from_folder(self, task_hitapp, dirpath, prefix="",
                                 num_judges=5):
        """Uploads task files in dirpath with prefix to UHRS, each hit will be
        judged by num_judges workers
        """
        if self.state != State.IDLE:
            raise Exception("cannot upload from state {}".format(self.state))
        self.state = State.UPLOAD_START
        task_group_id = self._get_task_group_id(task_hitapp)
        args = [self._uhrs_exe_path, "upload_from_folder", repr(task_group_id),
                dirpath, prefix, self._task_log, "0.0", repr(num_judges)]
        subprocess.check_call(args)
        self.state = State.UPLOAD_FINISH

    def download_tasks_to_folder(self, task_hitapp, dirpath):
        if self.state != State.WAIT_FINISH:
            raise Exception("cannot download from state {}".format(self.state))
        self.state = State.DOWNLOAD_START
        task_group_id = self._get_task_group_id(task_hitapp)
        args = [self._uhrs_exe_path, "download_to_folder", repr(task_group_id),
                dirpath, self._task_log]
        subprocess.check_call(args)
        self.state = State.IDLE

    def wait_until_task_finish(self, task_hitapp):
        if self.state != State.UPLOAD_FINISH:
            raise Exception("cannot wait from state {}".format(self.state))
        self.state = State.WAIT_FOR_TASK
        num_done, num_total = self._count_task_progress(
            task_hitapp, self._task_log)
        with tqdm(total=num_total) as pbar:
            while not self._is_task_finished(task_hitapp, self._task_log):
                num_done, _ = self._count_task_progress(
                    task_hitapp, self._task_log)
                pbar.update(num_done - pbar.n)
                time.sleep(300)
            num_done, _ = self._count_task_progress(
                    task_hitapp, self._task_log)
            pbar.update(num_done - pbar.n)
        self.state = State.WAIT_FINISH

    def _is_task_finished(self, task_hitapp, logfile):
        task_ids = self._read_task_ids_from_log(logfile)
        for i in task_ids:
            state = self._get_task_state(task_hitapp, i)[0]
            if state == 1:
                return False
            if state == 0 or state == 2:
                raise Exception("task id {} is not active now".format(i))
        return True

    def _count_task_progress(self, task_hitapp, logfile):
        task_ids = self._read_task_ids_from_log(logfile)
        num_done = 0
        num_total = 0
        for i in task_ids:
            d, t = self._get_task_state(task_hitapp, i)[1:3]
            num_done += d
            num_total += t
        return num_done, num_total

    def _read_task_ids_from_log(self, logfile):
        task_ids = []
        with open(logfile, 'r') as fp:
            for line in fp:
                task_ids.append(int(line.split('\t')[0]))
        return task_ids

    def _get_task_state(self, task_hitapp, task_id):
        """ Gets task state, judgments done, judgments total
        The state of the task: 0 - Disabled, 1 - Active, 2 - ManualCompleted,
        3 - Completed
        """
        task_group_id = self._get_task_group_id(task_hitapp)
        ret = subprocess.check_output([self._uhrs_exe_path, "get_task_state",
                                       repr(task_group_id), repr(task_id)])
        return [int(r) for r in ret.decode().rstrip('\r\n').split(' ')]

    def _get_task_group_id(self, task_hitapp):
        if task_hitapp == "verify_box":
            return 86314
        elif task_hitapp == "verify_cover":
            return 86329
        elif task_hitapp == "verify_box_group":
            return 91381
        else:
            raise Exception("Unknown task: {}".format(task_hitapp))
