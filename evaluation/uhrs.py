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
    rootpath = os.path.dirname(os.path.realpath(__file__))
    UHRS_EXE = os.path.join(rootpath, "./UHRSDataCollection/bin/Release/UHRSDataCollection.exe")
    MAX_TRIES = 5

    def __init__(self, task_log=None):
        self._task_log = task_log  # tsv file to store task id and name
        if task_log and os.path.isfile(task_log):
            logging.info("tasks already uploaded at {}".format(task_log))
            self.state = State.UPLOAD_FINISH
        elif task_log and os.path.isdir(task_log):
            raise Exception("task log should be a file")
        else:
            self.state = State.IDLE

    @classmethod
    def block_worker(cls, worker_id):
        args = [cls.UHRS_EXE, "BlockSingleJudge",
                "-judgeId", repr(int(worker_id))]
        for i in range(cls.MAX_TRIES):
            try:
                subprocess.check_call(args)
                break
            except:
                logging.info('fails, tried {} times'.format(i+1))
                time.sleep(5)

    @classmethod
    def block_workers(cls, worker_id_file):
        if not os.path.isfile(worker_id_file):
            raise Exception("file does not exist: {}".format(worker_id_file))
        args = [cls.UHRS_EXE, "BlockJudges",
                "-filepath", worker_id_file]
        for i in range(cls.MAX_TRIES):
            try:
                subprocess.check_call(args)
                break
            except:
                logging.info('fails, tried {} times'.format(i+1))
                time.sleep(5)

    @classmethod
    def upload_task(cls, task_group_id, filepath, num_judgment, consensus_thres=0.0, priority=1000):
        ret = subprocess.check_output([cls.UHRS_EXE, "UploadSingleTask",
                                        "-taskGroupId", repr(task_group_id),
                                        "-filePath", filepath,
                                        "-numJudgment", repr(num_judgment),
                                        "-consensusThreshold", repr(consensus_thres),
                                        "-priority", repr(priority)])

        task_id = cls._parse_subprocess_output(ret)
        return int(task_id)

    @classmethod
    def download_task(cls, task_group_id, task_id, filepath):
        subprocess.check_call([cls.UHRS_EXE, "DownloadSingleTask",
                                       "-taskGroupId", repr(task_group_id),
                                       "-taskId", repr(task_id),
                                       "-filePath", filepath])

    @classmethod
    def is_task_completed(cls, task_group_id, task_id):
        try:
            state = cls._get_task_state(task_group_id, task_id)[0]
        except:
            logging.info('Error when getting state for task group {}, task id {}'.format(
                task_group_id, task_id))
            return False
        if state == 1:
            return False
        if state == 0:
            logging.info("task id {} is disabled".format(task_id))
            return False
        return True

    def is_task_exist(self):
        if self.state == State.IDLE or self.state == State.UPLOAD_START:
            return False
        else:
            return True

    def upload_tasks_from_folder(self, task_group, dirpath, prefix="",
                                 consensus_thresh=0.0, num_judges=5, priority=1000):
        """Uploads task files in dirpath with prefix to UHRS, each hit will be
        judged by num_judges workers
        """
        if self.state != State.IDLE:
            raise Exception("cannot upload from state {}".format(self.state))
        self.state = State.UPLOAD_START
        task_group_id = self._get_task_group_id(task_group)
        args = [self.UHRS_EXE, "UploadTasksFromFolder",
                "-taskGroupId", repr(task_group_id),
                "-folderPath", dirpath,
                "-taskIdNameFile", self._task_log,
                "-consensusThreshold", repr(consensus_thresh),
                "-numJudgment", repr(num_judges),
                "-priority", repr(priority)]
        if prefix:
            args.extend(["-filePrefix", prefix])
        subprocess.check_call(args)
        self.state = State.UPLOAD_FINISH

    def download_tasks_to_folder(self, task_group, dirpath):
        if self.state != State.WAIT_FINISH:
            raise Exception("cannot download from state {}".format(self.state))
        self.state = State.DOWNLOAD_START
        task_group_id = self._get_task_group_id(task_group)
        args = [self.UHRS_EXE, "DownloadTasksToFolder",
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
            last_done = 0
            last_tic = time.time()
            while not self._is_task_finished(task_group, self._task_log):
                num_done, _ = self._count_task_progress(
                    task_group, self._task_log)
                pbar.update(num_done - pbar.n)
                # terminate if almost done and stuck for too long
                tic = time.time()
                if (float(num_done)/num_total > 0.95 or num_total < 3) \
                        and num_done==last_done and tic-last_tic>1800:
                    break
                if num_done != last_done:
                    last_done = num_done
                    last_tic = tic
                time.sleep(60)
            num_done, _ = self._count_task_progress(
                    task_group, self._task_log)
            pbar.update(num_done - pbar.n)
        self.state = State.WAIT_FINISH

    def _is_task_finished(self, task_group, logfile):
        task_group_id = self._get_task_group_id(task_group)
        task_ids = self._read_task_ids_from_log(logfile)
        for i in task_ids:
            if not self.is_task_completed(task_group_id, i):
                return False
        return True

    def _count_task_progress(self, task_group, logfile):
        task_group_id = self._get_task_group_id(task_group)
        task_ids = self._read_task_ids_from_log(logfile)
        num_done = 0
        num_total = 0
        for i in task_ids:
            d, t = self._get_task_state(task_group_id, i)[1:3]
            num_done += d
            num_total += t
        return num_done, num_total

    def _read_task_ids_from_log(self, logfile):
        task_ids = []
        with open(logfile, 'r') as fp:
            for line in fp:
                task_ids.append(int(line.split('\t')[0]))
        return task_ids

    @classmethod
    def _get_task_state(cls, task_group_id, task_id):
        """ Gets task state, judgments done, judgments total
        The state of the task: 0 - Disabled, 1 - Active, 2 - ManualCompleted,
        3 - Completed
        """
        ret = subprocess.check_output([cls.UHRS_EXE, "GetTaskState",
                                       "-taskGroupId", repr(task_group_id),
                                       "-taskId", repr(task_id)])
        return [int(r) for r in cls._parse_subprocess_output(ret).split(' ')]

    @staticmethod
    def _parse_subprocess_output(ret):
        # print output at the end of program
        return ret.decode().strip('\r\n').rsplit('\r\n')[-1]

    @classmethod
    def _get_task_group_id(cls, task_group):
        if task_group == "verify_box":
            return 86314
        elif task_group == "verify_cover":
            return 86329
        elif task_group == "crowdsource_verify_box":
            return 91381
        elif task_group == "crowdsource_verify_tag":
            return 91759
        elif task_group == "internal_verify_tag":
            return 125330
        elif task_group == "test":
            return 88209
        elif task_group == "internal_verify_box":
            return 111012
        elif task_group == "vendor_verify_box":
            return 113795
        elif task_group == "vendor_draw_box":
            return 113972
        elif task_group == "crowdsource_draw_box":
            return 86129
        elif task_group == "cv_internal":
            return 117630
        else:
            raise Exception("Unknown task: {}".format(task_group))

def upload_uhrs_tasks(upload_files, task_group_id, task_id_tsv, num_judgment=1):
    from qd.tsv_io import tsv_writer
    def gen_task_ids():
        for upload_file in upload_files:
            task_id = UhrsTaskManager.upload_task(task_group_id, upload_file, num_judgment=num_judgment)
            yield task_group_id, task_id, upload_file
    tsv_writer(gen_task_ids(), task_id_tsv)


def test():
    rootpath = "//ivm-server2/IRIS/OD/eval/tasks/test/"
    task_group = "test"

    uhrs_client = UhrsTaskManager(os.path.join(rootpath, "log.txt"))

    # test blocking workers
    w_id = 388605
    w_file = os.path.join(rootpath, "bad_worker.txt")
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
