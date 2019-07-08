from evaluation.uhrs import *


def test():
    rootpath = "//ivm-server2/iris/ChinaMobile/Video/uhrs/"
    task_group = "cv_internal"

    # log.txt contains the task id to download
    uhrs_client = UhrsTaskManager(os.path.join(rootpath, "log_CBA_2_part_2_3.txt"))

    download_dir = os.path.join(rootpath, "download")
    #uhrs_client.upload_tasks_from_folder(task_group, upload_dir, num_judges=1)
    # check active task at https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitappid=35295
    
    uhrs_client.state = State.WAIT_FINISH
    
    uhrs_client.download_tasks_to_folder(task_group, download_dir)
    
    
if __name__ == '__main__':
  test()