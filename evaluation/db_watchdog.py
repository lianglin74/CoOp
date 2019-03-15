import argparse
import json
import logging
import os
import os.path as op
import tempfile
import time

from qd import db, tsv_io, qd_common
import _init_paths
from evaluation import generate_task, analyze_task
from evaluation.uhrs import UhrsTaskManager

class VerificationConfig():
    task_type = "VerifyBox"
    uhrs_type = "crowdsource_verify_box"
    honeypot = "//vigdgx02/raid_data/uhrs/eval/honeypot/voc20_easy_gt.txt"
    num_tasks_per_hit = 10
    num_hp_per_hit = 2
    max_hits_per_file = 2000
    num_judgment = 4
    max_tasks_running = 5

def get_working_dir():
    dirpath = op.join(tempfile.gettempdir(), "uhrs")
    qd_common.ensure_directory(dirpath)
    return dirpath

def verify_bbox_db(cur_db, args):
    db_uhrs_submitted_task_key = 'uhrs_submitted_result'
    db_uhrs_completed_task_key = "uhrs_completed_result"
    db_bbox_id_key = "_id"
    db_url_key = "url"
    db_rects_key = "rect"

    # scan existing tasks
    submitted_tasks = list(cur_db.query_submitted())
    all_task_ids = set()
    for t in submitted_tasks:
        ids = t[db_uhrs_submitted_task_key]
        assert(len(ids) == 2)
        all_task_ids.add((ids[0], ids[1]))
    num_running_tasks = len(all_task_ids)
    logging.info("{} tasks running".format(num_running_tasks))

    # complete task if already finished
    for ids in all_task_ids:
        task_group_id, task_id = ids[0], ids[1]
        if UhrsTaskManager.is_task_completed(task_group_id, task_id):
            res_file = op.join(get_working_dir(), "uhrs_results.tsv")
            UhrsTaskManager.download_task(task_group_id, task_id, res_file)
            id2ans = analyze_completed_task(res_file)
            completed_tasks = []
            for t in submitted_tasks:
                _id = t[db_bbox_id_key]
                if _id in id2ans:
                    assert id2ans[_id][1] == t[db_url_key]
                    assert len(id2ans[_id][2]) == 1
                    old_bb = id2ans[_id][2][0]
                    new_bb = t[db_rects_key]
                    assert old_bb["class"] == old_bb["class"]
                    assert all(i==j for i, j in zip(old_bb["rect"], new_bb["rect"]))

                    t[db_uhrs_completed_task_key] = id2ans[_id][0]
                    completed_tasks.append(t)
            cur_db.complete(completed_tasks)
            num_running_tasks -= 1
            logging.info("completed {} bboxes".format(len(completed_tasks)))

    # retrieve tasks to submit
    if num_running_tasks < args.max_tasks_running:
        bb_tasks = cur_db.retrieve(args.num_tasks_per_hit * args.max_hits_per_file)
        bb_tasks = list(bb_tasks)
        print len(bb_tasks)
        if len(bb_tasks) > 0:
            def gen_labels():
                for bb_task in bb_tasks:
                    bbox = bb_task[db_rects_key]
                    assert all([k in bbox for k in ["class", "rect"]])
                    yield bb_task[db_bbox_id_key], json.dumps([bbox]), bb_task[db_url_key]
            task_group_id, task_id = create_new_task(gen_labels(), args)

            for t in bb_tasks:
                t[db_uhrs_submitted_task_key] = [task_group_id, task_id]
            cur_db.submitted(bb_tasks)
            logging.info("submitted {} bboxes".format(len(bb_tasks)))

def create_new_task(label_enumerator, args):
    # write task files
    outdir = get_working_dir()
    label_file = op.join(outdir, "label.tsv")
    tsv_io.tsv_writer(label_enumerator, label_file)
    task_files = generate_task.generate_task_files(
                        args.task_type, label_file, args.honeypot, op.join(outdir, "uhrs_task"),
                        num_tasks_per_hit=args.num_tasks_per_hit, num_hits_per_file=args.max_hits_per_file,
                        num_hp_per_hit=args.num_hp_per_hit)
    assert len(task_files) == 1
    task_file = task_files[0]

    # submit task to uhrs
    task_group_id = UhrsTaskManager._get_task_group_id(args.uhrs_type)
    task_id = UhrsTaskManager.upload_task(task_group_id, task_file, args.num_judgment)
    return task_group_id, task_id

def analyze_completed_task(res_file):
    num_rejuege, summary = analyze_task.analyze_verify_box_task(
                            [res_file], "uhrs", None,
                            None, worker_quality_file=None,
                            min_num_judges_per_hit=1, min_num_hp=5,
                            accuracy_threshold=0.85, neg_threshold=None)
    # summary columns are: image_info, url, bboxes, answers Counter
    id2ans = {parts[0]: [json.loads(parts[3]), parts[1], json.loads(parts[2])] for parts in summary}
    return id2ans

def main():
    qd_common.init_logging()
    args = VerificationConfig()
    cur_db = db.create_bbverification_db()

    while True:
        verify_bbox_db(cur_db, args)
        time.sleep(300)

if __name__ == "__main__":
    main()
