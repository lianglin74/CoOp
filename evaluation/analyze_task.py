import collections
from datetime import datetime
import json
import logging
import numpy as np
import os
import pandas as pd

from evaluation.generate_task import pack_task_with_honey_pot, write_task_file
from evaluation.uhrs import UhrsTaskManager
from evaluation.utils import load_escaped_json
from qd.tsv_io import tsv_writer


def analyze_draw_box_task(result_files, outfile_res, result_file_type="uhrs",
            start_time=None):
    # load results
    df_records = load_task_results(result_files, result_file_type, start_time=start_time)
    url2ans_map = collections.defaultdict(list)
    labelset = set()
    for _, row in df_records.iterrows():
        try:
            input_task = load_escaped_json(row['Input.input_content'])
        except Exception:
            continue
        # skip honey pot
        if "expected_output" in input_task:
            continue
        answer = []
        for b in json.loads(row['Answer.output']):
            b = parse_bbox(b)
            if b:
                answer.append(b)
        for b in answer:
            b["class"] = input_task["objects_to_find"]
        img_url = input_task["image_url"]
        labelset.add(input_task["objects_to_find"])
        url2ans_map[img_url].extend(answer)

    def gen_rows():
        for url in url2ans_map:
            # TODO: merge answers from several workers
            bbox_list = url2ans_map[url]
            yield url, json.dumps(bbox_list, separators=(',', ':'))
    tsv_writer(gen_rows(), outfile_res)
    return url2ans_map


def analyze_verify_box_task(result_files, result_file_type, outfile_res,
                            outfile_rejudge, worker_quality_file=None,
                            min_num_judges_per_hit=4, min_num_hp=5,
                            accuracy_threshold=0.8, neg_threshold=None):
    """Parses result files from verify_box tasks, where workers are ask to
    verify if a single bbox is good: 1-Yes, 2-No, 3-Can't judge.
    Consensus answers will be written to outfile_res, tasks that don't reach
    consensus will be written to outfile_rejudge
    """
    # load results
    df_records = load_task_results(result_files, result_file_type)

    # analyze worker quality
    bad_worker_ids = analyze_worker_quality(df_records, worker_quality_file,
                                            min_num_hp, accuracy_threshold,
                                            neg_threshold, False)

    # block bad workers, only support exe
    if os.name == "nt":
        uhrs_client = UhrsTaskManager(None)
        for w in bad_worker_ids:
            uhrs_client.block_worker(w)

    # remove bad worker results
    df_valid_records = df_records[~df_records.WorkerId.isin(bad_worker_ids)]

    hp_tasks = []
    hp_agreement = 0
    rejudge_tasks = []
    consensus_tasks = collections.defaultdict(list)
    # remove honey pot tasks, and reformat remaining results
    # one hit contains several tasks, one task was judged several times
    # uuid and task have a one-to-one mapping
    uuid2task_map = dict()
    uuid2answers_map = collections.defaultdict(list)

    for hit_id, group in df_valid_records.groupby('HITId'):
        input_tasks = load_escaped_json(
            group['Input.input_content'].iloc[0])
        all_answers = []
        for _, row in group['Answer.output'].iteritems():
            # the output might be empty (nan) due to UHRS server error
            if not row or isinstance(row, float):
                continue
            all_answers.append([int(i) for i in row.split(';')])
        all_answers = np.array(all_answers).T
        assert(len(input_tasks) == len(all_answers))
        for task, answers in zip(input_tasks, all_answers):
            # skip honey pot tasks
            if "expected_output" in task:
                hp_tasks.append(task)
                if get_consensus_answer(answers) == task["expected_output"]:
                    hp_agreement += 1
                continue
            task_uuid = task["uuid"]
            uuid2task_map[task_uuid] = task
            uuid2answers_map[task_uuid].extend(answers)
    # get consensus from results
    # rejudge_tasks: not enough valid answers or no consensus
    for uuid, answers in uuid2answers_map.items():
        num_answers = len(answers)
        task = uuid2task_map[uuid]
        if num_answers < min_num_judges_per_hit:
            # if there are no enough answers, rejudge the task
            rejudge_tasks.append(task)
        else:
            consensus_ans = get_consensus_answer(answers)
            if not consensus_ans:
                rejudge_tasks.append(task)
            else:
                bboxes = task["bboxes"]
                for b in bboxes:
                    b["uhrs"] =  ans_array_to_dict(answers)
                consensus_tasks[consensus_ans].append(task)

    num_rejudge = len(rejudge_tasks)
    num_consensus = sum(len(v) for k, v in consensus_tasks.items())
    num_total = num_rejudge + num_consensus
    logging.info('#honey pot: {}, agreement: {:.2f}%'.format(len(hp_tasks), float(hp_agreement)/len(hp_tasks)*100))
    logging.info('#total hits: {}, #need re-judge: {}({:.2f}%)'
                 .format(num_total, num_rejudge,
                         float(num_rejudge)/num_total*100))
    logging.info('option\tcount\tpercentage')
    for k, v in consensus_tasks.items():
        logging.info('{}\t{}\t{}'.format(k, len(v), float(len(v))/num_consensus*100))

    ambiguous_categories = [b["objects_to_find"] for b in consensus_tasks[3]]
    logging.info("Categories labelled as [can't judge]: {}"
                 .format(collections.Counter(ambiguous_categories)))

    # write consensus results
    result_data = [[str(k), json.dumps(v, separators=(',', ':'))] for k, v in consensus_tasks.items()]
    if outfile_res:
        tsv_writer(result_data, outfile_res)
        logging.info("Writing consensus results to: {}".format(outfile_res))

    # write uhrs answer summary
    summary = []
    for uuid in uuid2answers_map:
        ans = ans_array_to_dict(uuid2answers_map[uuid])
        task = uuid2task_map[uuid]
        summary.append([task["image_info"], task["image_url"],
                json.dumps(task["bboxes"], separators=(',', ':')),
                json.dumps(ans, separators=(',', ':'))])

    # write tasks needing re-judge
    if rejudge_tasks:
        rejudge_data = pack_task_with_honey_pot(rejudge_tasks, hp_tasks,
                                                "hp", 15, 3)
        if outfile_rejudge:
            write_task_file(rejudge_data, outfile_rejudge)
    return num_rejudge, summary


def ans_array_to_dict(arr):
    counts = collections.Counter(arr)
    res = {}
    for k in counts:
        res[str(k)] = counts[k]
    return res


def get_consensus_answer(answers, consensus_threshold=0.5):
    ans_count = collections.Counter(answers)
    threshold_count = len(answers) * consensus_threshold
    for ans, count in ans_count.items():
        if count > threshold_count:
            return ans
    return None


def load_task_results(result_files, result_file_type, start_time=None):
    logging.info('\n'.join(['Merging Labeling result file:{:s}\t'.format(a)
                            for a in result_files]))
    filtered_files = []
    for f in result_files:
        if os.stat(f).st_size == 0:
            logging.info("empty file: {}".format(f))
        else:
            filtered_files.append(f)

    results = pd.DataFrame()
    if result_file_type == "mturk":
        for resultfile in filtered_files:
            results = results.append(pd.read_csv(resultfile))
        logging.info('{:d} lines loaded.'.format(len(results)))
        df_records = results[['HITId', 'WorkerId', 'Answer.output',
                              'Input.input_content']]
    elif result_file_type == "uhrs":
        for resultfile in filtered_files:
            results = results.append(pd.read_csv(resultfile, sep='\t'))
        logging.info('{:d} lines loaded.'.format(len(results)))
        df_records = results[['HitID', 'JudgeID', 'output',
                              'input_content', 'JudgmentSubmitTime']].rename(
                     columns={'HitID': 'HITId', 'JudgeID': 'WorkerId',
                              'output': 'Answer.output',
                              'input_content': 'Input.input_content',
                              'JudgmentSubmitTime': 'submit_time'})
    else:
        raise Exception("invalid file type: {}".format(result_file_type))

    if start_time is not None:
        start = start_time.timestamp()
        num_orig = len(df_records)
        df_records = df_records[
            df_records['submit_time'].map(
                lambda t: datetime.strptime(t, '%m/%d/%Y %H:%M:%S %p'
            ).timestamp()) >= start]

        logging.info("Dropped {} rows before {}".format(num_orig - len(df_records), str(start_time)))

    return df_records


def parse_bbox(bbox):
    """ Parses bbox format from uhrs
    """
    if ('left' not in bbox):
        return {"SKIP": "", "class": "unknown"}

    left = bbox["left"]
    left = np.clip(left, 0, bbox["image_width"])
    right = bbox["left"] + bbox["width"]
    right = np.clip(right, left, bbox["image_width"])
    top = bbox["top"]
    top = np.clip(top, 0, bbox["image_height"])
    bottom = bbox["top"] + bbox["height"]
    bottom = np.clip(bottom, top, bbox["image_height"])
    if right - left <= 0 or bottom - top <= 0:
        return None
    res = {"rect": [int(n) for n in [left, top, right, bottom]], "class": bbox["label"]}
    for k in bbox:
        if k not in ["left", "top", "width", "height", "image_width", "image_height", "label"]:
            res[k] = bbox[k]
    return res


def analyze_worker_quality(df_records, worker_quality_file=None, min_num_hp=5,
                           accuracy_threshold=0.9, neg_threshold=None,
                           plot=False):
    """Analyzes worker quality basing on honey pot
    Args:
        df_records: DataFrame contaioning WorkerId, Answer.output
        worker_quality_file: filepath to write the worker quality statistics
        min_num_hp: minimum number of honey pot required to calculate accuracy
        accuracy_threshold: minumum accuracy required for good workers
        neg_threshold: optional, designed to block workers who asnwer "No" in
            most cases
        plot: indicate if plotting worker quality histogram
    """
    all_workers = []
    bad_worker_ids = []
    for worker, group in df_records.groupby('WorkerId'):
        num_honey_pot = 0
        num_correct_hp = 0
        hp_history = []
        for idx, row in group.iterrows():
            # some answer columns are empty due to UHRS server error
            if not row['Answer.output'] or row.isnull().values.any():
                continue
            answer_list = row['Answer.output'].split(';')
            input_list = load_escaped_json(row['Input.input_content'])
            if len(answer_list) != len(input_list):
                raise Exception("WorkerId: {} answer length not match".format(
                    worker))
            for answer, input_item in zip(answer_list, input_list):
                if "expected_output" in input_item:
                    num_honey_pot += 1
                    hp_history.append(int(answer))
                    if int(answer) == input_item["expected_output"]:
                        num_correct_hp += 1

        if num_honey_pot > min_num_hp:
            accuracy = float(num_correct_hp) / num_honey_pot
            hp_history = collections.Counter(hp_history)
            neg_ratio = hp_history[2] / num_honey_pot
            all_workers.append([worker, len(group), str(hp_history),
                                num_honey_pot, num_correct_hp, accuracy,
                                neg_ratio])
            if accuracy <= accuracy_threshold or \
                    (neg_threshold and neg_ratio >= neg_threshold):
                bad_worker_ids.append(worker)

    df = pd.DataFrame(all_workers, columns=['WorkerId', '#labeled', 'history',
                                            '#HoneyPots', '#Correct',
                                            'Accuracy_HP', 'ratio_no'])
    df = df.sort_values(by='Accuracy_HP')
    if worker_quality_file:
        logging.info('writing history of {:d} workers to {:s}...\n'
                     .format(len(all_workers), worker_quality_file))
        df.to_csv(worker_quality_file, index=False, sep='\t')

    bad_worker_ids = set(bad_worker_ids)
    results_to_throw_away = sum([w[1] for w in all_workers
                                if w[0] in bad_worker_ids])
    logging.info('bad worker criteria: #HP > {}, accuracy <= {}, negtive ratio >= {}'
                 .format(min_num_hp, accuracy_threshold, neg_threshold))
    logging.info('#bad workers: {:d} ({:.2f}%), #abandoned labels: {:d} ({:.2f}%)'
                 .format(len(bad_worker_ids),
                         float(len(bad_worker_ids))/len(all_workers)*100 if all_workers else 0,
                         results_to_throw_away,
                         float(results_to_throw_away)/len(df_records)*100))
    if plot:
        import matplotlib.pyplot as plt
        plt.hist(df['Accuracy_HP'], bins=30, range=[0.5, 1])
        plt.xlabel('Accuracy on Honey Pot')
        plt.title('UHRS workers')
        plt.show()
    return bad_worker_ids
