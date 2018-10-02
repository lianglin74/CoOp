import collections
import json
import logging
import numpy as np
import os
import pandas as pd

from generate_task import pack_task_with_honey_pot, write_task_file
from uhrs import UhrsTaskManager
from utils import load_escaped_json, write_to_file


def analyze_verify_box_task(result_files, result_file_type, outfile_res,
                            outfile_rejudge, worker_quality_file=None,
                            min_num_judges_per_hit=5, min_num_hp=5,
                            accuracy_threshold=0.9, neg_threshold=None):
    """Parses result files from verify_box tasks, where workers are ask to
    verify if a single bbox is good: 1-Yes, 2-No, 3-Can't judge.
    Consensus answers will be written to outfile_res, tasks that don't reach
    consensus will be written to outfile_rejudge
    """
    # load results
    logging.info('\n'.join(['Merging Labeling result file:{:s}\t'.format(a)
                            for a in result_files]))
    results = pd.DataFrame()
    if result_file_type == "mturk":
        for resultfile in result_files:
            results = results.append(pd.read_csv(resultfile))
        logging.info('{:d} lines loaded.'.format(len(results)))
        df_records = results[['HITId', 'WorkerId', 'Answer.output',
                              'Input.input_content']]
    elif result_file_type == "uhrs":
        for resultfile in result_files:
            results = results.append(pd.read_csv(resultfile, sep='\t'))
        logging.info('{:d} lines loaded.'.format(len(results)))
        df_records = results[['HitID', 'JudgeID', 'output',
                              'input_content']].rename(
                     columns={'HitID': 'HITId', 'JudgeID': 'WorkerId',
                              'output': 'Answer.output',
                              'input_content': 'Input.input_content'})
    else:
        raise Exception("invalid file type: {}".format(result_file_type))

    # analyze worker quality
    bad_worker_ids = analyze_worker_quality(df_records, worker_quality_file,
                                            min_num_hp, accuracy_threshold,
                                            neg_threshold, False)

    # block bad workers
    uhrs_client = UhrsTaskManager(None)
    for w in bad_worker_ids:
        uhrs_client.block_worker(w)

    # remove bad worker results
    df_valid_records = df_records[~df_records.WorkerId.isin(bad_worker_ids)]

    hp_tasks = []
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
            all_answers.append([int(i) for i in row.split(';')])
        all_answers = np.array(all_answers).T
        assert(len(input_tasks) == len(all_answers))
        for task, answers in zip(input_tasks, all_answers):
            # skip honey pot tasks
            if "expected_output" in task:
                hp_tasks.append(task)
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
                consensus_tasks[consensus_ans].append(task)

    num_rejudge = len(rejudge_tasks)
    num_consensus = sum(len(v) for k, v in consensus_tasks.items())
    num_total = num_rejudge + num_consensus
    logging.info('#honey pot: {}'.format(len(hp_tasks)))
    logging.info('#total hits: {}, #need re-judge: {}({:.2f}%)'
                 .format(num_total, num_rejudge,
                         float(num_rejudge)/num_total*100))
    logging.info('option\tcount\tpercentage')
    for k, v in consensus_tasks.items():
        v = len(v)
        logging.info('{}\t{}\t{}'.format(k, v, float(v)/num_consensus*100))

    ambiguous_categories = [b["objects_to_find"] for b in consensus_tasks[3]]
    logging.info("Categories labelled as [can't judge]: {}"
                 .format(collections.Counter(ambiguous_categories)))

    # write consensus results
    result_data = [[str(k), json.dumps(v)] for k, v in consensus_tasks.items()]
    write_to_file(result_data, outfile_res)
    logging.info("Writing consensus results to: {}".format(outfile_res))

    # write tasks needing re-judge
    if rejudge_tasks:
        rejudge_data = pack_task_with_honey_pot(rejudge_tasks, hp_tasks,
                                                "hp", 8, 2)
        write_task_file(rejudge_data, outfile_rejudge)
    return num_rejudge


def get_consensus_answer(answers, consensus_threshold=0.5):
    ans_count = collections.Counter(answers)
    threshold_count = len(answers) * consensus_threshold
    for ans, count in ans_count.items():
        if count > threshold_count:
            return ans
    return None


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
                         float(len(bad_worker_ids))/len(all_workers)*100,
                         results_to_throw_away,
                         float(results_to_throw_away)/len(df_records)*100))
    if plot:
        import matplotlib.pyplot as plt
        plt.hist(df['Accuracy_HP'], bins=30, range=[0.5, 1])
        plt.xlabel('Accuracy on Honey Pot')
        plt.title('UHRS workers')
        plt.show()
    return bad_worker_ids
