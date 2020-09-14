import re
import os
import logging
from qd.qd_common import init_logging
from qd.qd_common import get_file_size
from pprint import pformat
import os.path as op
import time


def iter_file(folder):
    for root, _, file_names in os.walk(folder):
        for f in file_names:
            yield op.join(root, f)

def del_backed_models(
        target_folder='',
        not_in_folder='',
        in_folders='',
        threshold_in_days=7,
        dry_run=True,
):
    # we will delete a file if 1) the file is in target folder & 2) the file is
    # not in not_in_folder and 3) teh file is in in_folder.
    # the scenarios, the azure blob folder contains a file, the file is in
    # local downloaded folder, but it is in back-ed up folder.
    total = 0
    for i, f in enumerate(iter_file(target_folder)):
        logging.info('checking {}'.format(f))
        not_f = f.replace(target_folder, not_in_folder)
        if op.isfile(not_f):
            logging.info('ignore since {} exists - {}'.format(
                not_f, total / 1024. ** 3))
            continue
        in_fs = [f.replace(target_folder, in_folder) for in_folder in in_folders]
        if all(not op.isfile(in_f) for in_f in in_fs):
            logging.info('ignore since {} does not exist - {}'.format(
                pformat(in_fs),
                total / 1024. ** 3
            ))
            continue
        if not old_enough(f, threshold_in_days):
            logging.info('ignore since not old enough - {}'.format(
                total / 1024. ** 3
            ))
            continue
        logging.info('removing {} - {} - {:.4f}'.format(
            f, total, 1. * total / 1024. ** 3))
        total += get_file_size(f)
        if not dry_run:
            from qd.qd_common import try_delete
            try_delete(f)
    logging.info('deleted {} GB'.format(total / 1024. / 1024. / 1024.))

def del_intermediate_models(folder='./output', threshold_in_days=30,
        must_have_in_folder=None, dry_run=False):
    logging.info('start')
    total_size = 0
    for aname in iter_to_be_deleted(folder, threshold_in_days,
            must_have_in_folder):
        s = os.stat(aname)
        total_size = total_size + s.st_size
        logging.info('removing {} - {}'.format(aname,
            total_size/1024./1024./1024.))
        if not dry_run:
            try:
                os.remove(aname)
            except:
                logging.info('failed: {}'.format(aname))
                continue
    logging.info(total_size / 1024.0 / 1024.0 / 1024.0)

def old_enough(fname, threshold_in_days):
    if not op.isfile(fname):
        return False
    d = time.time() - os.stat(fname).st_atime
    days = d / 3600. / 24.
    return days > threshold_in_days

def iter_to_be_deleted(folder, threshold_in_days=30, must_have_in_folder=None):
    if must_have_in_folder is None:
        must_have_in_folder = ['']
    total_size = 0
    for root, _, file_names in os.walk(folder):
        total_size += sum([get_file_size(op.join(root, f)) for f in file_names])
        matched = all(f in root for f in must_have_in_folder)
        if not matched:
            logging.info('skipping {}'.format(root))
            continue
        logging.info('deleting in {}'.format(root))
        from qd.qd_common import parse_iteration
        ms = [(f, parse_iteration(f)) for f in file_names]
        ms = [(f, m) for f, m in ms if m > 0]
        ms = [(f, iteration) for f, iteration in ms if old_enough(op.join(root,
            f), threshold_in_days)]
        if len(ms) == 0:
            continue
        max_iter = max(x[1] for x in ms)
        to = [f for f, i in ms if i < max_iter]
        if len(to) == 0:
            continue
        for t in to:
            yield os.path.join(root, t)
    logging.info('in total {}G'.format(total_size / 1024. / 1024. / 1024.))

if __name__ == '__main__':
    init_logging()
    from qd_common import parse_general_args

    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    return_value = locals()[function_name](**kwargs)
    if return_value is not None:
        logging.info('return = {}'.format(pformat(return_value)))

