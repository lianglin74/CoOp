import re
import os
import logging
from qd.qd_common import init_logging
from qd.qd_common import get_file_size
from pprint import pformat
import os.path as op
import time

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
    for root, dirnames, file_names in os.walk(folder):
        total_size += sum([get_file_size(op.join(root, f)) for f in file_names])
        matched = False
        for f in must_have_in_folder:
            if f in root:
                matched = True
                break
        if not matched:
            continue
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

