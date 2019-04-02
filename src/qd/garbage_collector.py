import re
import os
import logging
from .qd_common import init_logging
from pprint import pformat
import os.path as op
import time

def get_file_size(f):
    if not op.isfile(f):
        return 0
    return os.stat(f).st_size

def del_intermediate_models(folder='./output'):
    logging.info('start')
    total_size = 0
    for aname in iter_to_be_deleted(folder):
        s = os.stat(aname)
        total_size = total_size + s.st_size
        logging.info('removing {} - {}'.format(aname,
            total_size/1024./1024./1024.))
        try:
            os.remove(aname)
        except:
            logging.info('failed: {}'.format(aname))
            continue
    logging.info(total_size / 1024.0 / 1024.0 / 1024.0)

def old_enough(fname):
    if not op.isfile(fname):
        return False
    d = time.time() - os.stat(fname).st_atime
    days = d / 3600. / 24.
    return days > 30

def iter_to_be_deleted(folder):
    iter_extract_pattern = '.*model_iter_([0-9]*)e?\..*'
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
        ms = [(f, re.match(iter_extract_pattern, f)) for f in file_names]
        ms = [(f, int(m.groups()[0])) for f, m in ms if m]
        ms = [(f, iteration) for f, iteration in ms if old_enough(op.join(root,
            f))]
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
