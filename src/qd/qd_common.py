import ruamel.yaml as yaml
from collections import OrderedDict
import progressbar
import json
import sys
import os
import math
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Event
import logging
import numpy as np
import logging
import glob
import re
import tqdm
try:
    from itertools import izip as zip
except ImportError:
    # in python3, we don't need itertools.izip since zip is izip
    pass
import time
import matplotlib.pyplot as plt
from pprint import pprint
from pprint import pformat
import numpy as np
import os.path as op
import re
from google.protobuf import text_format
import base64
import cv2
import shutil
import argparse
import subprocess as sp
from datetime import datetime
from future.utils import viewitems
from ete3 import Tree
try:
    # py3
    from urllib.request import urlopen, Request
    from urllib.request import HTTPError
except ImportError:
    # py2
    from urllib2 import urlopen, Request
    from urllib2 import HTTPError
import copy


def print_trace():
    import traceback
    traceback.print_exc()

def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
            traceback.print_exc()
    return func_wrapper

@try_once
def try_delete(f):
    os.remove(f)

def list_to_nested_dict(xs, idxes):
    rest_idxes = set(range(len(xs[0]))).difference(idxes)
    result = {}
    for r in xs:
        curr_result = result
        for i in idxes[:-1]:
            if r[i] not in curr_result:
                curr_result[r[i]] = {}
            curr_result = curr_result[r[i]]
        key = r[idxes[-1]]
        if key not in curr_result:
            curr_result[key] = []
        value = [r[i] for i in rest_idxes]
        if len(value) == 1:
            value = value[0]
        curr_result[key].append(value)
    return result

def make_by_pattern_result(data, pattern_results):
    for p, result in pattern_results:
        match_result = re.match(p, data)
        if match_result is not None:
            return result

def make_by_pattern_maker(data, pattern_makers):
    for p, maker in pattern_makers:
        match_result = re.match(p, data)
        if match_result is not None:
            return maker()

def is_positive_uhrs_verified(r):
    uhrs = r['uhrs']
    y, n = uhrs.get('1', 0), uhrs.get('2', 0)
    return y > n

def is_negative_uhrs_verified(r):
    uhrs = r['uhrs']
    y, n = uhrs.get('1', 0), uhrs.get('2', 0)
    return n > y

def find_float_tolorance_unequal(d1, d2):
    # return a list of string. Each string means a path where the value is
    # different
    from past.builtins import basestring
    if all(isinstance(x, basestring) for x in [d1, d2]) or \
            all(type(x) is bool for x in [d1, d2]):
        if d1 != d2:
            return []
        else:
            return ['0']
    if type(d1) in [int, float] and type(d2) in [int, float]:
        equal = abs(d1 - d2) <= 0.00001 * abs(d1)
        if equal:
            return []
        else:
            return ['0']
    if type(d1) != type(d2):
        return ['0']
    if type(d1) in [dict, OrderedDict]:
        if len(d1) != len(d2):
            return ['0']
        path_d1 = dict_get_all_path(d1, with_type=True)
        result = []
        for p in path_d1:
            v1 = dict_get_path_value(d1, p, with_type=True)
            if not dict_has_path(d2, p, with_type=True):
                result.append(p)
            else:
                v2 = dict_get_path_value(d2, p, with_type=True)
                curr_result = find_float_tolorance_unequal(v1, v2)
                for r in curr_result:
                    result.append(p + '$' + r)
        return result
    elif type(d1) in [tuple, list]:
        if len(d1) != len(d2):
            return ['-1']
        result = []
        for i, (x1, x2) in enumerate(zip(d1, d2)):
            curr_result = find_float_tolorance_unequal(x1, x2)
            for r in curr_result:
                result.append('{}${}'.format(i, r))
        return result
    else:
        import torch
        if type(d1) is torch.Tensor:
            diff = (d1 - d2).abs().sum()
            s = d1.abs().sum()
            if float(s) < 1e-5:
                equal = diff < 1e-5
            else:
                equal = float(diff / s) < 1e-5
            if equal:
                return []
            else:
                import ipdb;ipdb.set_trace(context=15)
                return ['0']
        else:
            raise Exception('unknown type')

def float_tolorance_equal(d1, d2, check_order=True):
    from past.builtins import basestring
    if isinstance(d1, basestring) and isinstance(d2, basestring):
        return d1 == d2
    if type(d1) in [int, float] and type(d2) in [int, float]:
        return abs(d1 - d2) <= 0.00001 * abs(d1)
    if type(d1) != type(d2) and \
            (not (type(d1) in [tuple, list] and
                type(d2) in [tuple, list])):
        return False
    if type(d1) in [dict, OrderedDict]:
        if len(d1) != len(d2):
            return False
        for k in d1:
            if k not in d2:
                return False
            v1, v2 = d1[k], d2[k]
            if not float_tolorance_equal(v1, v2):
                return False
        return True
    elif type(d1) in [tuple, list]:
        if len(d1) != len(d2):
            return False
        if not check_order:
            d1 = sorted(d1, key=lambda x: pformat(x))
            d2 = sorted(d2, key=lambda x: pformat(x))
        for x1, x2 in zip(d1, d2):
            if not float_tolorance_equal(x1, x2, check_order):
                return False
        return True
    elif type(d1) is bool:
        return d1 == d2
    elif d1 is None:
        return d1 == d2
    elif type(d1) is datetime:
        if d1.tzinfo != d2.tzinfo:
            return d1.replace(tzinfo=d2.tzinfo) == d2
        else:
            return d1 == d2
    else:
        import torch
        if type(d1) is torch.Tensor:
            diff = (d1 - d2).abs().sum()
            s = d1.abs().sum()
            if s < 1e-5:
                return diff < 1e-5
            else:
                return diff / s < 1e-5
        else:
            raise Exception('unknown type')

def case_incensitive_overlap(all_terms):
    all_lower_to_term = [{t.lower(): t for t in terms} for terms in all_terms]
    all_lowers = [set(l.keys()) for l in all_lower_to_term]
    anchor = all_lowers[0].intersection(*all_lowers[1:])

    return [[lower_to_term[l] for l in anchor]
        for lower_to_term in all_lower_to_term]

def compile_by_docker(src_zip, docker_image, dest_zip):
    # compile the qd zip file and generate another one by compiling. so that
    # there is no need to compile it again.
    src_fname = op.basename(src_zip)
    src_folder = op.dirname(src_zip)

    docker_src_folder = '/tmpwork'
    docker_src_zip = op.join(docker_src_folder, src_fname)
    docker_out_src_fname = src_fname + '.out.zip'
    docker_out_zip = op.join(docker_src_folder, docker_out_src_fname)
    out_zip = op.join(src_folder, docker_out_src_fname)
    docker_compile_folder = '/tmpcompile'
    cmd = ['docker', 'run',
            '-v', '{}:{}'.format(src_folder, docker_src_folder),
            docker_image,
            ]
    cmd.append('/bin/bash')
    cmd.append('-c')
    compile_cmd = [
            'mkdir -p {}'.format(docker_compile_folder),
            'cd {}'.format(docker_compile_folder),
            'unzip {}'.format(docker_src_zip),
            'bash compile.aml.sh',
            'zip -yrv x.zip *',
            'cp x.zip {}'.format(docker_out_zip),
            'chmod a+rw {}'.format(docker_out_zip),
            ]
    cmd.append(' && '.join(compile_cmd))
    cmd_run(cmd)
    ensure_directory(op.dirname(dest_zip))
    copy_file(out_zip, dest_zip)

def zip_qd(out_zip):
    ensure_directory(op.dirname(out_zip))
    cmd = ['zip',
            '-yrv',
            out_zip,
            '*',
            '-x',
            '\*src/build/lib.linux-x86_64-2.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-2.7/\*',
            '-x',
            '\*build/temp.linux-x86_64-2.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.5/\*',
            '-x',
            '\*build/temp.linux-x86_64-3.5/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.7/\*',
            '-x',
            '\*assets\*',
            '-x',
            '\*build/temp.linux-x86_64-3.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.6/\*',
            '-x',
            '\*build/temp.linux-x86_64-3.6/\*',
            '-x',
            '\*src/CCSCaffe/models/\*',
            '-x',
            '\*src/CCSCaffe/data/\*',
            '-x',
            '\*src/CCSCaffe/examples/\*',
            '-x',
            'aux_data/yolo9k/\*',
            '-x',
            'visualization\*',
            '-x',
            '\*.build_release\*',
            '-x',
            '\*.build_debug\*',
            '-x',
            '\*.build\*',
            '-x',
            '\*tmp_run\*',
            '-x',
            '\*src/CCSCaffe/MSVC/\*',
            '-x',
            '\*.pyc',
            '-x',
            '\*.so',
            '-x',
            '\*.o',
            '-x',
            '\*src/CCSCaffe/docs/tutorial/\*',
            '-x',
            '\*src/CCSCaffe/matlab/\*',
            '-x',
            '\*.git\*',
            '-x',
            '\*src/qd_classifier/.cache/\*']
    cmd_run(cmd, working_dir=os.getcwd(), shell=True)

def limited_retry_agent(num, func, *args, **kwargs):
    for i in range(num):
        try:
            return func(*args, **kwargs)
        except:
            logging.info('fails: tried {}-th time'.format(i + 1))
            import time
            print_trace()
            time.sleep(5)

def retry_agent(func, *args, **kwargs):
    i = 0
    while True:
        try:
            return func(*args, **kwargs)
            break
        except Exception as e:
            logging.info('fails: try {}-th time'.format(i))
            i = i + 1
            import time
            time.sleep(5)

def ensure_copy_folder(src_folder, dst_folder):
    if op.isdir(dst_folder):
        return
    shutil.copytree(src_folder, dst_folder)

def get_current_time_as_str():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def iter_swap_param(swap_params):
    num = len(swap_params)
    counts = [len(p[1]) for p in swap_params]
    assert all(c > 0 for c in counts)
    assert all(type(p[1]) is list or type(p[1]) is tuple for p in swap_params)
    idx = [0] * num

    while True:
        result = {}
        for p, i in zip(swap_params, idx):
            dict_update_path_value(result, p[0], p[1][i])
        yield result

        for i in range(num - 1, -1, -1):
            idx[i] = idx[i] + 1
            if idx[i] < counts[i]:
                break
            else:
                idx[i] = 0
                if i == 0:
                    return

def gen_uuid():
    import uuid
    return uuid.uuid4().hex

def remove_dir(d):
    ensure_remove_dir(d)

def ensure_remove_dir(d):
    is_dir = op.isdir(d)
    is_link = op.islink(d)
    if is_dir:
        if not is_link:
            shutil.rmtree(d)
        else:
            os.unlink(d)

def split_to_chunk(all_task, num_chunk=None, num_task_each_chunk=None):
    if num_task_each_chunk is None:
        num_task_each_chunk = (len(all_task) + num_chunk - 1) // num_chunk
    result = []
    i = 0
    while True:
        start = i * num_task_each_chunk
        end = start + num_task_each_chunk
        if start >= len(all_task):
            break
        if end > len(all_task):
            end = len(all_task)
        result.append(all_task[start:end])
        i = i + 1
    return result

def hash_sha1(s):
    import hashlib
    if type(s) is not str:
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def copy_file(src, dest):
    tmp = dest + '.tmp'
    # we use rsync because it could output the progress
    cmd_run('rsync {} {} --progress'.format(src, tmp).split(' '))
    os.rename(tmp, dest)

def ensure_copy_file(src, dest):
    ensure_directory(op.dirname(dest))
    if not op.isfile(dest):
        copy_file(src, dest)

def cmd_run(list_cmd, return_output=False, env=None,
        working_dir=None,
        stdin=sp.PIPE,
        shell=False,
        dry_run=False,
        ):
    logging.info('start to cmd run: {}'.format(' '.join(map(str, list_cmd))))
    # if we dont' set stdin as sp.PIPE, it will complain the stdin is not a tty
    # device. Maybe, the reson is it is inside another process.
    # if stdout=sp.PIPE, it will not print the result in the screen
    e = os.environ.copy()
    if 'SSH_AUTH_SOCK' in e:
        del e['SSH_AUTH_SOCK']
    if working_dir:
        ensure_directory(working_dir)
    if env:
        for k in env:
            e[k] = env[k]
    if dry_run:
        # we need the log result. Thus, we do not return at teh very beginning
        return
    if not return_output:
        #if env is None:
            #p = sp.Popen(list_cmd, stdin=sp.PIPE, cwd=working_dir)
        #else:
        if shell:
            p = sp.Popen(' '.join(list_cmd),
                    stdin=stdin,
                    env=e,
                    cwd=working_dir,
                    shell=True)
        else:
            p = sp.Popen(list_cmd,
                    stdin=sp.PIPE,
                    env=e,
                    cwd=working_dir)
        message = p.communicate()
        if p.returncode != 0:
            raise ValueError(message)
    else:
        if shell:
            message = sp.check_output(' '.join(list_cmd),
                    env=e,
                    cwd=working_dir,
                    shell=True)
        else:
            message = sp.check_output(list_cmd,
                    env=e,
                    cwd=working_dir)
        logging.info('finished the cmd run')
        return message.decode('utf-8')


def parallel_map(func, all_task, isDebug=False):
    if not isDebug:
        from pathos.multiprocessing import ProcessingPool as Pool
        num_worker = 16
        m = Pool(num_worker)
        return m.map(func, all_task)
    else:
        result = []
        for t in all_task:
            result.append(func(t))
        return result

def url_to_file_by_wget(url, fname):
    ensure_directory(op.dirname(fname))
    cmd_run(['wget', url, '-O', fname])

# this is specifically for azure blob url, where the last 1k bytes operation is
# not supported. We have to first find the length and then find the start
# point
def get_url_fsize(url):
    result = cmd_run(['curl', '-sI', url], return_output=True)
    for row in result.split('\n'):
        ss = [s.strip() for s in row.split(':')]
        if len(ss) == 2 and ss[0] == 'Content-Length':
            size_in_bytes = int(ss[1])
            return size_in_bytes

def url_to_file_by_curl(url, fname, bytes_start=None, bytes_end=None):
    ensure_directory(op.dirname(fname))
    if bytes_start is None:
        bytes_start = 0
    elif bytes_start < 0:
        size = get_url_fsize(url)
        bytes_start = size + bytes_start
        if bytes_start < 0:
            bytes_start = 0
    if bytes_end is None:
        cmd_run(['curl', '-r', '{}-'.format(bytes_start),
            url, '--output', fname])
    else:
        raise NotImplementedError

def url_to_str(url):
    try:
        fp = urlopen(url, timeout=30)
        buf = fp.read()
        real_url = fp.geturl()
        if real_url != url and (not real_url.startswith('https') or
                real_url.replace('https', 'http') != url):
            logging.info('new url = {}; old = {}'.format(fp.geturl(), url))
            # the image gets redirected, which means the image is not available
            return None
        if type(buf) is str:
            # py2
            return buf
        else:
            # py3
            return buf.decode()
    except HTTPError as err:
        logging.error("url: {}; error code {}; message: {}".format(
            url, err.code, err.msg))
        return None
    except:
        import traceback
        logging.error("url: {}; unknown {}".format(
            url, traceback.format_exc()))
        return None

def image_url_to_bytes(url):
    req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
    try:
        response = urlopen(req, None, 10)
        if response.code != 200:
            logging.info("url: {}, error code: {}".format(url, response.code))
            return None
        data = response.read()
        response.close()
        return data
    except Exception as e:
        logging.info("error downloading: {}".format(e))
    return None

def str_to_image(buf):
    image = np.asarray(bytearray(buf), dtype='uint8')
    im = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return im

def url_to_image(url):
    buf = url_to_str(url)
    if buf is None:
        return None
    else:
        image = np.asarray(bytearray(buf), dtype='uint8')
        return cv2.imdecode(image, cv2.IMREAD_COLOR)


def scrape_bing(query_term, depth, trans_bg=False):
    '''
    e.g. scrape_bing('elder person', 300)
    '''
    import requests
    import xml.etree.ElementTree as ET
    format_str = \
            'http://www.bing.com/images/search?q={}&form=MONITR&qs=n&format=pbxml&first={}&count={}&fdpriority=premium&mkt=en-us'
    start = 0
    all_url = []
    while True:
        count = min(depth - start, 150)
        if count <= 0:
            break
        query_str = format_str.format(query_term, start, count)
        if trans_bg:
            query_str += "&&qft=+filterui:photo-transparent"
        start = start + count
        logging.info(query_str)
        r = requests.get(query_str, allow_redirects=True)
        content = r.content
        #content = urllib2.urlopen(query_str).read()
        root = ET.fromstring(content)
        for t in root.iter('k_AnswerDataKifResponse'):
            results = json.loads(t.text)['results']
            for r in results:
                rl = {k.lower() : r[k] for k in r}
                media_url = rl.get('mediaurl', '')
                url = rl.get('url', '')
                title = rl.get('title', '')
                all_url.append(media_url)
            break
    return all_url


def calculate_correlation_between_terms(iter1, iter2):
    label_to_num1 = {}
    label_to_num2 = {}
    ll_to_num = {}

    for (k1, str_rects1), (k2, str_rects2) in zip(iter1, iter2):
        assert k1 == k2, 'keys should be aligned ({} != {})'.format(k1, k2)
        rects1 = json.loads(str_rects1)
        rects2 = json.loads(str_rects2)
        for r in rects1:
            c = r['class']
            label_to_num1[c] = label_to_num1.get(c, 0) + 1
        for r in rects2:
            c = r['class']
            label_to_num2[c] = label_to_num2.get(c, 0) + 1
        for r1 in rects1:
            for r2 in rects2:
                i = calculate_iou(r1['rect'], r2['rect'])
                if i > 0.01:
                    k = (r1['class'], r2['class'])
                    ll_to_num[k] = ll_to_num.get(k, 0) + i
    ll_correlation = [(ll[0], ll[1], 1. * ll_to_num[ll] / (label_to_num1[ll[0]]
        + label_to_num2[ll[1]] - ll_to_num[ll]))
        for ll in ll_to_num]
    ll_correlation = [(left, right, c) for left, right, c in ll_correlation
            if left.lower() != right.lower()]
    ll_correlation = sorted(ll_correlation, key=lambda x: -x[2])

    return ll_correlation

def json_dump(obj):
    # order the keys so that each operation is deterministic though it might be
    # slower
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))

def set_if_not_exist(d, key, value):
    if key not in d:
        d[key] = value

def print_as_html(table, html_output):
    from jinja2 import Environment, FileSystemLoader
    j2_env = Environment(loader=FileSystemLoader('./'), trim_blocks=True)
    # find the cols with longest length. If it does not include all cols, then
    # append those not included
    _, cols = max([(len(table[row]), table[row]) for row in table],
            key=lambda x: x[0])
    cols = list(cols)
    for row in table:
        for c in table[row]:
            if c not in cols:
                cols.append(c)
    r = j2_env.get_template('aux_data/html_template/table_viewer.html').render(
        table=table,
        rows=table.keys(),
        cols=cols)
    write_to_file(r, html_output)

def parse_general_args():
    parser = argparse.ArgumentParser(description='General Parser')
    parser.add_argument('-c', '--config_file', help='config file',
            type=str)
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    parser.add_argument('-bp', '--base64_param', help='base64 encoded yaml format',
            type=str)
    args = parser.parse_args()
    kwargs =  {}
    if args.config_file:
        logging.info('loading parameter from {}'.format(args.config_file))
        configs = load_from_yaml_file(args.config_file)
        for k in configs:
            kwargs[k] = configs[k]
    if args.base64_param:
        configs = load_from_yaml_str(base64.b64decode(args.base64_param))
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    if args.param:
        configs = load_from_yaml_str(args.param)
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k],
                    configs[k], k))
                kwargs[k] = configs[k]
    return kwargs

class ProgressBar(object):
    def __init__(self, maxval):
        assert maxval > 0
        self.maxval = maxval

    def __enter__(self):
        self.pbar = progressbar.ProgressBar(maxval=self.maxval).start()
        return self

    def __exit__(self, t, v, traceback):
        self.update(self.maxval)
        sys.stdout.write('\n')

    def update(self, i):
        self.pbar.update(i)

def concat_files(ins, out):
    ensure_directory(op.dirname(out))
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)

def get_mpi_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

def get_mpi_size():
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))

def get_mpi_local_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))

def get_mpi_local_size():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))

def load_class_ap(full_expid, predict_file):
    # the normal map
    report_file = op.splitext(predict_file)[0] + '.report'
    fname = op.join('output', full_expid, 'snapshot', report_file +
            '.class_ap.json')
    if op.isfile(fname):
        return json.loads(read_to_buffer(fname))

    glob_pattern = op.splitext(predict_file)[0] + '.neg_aware_gmap*report'
    fnames = glob.glob(op.join('output', full_expid, 'snapshot',
        glob_pattern))
    if len(fnames) > 0 and op.isfile(fnames[0]):
        fname = fnames[0]
        result = load_from_yaml_file(fname)
        return {'overall': {'0.5': {'class_ap': result['class_ap']}}}


def calculate_ap_by_true_list(corrects, total):
    precision = (1. * np.cumsum(corrects)) / np.arange(1, 1 + len(corrects))
    if np.sum(corrects) == 0:
        return 0
    return np.sum(precision * corrects) / total

def calculate_ap_by_true_list_count_num(corrects, total):
    precision = (1. * np.cumsum(corrects)) / np.arange(1, 1 + len(corrects))
    if np.sum(corrects) == 0:
        return 0
    return np.sum(precision) / len(precision) * np.sum(corrects) / total

def calculate_weighted_ap_by_true_list(corrects, weights, total):
    precision = np.cumsum(corrects * weights) / (np.cumsum(weights) + 0.0001)
    if total == 0:
        return 0
    return np.mean(precision) * np.sum(corrects) / total

def calculate_ap_by_true_list_100(corrects, confs, total):
    precision = (1. * np.cumsum(corrects)) / map(lambda x: 100. * (1 - x) + 1, confs)
    return np.sum(precision * corrects) / total

def calculate_image_ap_weighted(predicts, gts, weights):
    corrects, _ = match_prediction_to_gt(predicts, gts)
    return calculate_weighted_ap_by_true_list(corrects, weights, len(gts))

def match_prediction_to_gt(predicts, gts, iou_th=0.5):
    matched = [False] * len(gts)
    corrects = np.zeros(len(predicts))
    match_idx = [-1] * len(predicts)
    for j, p in enumerate(predicts):
        idx_gts = [(i, g) for i, g in enumerate(gts) if not matched[i]]
        if len(idx_gts) == 0:
            # max does not support empty input
            continue
        idx_gt_ious = [(i, g, calculate_iou(p, g)) for i, g in idx_gts]
        max_idx, _, max_iou = max(idx_gt_ious, key=lambda x: x[-1])
        if max_iou > iou_th:
            matched[max_idx] = True
            corrects[j] = 1
            match_idx[j] = max_idx
    return corrects, match_idx

def calculate_image_ap(predicts, gts, count_num=False):
    '''
    a list of rects, use 2 to return more info
    '''
    corrects, _ = match_prediction_to_gt(predicts, gts)
    if not count_num:
        return calculate_ap_by_true_list(corrects, len(gts))
    else:
        return calculate_ap_by_true_list_count_num(corrects, len(gts))


def calculate_image_ap2(predicts, gts):
    '''
    a list of rects
    '''
    corrects, match_idx = match_prediction_to_gt(predicts, gts)
    return calculate_ap_by_true_list(corrects, len(gts)), match_idx

def get_parameters_by_full_expid(full_expid):
    yaml_file = op.join('output', full_expid, 'parameters.yaml')
    if not op.isfile(yaml_file):
        return None
    param = load_from_yaml_file(yaml_file)
    if 'data' not in param:
        param['data'], param['net'] = parse_data_net(full_expid,
                param['expid'])
    return param

def get_all_model_expid():
    names = os.listdir('./output')
    return names

def get_target_images(predicts, gts, cat, threshold):
    image_aps = []
    for key in predicts:
        rects = predicts[key]
        curr_gt = [g for g in gts[key] if cat == 'any' or g['class'] == cat]
        curr_pred = [p for p in predicts[key] if cat == 'any' or (p['class'] == cat and
                p['conf'] > threshold)]
        if len(curr_gt) == 0 and len(curr_pred) == 0:
            continue
        curr_pred = sorted(curr_pred, key=lambda x: -x['conf'])
        ap = calculate_image_ap([p['rect'] for p in curr_pred],
                [g['rect'] for g in curr_gt])
        image_aps.append((key, ap))
    image_aps = sorted(image_aps, key=lambda x: x[1])
    #image_aps = sorted(image_aps, key=lambda x: -x[1])
    target_images = [key for key, ap in image_aps]
    return target_images, image_aps

def readable_confusion_entry(entry):
    '''
    entry: dictionary, key: label, value: count
    '''
    label_count = [(label, entry[label]) for label in entry]
    label_count.sort(key=lambda x: -x[1])
    total = sum([count for label, count in label_count])
    percent = [1. * count / total for label, count in label_count]
    cum_percent = np.cumsum(percent)
    items = []
    for i, (label, count) in enumerate(label_count):
        if i >= 5:
            continue
        items.append((label, '{}'.format(count), '{:.1f}'.format(100. *
            percent[i]),
            '{:.1f}'.format(100. * cum_percent[i])))
    return items

def get_all_tree_data():
    names = sorted(os.listdir('./data'))
    return [name for name in names
        if op.isfile(op.join('data', name, 'root_enriched.yaml'))]

def parse_test_data_with_version_with_more_param(predict_file):
    pattern = \
        'model(?:_iter)?_-?[0-9]*[e]?\.(?:caffemodel|pth\.tar|pth|pt)\.(.*)\.(trainval|train|test)\..*?(\.v[0-9])?\.(?:predict|report)'
    match_result = re.match(pattern, predict_file)
    if match_result:
        assert match_result
        result = match_result.groups()
        if result[2] is None:
            v = 0
        else:
            v = int(result[2][2])
        return result[0], result[1], v

def parse_test_data_with_version(predict_file):
    # with version
    result = parse_test_data_with_version_with_more_param(predict_file)
    if result is not None:
        return result
    pattern = \
        'model(?:_iter)?_-?[0-9]*[e]?\.(?:caffemodel|pth\.tar|pth|pt)\.(.*)\.(trainval|train|test)\.(\.v[0-9])?(?:predict|report)'
    match_result = re.match(pattern, predict_file)
    if match_result is None:
        pattern = \
            'model(?:_iter)?_-?[0-9]*[e]?\.(?:caffemodel|pth\.tar|pth|pt)\.([^\.]*).*?(\.v[0-9])?\.(?:predict|report)'
        match_result = re.match(pattern, predict_file)
        assert match_result
        result = match_result.groups()
        if result[1] is None:
            v = 0
        else:
            v = int(result[1][2])
        return result[0], 'test', v
    else:
        assert match_result
        result = match_result.groups()
        if result[2] is None:
            v = 0
        else:
            v = int(result[2][2])
        return result[0], result[1], v

def parse_test_data(predict_file):
    return parse_test_data_with_version(predict_file)[:2]

def parse_data(full_expid):
    all_data = os.listdir('data/')
    candidates = [data for data in all_data if full_expid.startswith(data)]
    max_length = max([len(c) for c in candidates])
    return [c for c in candidates if len(c) == max_length][0]

def parse_iteration(file_name):
    patterns = ['.*model(?:_iter)?_([0-9]*)\..*',
                '.*model(?:_iter)?_([0-9]*)e\..*',
                ]
    for p in patterns:
        r = re.match(p, file_name)
        if r is not None:
            return int(float(r.groups()[0]))
    logging.info('unable to parse the iterations for {}'.format(file_name))
    return -2

def parse_snapshot_rank(predict_file):
    '''
    it could be iteration, or epoch
    '''
    pattern = 'model_iter_([0-9]*)e*\.|model_([0-9]*)e*\.pth'
    match_result = re.match(pattern, predict_file)
    if match_result is None:
        return -1
    else:
        matched_iters = [r for r in match_result.groups() if r is not None]
        assert len(matched_iters) == 1
        return int(matched_iters[0])

def get_all_predict_files(full_expid):
    model_folder = op.join('output', full_expid, 'snapshot')

    predict_files = []

    found = glob.glob(op.join(model_folder, '*.predict'))
    predict_files.extend([op.basename(f) for f in found])

    found = glob.glob(op.join(model_folder, '*.predict.tsv'))
    predict_files.extend([op.basename(f) for f in found])

    iterations = [(parse_snapshot_rank(p), p) for p in predict_files]
    iterations.sort(key=lambda x: -x[0])
    return [p for i, p in iterations]

def dict_to_list(d, idx):
    result = []
    for k in d:
        vs = d[k]
        for v in vs:
            try:
                r = []
                # if v is a list or tuple
                r.extend(v[:idx])
                r.append(k)
                r.extend(v[idx: ])
            except TypeError:
                r = []
                if idx == 0:
                    r.append(k)
                    r.append(v)
                else:
                    assert idx == 1
                    r.append(v)
                    r.append(k)
            result.append(r)
    return result

def list_to_dict_unique(l, idx):
    result = list_to_dict(l, idx)
    for key in result:
        result[key] = list(set(result[key]))
    return result

def list_to_dict(l, idx, keep_one=False):
    result = OrderedDict()
    for x in l:
        if x[idx] not in result:
            result[x[idx]] = []
        y = x[:idx] + x[idx + 1:]
        if not keep_one and len(y) == 1:
            y = y[0]
        result[x[idx]].append(y)
    return result

def generate_lineidx(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0;
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n");
            tsvin.readline()
            fpos = tsvin.tell();
    os.rename(idxout_tmp, idxout)

def drop_second_batch_in_bn(net):
    assert net.layer[0].type == 'TsvBoxData'
    assert net.layer[1].type == 'TsvBoxData'
    slice_batch_layers = [l for l in net.layer if l.name == 'slice_batch']
    assert len(slice_batch_layers) == 1
    slice_batch_layer = slice_batch_layers[0]
    slice_point = slice_batch_layer.slice_param.slice_point[0]

    for i, l in enumerate(net.layer):
        if l.type == 'BatchNorm':
            top_name = l.top[0]
            top_name2 = top_name + '_n'
            l.top[0] = top_name2
            for m in net.layer[i + 1:]:
                for j, b in enumerate(m.bottom):
                    if b == top_name:
                        m.bottom[j] = top_name2
                for j, t in enumerate(m.top):
                    if t == top_name:
                        m.top[j] = top_name2

    all_origin_layer = []
    for l in net.layer:
        all_origin_layer.append(l)
    all_layer = []
    for l in all_origin_layer:
        if l.type != 'BatchNorm':
            all_layer.append(l)
            continue
        bn_input = l.bottom[0]
        bn_output = l.top[0]

        slice_layer = net.layer.add()
        slice_layer.name = l.name + '/slice'
        slice_layer.type = 'Slice'
        slice_layer.bottom.append(bn_input)
        slice_layer.top.append(l.name + '/slice0')
        slice_layer.top.append(l.name + '/slice1')
        slice_layer.slice_param.axis = 0
        slice_layer.slice_param.slice_point.append(slice_point)
        all_layer.append(slice_layer)

        l.bottom.remove(l.bottom[0])
        l.bottom.append(l.name + '/slice0')
        l.top.remove(l.top[0])
        l.top.append(l.name + '/slice0')
        all_layer.append(l)

        fix_bn_layer = net.layer.add()
        fix_bn_layer.name = l.name + '/bn1'
        fix_bn_layer.bottom.append(l.name + '/slice1')
        fix_bn_layer.top.append(l.name + '/slice1')
        fix_bn_layer.type = 'BatchNorm'
        for _ in range(3):
            p = fix_bn_layer.param.add()
            p.lr_mult = 0
            p.decay_mult = 0
        fix_bn_layer.batch_norm_param.use_global_stats = True
        all_layer.append(fix_bn_layer)

        cat_layer = net.layer.add()
        cat_layer.name = l.name + '/concat'
        cat_layer.type = 'Concat'
        cat_layer.bottom.append(l.name + '/slice0')
        cat_layer.bottom.append(l.name + '/slice1')
        cat_layer.top.append(bn_output)
        cat_layer.concat_param.axis = 0
        all_layer.append(cat_layer)

    while len(net.layer) > 0:
        net.layer.remove(net.layer[0])
    net.layer.extend(all_layer)

def fix_net_bn_layers(net, num_bn_fix):
    for l in net.layer:
        if l.type == 'BatchNorm':
            if num_bn_fix > 0:
                l.batch_norm_param.use_global_stats = True
                num_bn_fix = num_bn_fix - 1
            else:
                break

def is_cluster(ssh_info):
    return '-p' in ssh_info and '-i' not in ssh_info

def visualize_net(net):
    delta = 0.000001
    data_values = []
    for key in net.blobs:
        data_value = np.mean(np.abs(net.blobs[key].data))
        data_values.append(data_value + delta)
    diff_values = []
    for key in net.blobs:
        diff_values.append(np.mean(np.abs(net.blobs[key].diff))
            + delta)
    param_keys = []
    param_data = []
    for key in net.params:
        for i, b in enumerate(net.params[key]):
            param_keys.append('{}_{}'.format(key, i))
            param_data.append(np.mean(np.abs(b.data)) + delta)
    param_diff = []
    for key in net.params:
        for i, b in enumerate(net.params[key]):
            param_diff.append(np.mean(np.abs(b.diff)) + delta)

    xs = range(len(net.blobs))
    plt.gcf().clear()
    plt.subplot(2, 1, 1)

    plt.semilogy(xs, data_values, 'r-o')
    plt.semilogy(xs, diff_values, 'b-*')
    plt.xticks(xs, net.blobs.keys(), rotation='vertical')
    plt.grid()

    plt.subplot(2, 1, 2)
    xs = range(len(param_keys))
    plt.semilogy(xs, param_data, 'r-o')
    plt.semilogy(xs, param_diff, 'b-*')
    plt.xticks(xs, param_keys, rotation='vertical')
    plt.grid()
    plt.draw()
    plt.pause(0.001)

def visualize_train(solver):
    plt.figure()
    features = []
    for i in range(100):
        visualize_net(solver.net)
        solver.step(10)

def network_input_to_image(data, mean_value):
    all_im = []
    for d in data:
        im = (d.transpose((1, 2, 0)) + np.asarray(mean_value).reshape(1, 1,
            3)).astype(np.uint8).copy()
        all_im.append(im)
    return all_im

def remove_data_augmentation(data_layer):
    assert data_layer.type == 'TsvBoxData'
    data_layer.box_data_param.jitter = 0
    data_layer.box_data_param.hue = 0
    data_layer.box_data_param.exposure = 1
    data_layer.box_data_param.random_scale_min = 1
    data_layer.box_data_param.random_scale_max = 1
    data_layer.box_data_param.fix_offset = True
    data_layer.box_data_param.saturation = True

def check_best_iou(biases, gt_w, gt_h, n):
    def iou(gt_w, gt_h, w, h):
        inter = min(gt_w, w) * min(gt_h, h)
        return inter / (gt_w * gt_h + w * h - inter)

    best_iou = -1
    best_n = -1
    for i in range(len(biases) / 2):
        u = iou(gt_w, gt_h, biases[2 * i], biases[2 * i + 1])
        if u > best_iou:
            best_iou = u
            best_n = i
    assert best_n == n

def calculate_iou1(rect0, rect1):
    '''
    x0, y1, x2, y3
    '''
    w = min(rect0[2], rect1[2]) - max(rect0[0], rect1[0]) + 1
    if w < 0:
        return 0
    h = min(rect0[3], rect1[3]) - max(rect0[1], rect1[1]) + 1
    if h < 0:
        return 0
    i = w * h
    a1 = (rect1[2] - rect1[0] + 1) * (rect1[3] - rect1[1] + 1)
    a0 = (rect0[2] - rect0[0] + 1) * (rect0[3] - rect0[1] + 1)
    if a0 == 0 and a1 == 0 and i == 0:
        return 1.
    return 1. * i / (a0 + a1 - i)

def calculate_iou(rect0, rect1):
    '''
    x0, y1, x2, y3
    '''
    w = min(rect0[2], rect1[2]) - max(rect0[0], rect1[0])
    if w < 0:
        return 0
    h = min(rect0[3], rect1[3]) - max(rect0[1], rect1[1])
    if h < 0:
        return 0
    i = w * h
    a1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    a0 = (rect0[2] - rect0[0]) * (rect0[3] - rect0[1])
    if a0 == 0 and a1 == 0 and i == 0:
        return 1.
    return 1. * i / (a0 + a1 - i)

def process_run(func, *args, **kwargs):
    def internal_func(queue):
        result = func(*args, **kwargs)
        queue.put(result)
    queue = mp.Queue()
    p = Process(target=internal_func, args=(queue,))
    p.start()
    p.join()
    assert p.exitcode == 0
    return queue.get()

def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)
    try:
        yaml.add_representer(unicode, unicode_representer)
    except NameError:
        logging.info('python 3 env')

def init_logging():
    np.seterr(divide = "raise", over="warn", under="warn",  invalid="raise")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
    ch.setFormatter(logger_fmt)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(ch)
    root.setLevel(logging.INFO)

    setup_yaml()

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise
        # we should always check if it succeeds.
        assert op.isdir(op.abspath(path)), path

def parse_pattern(pattern, s):
    result = re.search(pattern, s)
    if result is None:
        return result
    return [float(g) for g in result.groups()]

def parse_yolo_log(log_file):
    pattern = 'loss_xy: ([0-9, .]*); loss_wh: ([0-9, .]*); '
    pattern = pattern + 'loss_objness: ([0-9, .]*); loss_class: ([0-9, .]*)'

    base_log_lines = read_lines(log_file)
    xys = []
    whs = []
    loss_objnesses = []
    loss_classes = []
    for line in base_log_lines:
        gs = parse_pattern(pattern, line)
        if gs is None:
            continue
        idx = 0
        xys.append(float(gs[idx]))
        idx = idx + 1
        whs.append(float(gs[idx]))
        idx = idx + 1
        loss_objnesses.append(float(gs[idx]))
        idx = idx + 1
        loss_classes.append(float(gs[idx]))

    return xys, whs, loss_objnesses, loss_classes

def parse_nums(p, log_file):
    result = []
    for line in read_lines(log_file):
        gs = parse_pattern(p, line)
        if gs is None:
            continue
        result.append(gs)
    return result

def parse_yolo_log_st(log_file):
    p = 'region_loss_layer\.cpp:1138] ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*)'
    ss = parse_nums(p, log_file)
    p = 'region_loss_layer\.cpp:1140] ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*)'
    tt = parse_nums(p, log_file)
    return ss, tt

def parse_yolo_log_acc(log_file):
    p = 'Region Avg IOU: ([0-9, .]*), Class: ([0-9, .]*), '
    p = p + 'Obj: ([0-9, .]*), No Obj: ([0-9, .]*), Avg Recall: ([0-9, .]*),  count: ([0-9]*)'
    all_ious = []
    all_probs = []
    all_obj = []
    all_noobj = []
    all_recall = []
    all_count = []
    for line in read_lines(log_file):
        gs = parse_pattern(p, line)
        if gs is None:
            continue
        all_ious.append(gs[0])
        all_probs.append(gs[1])
        all_obj.append(gs[2])
        all_noobj.append(gs[3])
        all_recall.append(gs[4])
        all_count.append(gs[5])
    return all_ious, all_probs, all_obj, all_noobj, all_recall, all_count


def read_lines(file_name):
    with open(file_name, 'r') as fp:
        for line in fp:
            yield line

def read_to_buffer(file_name):
    with open(file_name, 'rb') as fp:
        all_line = fp.read()
    return all_line

class Model(object):
    def __init__(self, test_proto_file, train_proto_file, model_param, mean_value, scale, model_iter):
        self.test_proto_file = test_proto_file
        self.model_param = model_param
        self.mean_value = mean_value
        self.model_iter = model_iter
        self.scale = scale
        self.train_proto_file = train_proto_file

def adjust_tree_prediction_threshold(n, tree_th):
    found = False
    for l in n.layer:
        if l.type == 'SoftmaxTreePrediction':
            found = True
            l.softmaxtreeprediction_param.threshold = tree_th
    assert found

def remove_nms(n):
    for l in n.layer:
        if l.type == 'RegionOutput':
            l.region_output_param.nms = -1
        if l.type == 'RegionPrediction':
            l.region_prediction_param.nms = -1

def update_conv_channels(net, factor, skip):
    c = 0
    s = 0
    for l in net.layer:
        if l.type == 'Convolution':
            if s < skip:
                s = s + 1
                continue
            o = l.convolution_param.num_output
            l.convolution_param.num_output = int(o * factor)
            c = c + 1
    logging.info('updated {} layers for channel factor'.format(c))

def get_channel(net, blob_name):
    for l in net.layer:
        if l.type == 'Convolution':
            assert len(l.top) == 1
            if l.top[0] == blob_name:
                return l.convolution_param.num_output
    assert False, 'not found'

def fix_net_parameters(net, last_fixed_param):
    found = False
    no_param_layers = set(['TsvBoxData', 'ReLU', 'Pooling', 'Reshape',
            'EuclideanLoss', 'Sigmoid'])
    unknown_layers = []
    for l in net.layer:
        if l.type == 'Convolution' or l.type == 'Scale':
            if l.type == 'Convolution':
                assert len(l.param) >= 1
            else:
                if len(l.param) == 0:
                    p = l.param.add()
                    p.lr_mult = 0
                    p.decay_mult = 0
                    if l.scale_param.bias_term:
                        p = l.param.add()
                        p.lr_mult = 0
                        p.decay_mult = 0
            for p in l.param:
                p.lr_mult = 0
                p.decay_mult = 0
        elif l.type == 'BatchNorm':
            l.batch_norm_param.use_global_stats = True
        else:
            if l.type not in no_param_layers:
                unknown_layers.append(l.type)
        if l.name == last_fixed_param:
            for b in l.bottom:
                l.propagate_down.append(False)
            found = True
            break
    assert len(unknown_layers) == 0, ', '.join(unknown_layers)
    assert found

def set_no_bias(net, layer_name):
    for l in net.layer:
        if l.name == layer_name:
            assert l.type == 'Convolution'
            l.convolution_param.bias_term = False
            if len(l.param) == 2:
                del l.param[1]
            else:
                assert len(l.param) == 0
            return
    assert False

def add_yolo_angular_loss_regularizer(net, **kwargs):
    for l in net.layer:
        if l.name == 'angular_loss':
            logging.info('angular loss exists')
            return
    conf_layer = None
    for l in net.layer:
        if l.name == 'conf':
            conf_layer = l
            assert 'conf' in l.top
    found_t_label = False
    for l in net.layer:
        if 't_label' in l.top:
            found_t_label = True
            break
    assert conf_layer and found_t_label

    conf_layer.param[0].name = 'conf_w'
    CA = conf_layer.convolution_param.num_output
    assert len(conf_layer.bottom) == 1
    num_feature = get_channel(net, conf_layer.bottom[0])

    param_layer = net.layer.add()
    param_layer.name = 'param_conf_w'
    param_layer.type = 'Parameter'
    param_layer.parameter_param.shape.dim.append(CA)
    param_layer.parameter_param.shape.dim.append(num_feature)
    param_layer.parameter_param.shape.dim.append(1)
    param_layer.parameter_param.shape.dim.append(1)
    param_layer.top.append('conf_w')
    p = param_layer.param.add()
    p.name = 'conf_w'

    layer = net.layer.add()
    layer.name = 'angular_loss'
    layer.type = 'Python'
    layer.bottom.append(conf_layer.bottom[0])
    layer.bottom.append('t_label')
    layer.bottom.append('conf_w')
    layer.python_param.module = 'kcenter_exp'
    layer.python_param.layer = 'YoloAngularLossLayer'
    layer.propagate_down.append(True)
    layer.propagate_down.append(False)
    layer.propagate_down.append(False)
    layer.top.append('angular_loss')
    weight = kwargs.get('yolo_angular_loss_weight', 1)
    layer.loss_weight.append(weight)

def add_yolo_low_shot_regularizer(net, low_shot_label_idx):
    assert net.layer[-1].type == 'RegionLoss'
    assert net.layer[-2].type == 'Convolution'
    assert net.layer[-1].bottom[0] == net.layer[-2].top[0]
    assert net.layer[-2].convolution_param.kernel_size[0] == 1
    assert net.layer[-2].convolution_param.kernel_h == 0
    assert net.layer[-2].convolution_param.kernel_w == 0

    num_classes = net.layer[-1].region_loss_param.classes
    num_anchor = len(net.layer[-1].region_loss_param.biases) / 2

    param_dim1 = net.layer[-2].convolution_param.num_output
    param_dim2 = get_channel(net, net.layer[-2].bottom[0])

    # add the parameter name into the convolutional layer
    last_conv_param_name = 'last_conv_param_low_shot'
    net.layer[-2].param[0].name = last_conv_param_name

    # add the parameter layer to expose the parameter
    param_layer = net.layer.add()
    param_layer.type = 'Parameter'
    param_layer.name = 'param_last_conv'
    param_layer.top.append(last_conv_param_name)
    p = param_layer.param.add()
    p.name = last_conv_param_name
    p.lr_mult = 1
    p.decay_mult = 1
    param_layer.parameter_param.shape.dim.append(param_dim1)
    param_layer.parameter_param.shape.dim.append(param_dim2)

    # add the regularizer layer
    reg_layer = net.layer.add()
    reg_layer.type = 'Python'
    reg_layer.name = 'equal_norm'
    reg_layer.bottom.append(last_conv_param_name)
    reg_layer.top.append('equal_norm')
    reg_layer.loss_weight.append(1)
    reg_layer.python_param.module = 'equal_norm_loss'
    reg_layer.python_param.layer = 'YoloAlignNormToBaseLossLayer'
    reg_param = {'num_classes': num_classes,
            'low_shot_label_idx': low_shot_label_idx,
            'num_anchor': num_anchor}
    reg_layer.python_param.param_str = json.dumps(reg_param)

def update_kernel_active(net, kernel_active, kernel_active_skip):
    assert False, 'use update_kernel_active2'
    c = 0
    skipped = 0
    logging.info('{}-{}'.format(kernel_active, kernel_active_skip));
    for l in net.layer:
        if l.type == 'Convolution':
            if skipped < kernel_active_skip:
                skipped = skipped + 1
                logging.info('skiping to update active kernel')
                continue
            l.convolution_param.kernel_active = kernel_active
            c = c + 1

    logging.info('update {} layers'.format(c))


def plot_to_file(xs, ys, file_name, **kwargs):
    fig = plt.figure()
    semilogy = kwargs.get('semilogy')
    if all(isinstance(x, str) for x in xs):
        xs2 = range(len(xs))
        #plt.xticks(xs2, xs, rotation=15, ha='right')
        plt.xticks(xs2, xs, rotation='vertical')
        xs = xs2
    if type(ys) is dict:
        for key in ys:
            if semilogy:
                plt.semilogy(xs, ys[key], '-o')
            else:
                plt.plot(xs, ys[key], '-o')
    else:
        if semilogy:
            plt.semilogy(xs, ys, '-o')
        else:
            plt.plot(xs, ys, '-o')
    plt.grid()
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    ensure_directory(op.dirname(file_name))
    plt.tight_layout()
    # explicitly remove the file because philly does not support overwrite
    if op.isfile(file_name):
        try:
            os.remove(file_name)
        except:
            logging.info('{} exists but could not be deleted'.format(
                file_name))
    fig.savefig(file_name)
    plt.close(fig)

def parse_training_time(log_file):
    log = read_to_buffer(log_file)
    all_time_cost = []
    all_iters = []
    for line in log.split('\n'):
        m = re.match('.*Iteration.*iter\/s, ([0-9\.]*)s\/([0-9]*) iters.*', line)
        if m:
            r = m.groups()
            time_cost = float(r[0])
            iters = float(r[1])
            all_iters.append(iters)
            all_time_cost.append(time_cost)
    return all_iters, all_time_cost

def encode_expid(prefix, *args):
    parts = [prefix]
    for (t, a) in args:
        p = ''
        if a != None:
            if type(a) == str:
                a = a.replace(':', '_')
            if t != None and len(t) > 0:
                p = p + '_{}'.format(t)
            p = p + '_{}'.format(a)
        parts.append(p)
    return ''.join(parts)

def unicode_representer(dumper, uni):
    node = yaml.ScalarNode(tag=u'tag:yaml.org,2002:str', value=uni)
    return node

def dump_to_yaml_str(context):
    return yaml.dump(context, default_flow_style=False,
            encoding='utf-8', allow_unicode=True)

def write_to_yaml_file(context, file_name):
    ensure_directory(op.dirname(file_name))
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, default_flow_style=False,
                encoding='utf-8', allow_unicode=True)

def load_from_yaml_str(s):
    return yaml.load(s, Loader=yaml.CLoader)

def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)

def write_to_file(contxt, file_name):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    if type(contxt) is str:
        contxt = contxt.encode()
    with open(file_name, 'wb') as fp:
        fp.write(contxt)

def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

class LoopProcess(Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        '''
        same signiture with Process.__init__
        The process will keep running the function of target and will wait for
        several seconds in between. This is useful to run some monitoring job
        or regular job
        '''
        super(LoopProcess, self).__init__(group, target, name, args, kwargs)
        self._exit = Event()

    def run(self):
        sleep_time = 5
        while not self._exit.is_set():
            if self._target:
                self._target(*self._args, **self._kwargs)
            time.sleep(sleep_time)

    def init_shutdown(self):
        self._exit.set()

class PyTee(object):
    def __init__(self, logstream, stream_name):
        valid_streams = ['stderr','stdout'];
        if  stream_name not in valid_streams:
            raise IOError("valid stream names are %s" % ', '.join(valid_streams))
        self.logstream =  logstream
        self.stream_name = stream_name;
    def __del__(self):
        pass;
    def write(self, data):  #tee stdout
        self.logstream.write(data);
        self.fstream.write(data);
        self.logstream.flush();
        self.fstream.flush();

    def flush(self):
        self.logstream.flush();
        self.fstream.flush();

    def __enter__(self):
        if self.stream_name=='stdout' :
            self.fstream   =  sys.stdout
            sys.stdout = self;
        else:
            self.fstream   =  sys.stderr
            sys.stderr = self;
        self.fstream.flush();
    def __exit__(self, _type, _value, _traceback):
        if self.stream_name=='stdout' :
            sys.stdout = self.fstream;
        else:
            sys.stderr = self.fstream;

def parse_basemodel_with_depth(net):
    '''
    darknet19->darknet19
    darknet19_abc->darknet19
    '''
    if '_' not in net:
        return net
    else:
        i = net.index('_')
        return net[: i]

def worth_create(base_file_name, derived_file_name, buf_second=0):
    if not op.isfile(base_file_name):
        return False
    if os.path.isfile(derived_file_name) and \
            os.path.getmtime(derived_file_name) > os.path.getmtime(base_file_name) - buf_second:
        return False
    else:
        return True

def basename_no_ext(file_name):
    return op.splitext(op.basename(file_name))[0]

def default_data_path(dataset):
    '''
    use TSVDataset instead
    '''
    proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)));
    result = {}
    data_root = os.path.join(proj_root, 'data', dataset)
    result['data_root'] = data_root
    result['source'] = os.path.join(data_root, 'train.tsv')
    result['trainval'] = op.join(data_root, 'trainval.tsv')
    result['test_source'] = os.path.join(data_root, 'test.tsv')
    result['labelmap'] = os.path.join(data_root, 'labelmap.txt')
    result['source_idx'] = os.path.join(data_root, 'train.lineidx')
    result['test_source_idx'] = os.path.join(data_root, 'test.lineidx')
    return result

class FileProgressingbar:
    fileobj = None
    pbar = None
    def __init__(self, fileobj, keyword='Test'):
        fileobj.seek(0,os.SEEK_END)
        flen = fileobj.tell()
        fileobj.seek(0,os.SEEK_SET)
        self.fileobj = fileobj
        widgets = ['{}: '.format(keyword), progressbar.AnimatedMarker(),' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=flen).start()
    def update(self):
        self.pbar.update(self.fileobj.tell())

def encoded_from_img(im, quality=None):
    if quality:
        x = cv2.imencode('.jpg', im, (cv2.IMWRITE_JPEG_QUALITY, quality))[1]
    else:
        x = cv2.imencode('.jpg', im)[1]
    return base64.b64encode(x)

def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        return r
    except:
        return None;

def img_from_base64_ignore_rotation(str_im):
    jpgbytestring = base64.b64decode(str_im)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_IGNORE_ORIENTATION);
    return im

def int_rect(rect, enlarge_factor=1.0, im_h=None, im_w=None):
    assert(len(rect) == 4)
    left, top, right, bot = rect
    rw = right - left
    rh = bot - top

    new_x = int(left + (1.0 - enlarge_factor) * rw / 2.0)
    new_y = int(top + (1.0 - enlarge_factor) * rh / 2.0)
    new_w = int(math.ceil(enlarge_factor * rw))
    new_h = int(math.ceil(enlarge_factor * rh))
    if im_h and im_w:
        new_x = np.clip(new_x, 0, im_w)
        new_y = np.clip(new_y, 0, im_h)
        new_w = np.clip(new_w, 0, im_w - new_x)
        new_h = np.clip(new_h, 0, im_h - new_y)

    return list(map(int, [new_x, new_y, new_x + new_w, new_y + new_h]))

def is_valid_rect(rect):
    return len(rect) == 4 and rect[0] < rect[2] and rect[1] < rect[3]

def pass_key_value_if_has(d_from, from_key, d_to, to_key):
    if from_key in d_from:
        d_to[to_key] = d_from[from_key]

def dict_update_nested_dict(a, b, overwrite=True):
    for k, v in viewitems(b):
        if k not in a:
            a[k] = v
        else:
            if isinstance(a[k], dict) and isinstance(v, dict):
                dict_update_nested_dict(a[k], v, overwrite)
            else:
                if overwrite:
                    a[k] = v

def dict_ensure_path_key_converted(a):
    for k in list(a.keys()):
        v = a[k]
        if '$' in k:
            parts = k.split('$')
            x = {}
            x_curr = x
            for p in parts[:-1]:
                x_curr[p] = {}
                x_curr = x_curr[p]
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)
            x_curr[parts[-1]] = v
            dict_update_nested_dict(a, x)
            del a[k]
        else:
            if isinstance(v, dict):
                dict_ensure_path_key_converted(v)

def dict_get_all_path(d, with_type=False):
    all_path = []
    for k, v in viewitems(d):
        if with_type:
            if type(k) is str:
                k = 's' + k
            elif type(k) is int:
                k = 'i' + str(k)
            else:
                raise NotImplementedError
        if not isinstance(v, dict):
            all_path.append(k)
        else:
            all_sub_path = dict_get_all_path(v, with_type)
            all_path.extend([k + '$' + p for p in all_sub_path])
    return all_path

def dict_parse_key(k, with_type):
    if with_type:
        if k[0] == 'i':
            return int(k[1:])
        else:
            return k[1:]
    return k

def dict_has_path(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            if not isinstance(cur_dict, dict):
                return False
            k = dict_parse_key(ps[0], with_type)
            if k in cur_dict:
                cur_dict = cur_dict[k]
                ps = ps[1:]
            else:
                return False
        else:
            return True


def dict_set_path_if_not_exist(param, k, v):
    if not dict_has_path(param, k):
        dict_update_path_value(param, k, v)

def dict_update_path_value(d, p, v):
    ps = p.split('$')
    while True:
        if len(ps) == 1:
            d[ps[0]] = v
            break
        else:
            if ps[0] not in d:
                d[ps[0]] = {}
            d = d[ps[0]]
            ps = ps[1:]

def dict_remove_path(d, p):
    ps = p.split('$')
    assert len(ps) > 0
    cur_dict = d
    need_delete = ()
    while True:
        if len(ps) == 1:
            if len(need_delete) > 0 and len(cur_dict) == 1:
                del need_delete[0][need_delete[1]]
            else:
                del cur_dict[ps[0]]
            return
        else:
            if len(cur_dict) == 1:
                if len(need_delete) == 0:
                    need_delete = (cur_dict, ps[0])
            else:
                need_delete = (cur_dict, ps[0])
            cur_dict = cur_dict[ps[0]]
            ps = ps[1:]

def dict_get_path_value(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            cur_dict = cur_dict[k]
            ps = ps[1:]
        else:
            return cur_dict

def get_file_size(f):
    if not op.isfile(f):
        return 0
    return os.stat(f).st_size

def convert_to_yaml_friendly(result):
    if type(result) is dict:
        for key, value in result.items():
            if isinstance(value, dict):
                result[key] = convert_to_yaml_friendly(value)
            elif isinstance(value, np.floating):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                raise NotImplementedError()
            elif type(value) in [int, str, float, bool]:
                continue
            else:
                raise NotImplementedError()
    else:
        raise NotImplementedError()
    return result

def natural_key(text):
    import re
    result = []
    for c in re.split(r'([0-9]+(?:[.][0-9]*)?)', text):
        try:
            result.append(float(c))
        except:
            continue
    return result

def natural_sort(strs):
    strs.sort(key=natural_key)

def get_pca(x, com):
    x -= np.mean(x, axis = 0)
    cov = np.cov(x, rowvar=False)
    from scipy import linalg as LA
    evals , evecs = LA.eigh(cov)
    total_val = np.sum(evals)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    component_val = np.sum(evals[:com])
    logging.info('kept: {}/{}={}'.format(component_val,
            total_val, component_val / total_val))
    a = np.dot(x, evecs[:, :com])
    return a

def plot_distribution(x, y, color=None, fname=None):
    import seaborn as sns
    x = sns.jointplot(x, y, kind='kde',
            color=color)
    if fname:
        x.savefig(fname)
        plt.close()
    else:
        plt.show()

def run_if_not_cached(func, *args, **kwargs):
    import pickle as pkl
    key = hash_sha1(pkl.dumps(OrderedDict({'arg': args, 'kwargs': kwargs, 'func_name':
        func.__name__})))
    cache_folder = op.expanduser('./output/run_if_not_cached/')
    cache_file = op.join(cache_folder, key)

    if op.isfile(cache_file):
        logging.info('loading {}'.format(cache_file))
        return pkl.loads(read_to_buffer(cache_file))
    else:
        result = func(*args, **kwargs)
        logging.info('caching to file: {}'.format(cache_file))
        write_to_file(pkl.dumps(result), cache_file)
        return result

def convert_to_command_line(param, script):
    logging.info(pformat(param))
    x = copy.deepcopy(param)
    from qd.qd_common import dump_to_yaml_str
    result = "python {} -bp {}".format(
            script,
            base64.b64encode(dump_to_yaml_str(x)).decode())
    return result

def print_table(a_to_bs, all_key=None):
    all_line = get_table_print_lines(a_to_bs, all_key)
    logging.info('\n{}'.format('\n'.join(all_line)))

def get_table_print_lines(a_to_bs, all_key):
    if len(a_to_bs) == 0:
        logging.info('no rows')
        return []
    if not all_key:
        all_key = []
        for a_to_b in a_to_bs:
            all_key.extend(a_to_b.keys())
        all_key = list(set(all_key))
    all_width = [max([len(str(a_to_b.get(k, ''))) for a_to_b in a_to_bs] +
        [len(k)]) for k in all_key]
    row_format = ' '.join(['{{:{}}}'.format(w) for w in all_width])

    all_line = []
    line = row_format.format(*all_key)
    all_line.append(line.strip())
    for a_to_b in a_to_bs:
        line = row_format.format(*[str(a_to_b.get(k, '')) for k in all_key])
        all_line.append(line)
    return all_line

def is_hvd_initialized():
    try:
        import horovod.torch as hvd
        hvd.size()
        return True
    except ImportError:
        return False
    except ValueError:
        return False

def get_user_name():
    import getpass
    return getpass.getuser()

def decode_general_cmd(extraParam):
    re_result = re.match('.*python (?:scripts|src)/.*\.py -bp (.*)', extraParam)
    if re_result and len(re_result.groups()) == 1:
        ps = load_from_yaml_str(base64.b64decode(re_result.groups()[0]))
        return ps

def print_job_infos(all_job_info):
    all_key = [
            'cluster',
            'status', 'appID-s', 'elapsedTime', 'elapsedFinished',
            'retries', 'preempts', 'mem_used', 'gpu_util',
            'speed', 'left']
    keys = ['data', 'net', 'expid']
    meta_keys = ['num_gpu']
    all_key.extend(keys)
    all_key.extend(meta_keys)

    # find the keys whose values are the same
    def all_equal(x):
        assert len(x) > 0
        return all(y == x[0] for y in x[1:])

    if len(all_job_info) > 1:
        equal_keys = [k for k in all_key if all_equal([j.get(k) for j in all_job_info])]
        if len(equal_keys) > 0:
            logging.info('equal key values for all jobs')
            print_table(all_job_info[0:1], all_key=equal_keys)
        all_key = [k for k in all_key if not all_equal([j.get(k) for j in all_job_info])]

    print_table(all_job_info, all_key=all_key)

def parse_eta_in_hours(left):
    pattern = '(?:([0-9]*) day[s]?, )?([0-9]*):([0-9]*):([0-9]*)'
    result = re.match(pattern, left)
    if result:
        gs = result.groups()
        gs = [float(g) if g else 0 for g in gs]
        assert int(gs[0]) == gs[0]
        days = int(gs[0])
        hours = gs[1] + gs[2] / 60. + gs[3] / 3600
        return days, hours
    return -1, -1

def attach_philly_maskrcnn_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        # Philly, maskrcnn-benchmark log
        pattern = '(.*): .*: eta: (.*) iter: [0-9]*  speed: ([0-9\.]*).*'
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left, speed = result.groups()
            job_info['speed'] = speed
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            delay = (now - log_time).total_seconds()
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h({:.1f}s)'.format(d, h, delay)
            return True
    return False

def attach_aml_maskrcnn_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        pattern = r'(.*),.* trainer\.py.*do_train\(\): eta: (.*) iter: [0-9]*  speed: ([0-9\.]*).*'
        result = re.match(pattern, log)
        if result and result.groups():
            log_time, left, speed = result.groups()
            job_info['speed'] = speed
            from dateutil.parser import parse
            log_time = parse(log_time)
            job_info['log_time'] = log_time
            # log_time here is UTC. convert it to local time
            d, h = parse_eta_in_hours(left)
            job_info['left'] = '{}-{:.1f}h'.format(d, h)
            return True
    return False

def attach_philly_caffe_log_if_is(all_log, job_info):
    for log in reversed(all_log):
        # philly, caffe log
        pattern = '.*solver\.cpp:[0-9]*] Iteration [0-9]* \(.* iter\/s, ([0-9\.]*s\/100) iters, left: ([0-9\.]*h)\), loss = [0-9\.]*'
        result = re.match(pattern, log)
        if result and result.groups():
            job_info['speed'], job_info['left'] = result.groups()
            return True
    return False

def attach_gpu_utility_from_log(all_log, job_info):
    for log in reversed(all_log):
        # philly, caffe log aml_server or philly_server log
        pattern = '.*_server.py:.*monitor.*\[(.*)\]'
        result = re.match(pattern, log)
        if result and result.groups():
            all_info = json.loads('[{}]'.format(result.groups()[0].replace('\'', '\"')))
            min_gpu_mem = min([i['mem_used'] for i in all_info])
            max_gpu_mem = max([i['mem_used'] for i in all_info])
            min_gpu_util = min([i['gpu_util'] for i in all_info])
            max_gpu_util = max([i['gpu_util'] for i in all_info])
            # GB
            job_info['mem_used'] = '{}-{}'.format(round(min_gpu_mem/1024, 1),
                    round(max_gpu_mem/1024., 1))
            job_info['gpu_util'] = '{}-{}'.format(min_gpu_util, max_gpu_util)
            return True
    return False

def attach_log_parsing_result(job_info):
    # run unit test if modified
    logs = job_info['latest_log']
    all_log = logs.split('\n')
    del job_info['latest_log']
    attach_gpu_utility_from_log(all_log, job_info)
    if attach_philly_maskrcnn_log_if_is(all_log, job_info):
        return
    if attach_aml_maskrcnn_log_if_is(all_log, job_info):
        return
    if attach_philly_caffe_log_if_is(all_log, job_info):
        return

def print_offensive_folder(folder):
    all_folder = os.listdir(folder)
    name_to_size = {}
    for i, f in enumerate(tqdm(all_folder)):
        sec = 60 * 10
        f = op.join(folder, f)
        size = run_if_not_cached(get_folder_size, f, sec)
        name_to_size[f] = size
        logging.info('{}: {}'.format(f, size))
    logging.info(', '.join([op.basename(n) for n, s in name_to_size.items() if
        s < 0]))

def get_folder_size(f, sec):
    cmd = ['du', '--max-depth=0', f]
    import subprocess
    from subprocess import check_output
    try:
        out = check_output(cmd, timeout=sec)
    except subprocess.TimeoutExpired:
        logging.info('{}'.format(f))
        return -1
    out = out.decode()
    size = [x.strip() for x in out.split('\t')][0]
    return int(size)

class make_namespace_by_dict(object):
    def __init__(self, d):
        for k in d:
            v = d[k]
            if type(v) is dict:
                self.__dict__[k] = make_namespace_by_dict(v)
            else:
                self.__dict__[k] = v
    def clone(self):
        c = copy.deepcopy(self.__dict__)
        return make_namespace_by_dict(c)

@try_once
def try_get_cpu_info():
    command = 'cat /proc/cpuinfo'
    return os.popen(command).read().strip()

# ---------------------------------------------------- pytorch speed analysis
def create_speed_node(info):
    node = Tree()
    node.add_features(**info)
    return node

def speed_tree_insert(root, node):
    while True:
        need_merge_nodes = [c for c in root.children
                if is_child_parent(c.name, node.name)]
        if len(need_merge_nodes) > 0:
            for x in need_merge_nodes:
                x.detach()
            for x in need_merge_nodes:
                node.add_child(x)
            root.add_child(node)
            return
        go_deeper_nodes = [c for c in root.children if
                is_child_parent(node.name, c.name)]
        if len(go_deeper_nodes) == 0:
            root.add_child(node)
            return
        else:
            assert len(go_deeper_nodes) == 1
            root = go_deeper_nodes[0]

def is_child_parent(c, p):
    if p == '':
        return True
    return c.startswith(p + '.')

def speed_trees_insert(roots, info):
    node = create_speed_node(info)
    # we assume the name are not equal
    need_merge_nodes = [r for r in roots
            if is_child_parent(r.name, info['name'])]
    if len(need_merge_nodes) > 0:
        for x in need_merge_nodes:
            node.add_child(x)
            roots.remove(x)
        roots.append(node)
        return
    need_insert_roots = [r for r in roots
            if is_child_parent(info['name'], r.name)]
    if len(need_insert_roots) == 0:
        roots.append(node)
    elif len(need_insert_roots) == 1:
        speed_tree_insert(need_insert_roots[0], node)
    else:
        raise Exception()

def build_speed_tree(component_speeds):
    roots = []
    for c in component_speeds:
        speed_trees_insert(roots, c)
    return roots

def get_vis_str(component_speeds):
    roots = build_speed_tree(component_speeds)
    assert len(roots) == 1
    root = roots[0]
    for n in root.iter_search_nodes():
        n.global_avg_in_ms = round(1000. * n.global_avg, 1)
    for n in root.iter_search_nodes():
        s = sum([c.global_avg for c in n.children])
        n.unique_in_ms = round(1000. * (n.global_avg - s), 1)
    return root.get_ascii(attributes=
        ['name', 'global_avg_in_ms', 'unique_in_ms'])

def create_vis_net_file(speed_yaml, vis_txt):
    info = load_from_yaml_file(speed_yaml)
    if type(info) is list:
        info = info[0]
    assert type(info) is dict
    component_speeds = info['meters']
    write_to_file(get_vis_str(component_speeds), vis_txt)

# ---------------------------------------------------------------------

def dict_add(d, k, v):
    if k not in d:
        d[k] = v
    else:
        d[k] += v

def calc_mean(x):
    return sum(x) / len(x)

def compare_gmap_evals(all_eval_file,
        label_to_testcount=None,
        output_prefix='out'):
    result = ['\n']
    all_result = [load_from_yaml_file(f) for f in all_eval_file]
    all_cat2map = [result['class_ap'] for result in all_result]

    cats = list(all_cat2map[0].keys())
    gains = [all_cat2map[1][c] - all_cat2map[0][c] for c in cats]

    all_info = [{'name': c, 'acc_gain': g} for c, g in zip(cats, gains)]
    all_info = sorted(all_info, key=lambda x: x['acc_gain'])

    all_map = [sum(cat2map.values()) / len(cat2map) for cat2map in all_cat2map]
    result.append('all map = {}'.format(', '.join(
        map(lambda x: str(round(x, 3)), all_map))))

    non_zero_cats = [cat for cat, ap in all_cat2map[0].items()
            if all_cat2map[1][cat] > 0 and  ap > 0]
    logging.info('#non zero cat = {}'.format(len(non_zero_cats)))
    for cat2map in all_cat2map:
        logging.info('non zero cat mAP = {}'.format(
            calc_mean([cat2map[c] for c in non_zero_cats])))

    if label_to_testcount is not None:
        all_valid_map = [calc_mean([ap for cat, ap in cat2map.items() if
            label_to_testcount.get(cat, 0) >
                50]) for cat2map in all_cat2map]
        result.append('all valid map = {}'.format(', '.join(
            map(lambda x: str(round(x, 3)), all_valid_map))))
        valid_cats = set([l for l, c in label_to_testcount.items() if c > 50])

    max_aps = [max([cat2map[c] for cat2map in all_cat2map]) for c in cats]
    max_map = sum(max_aps) / len(max_aps)
    result.append('max map = {:.3f}'.format(max_map))

    for info in all_info:
        for k in info:
            if type(info[k]) is float:
                info[k] = round(info[k], 2)
    result.extend(get_table_print_lines(all_info[:5] + all_info[-6:], ['name',
        'acc_gain',
        ]))
    if label_to_testcount is not None:
        result.append('valid cats only:')
        all_valid_info = [i for i in all_info if i['name'] in valid_cats]
        result.extend(get_table_print_lines(all_valid_info[:5] + all_valid_info[-6:],
            ['name', 'acc_gain',
            ]))

    all_acc_gain = [info['acc_gain'] for info in all_info]
    logging.info('\n'.join(result))

    plot_to_file(list(range(len(all_acc_gain))),
            all_acc_gain,
            output_prefix + '.png')

def merge_class_names_by_location_id(anno):
    if any('location_id' in a for a in anno):
        assert all('location_id' in a for a in anno)
        location_id_rect = [(a['location_id'], a) for a in anno]
        from qd.qd_common import list_to_dict
        location_id_to_rects = list_to_dict(location_id_rect, 0)
        merged_anno = []
        for _, rects in location_id_to_rects.items():
            r = copy.deepcopy(rects[0])
            r['class'] = [r['class']]
            r['class'].extend((rects[i]['class'] for i in range(1,
                len(rects))))
            r['conf'] = [r['conf']]
            r['conf'].extend((rects[i].get('conf', 1) for i in range(1,
                len(rects))))
            merged_anno.append(r)
        return merged_anno
    else:
        assert all('location_id' not in a for a in anno)
        for a in anno:
            a['class'] = [a['class']]
            a['conf'] = [a['conf']]
        return anno

def softnms_c(rects, **kwargs):
    from fast_rcnn.nms_wrapper import soft_nms
    nms_input = np.zeros((len(rects), 5), dtype=np.float32)
    for i, r in enumerate(rects):
        nms_input[i, 0:4] = r['rect']
        nms_input[i, -1] = r['conf']
    nms_out = soft_nms(nms_input, **kwargs)
    return [{'rect': x[:4], 'conf': x[-1]} for x in nms_out]

def softnms(rects, th=0.5):
    rects = copy.deepcopy(rects)
    result = []
    while len(rects) > 0:
        max_idx = max(range(len(rects)), key=lambda i:
                rects[i]['conf'])
        max_det = rects[max_idx]
        result.append(max_det)
        rects.remove(max_det)
        for j in range(len(rects)):
            j_rect = rects[j]
            ij_iou = calculate_iou1(max_det['rect'], j_rect['rect'])
            rects[j]['conf'] *= math.exp(-ij_iou * ij_iou / th)
    return result

def acquireLock(lock_f='/tmp/lockfile.LOCK'):
    ''' acquire exclusive lock file access '''
    import fcntl
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor

def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()

if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

