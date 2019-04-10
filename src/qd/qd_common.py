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
try:
    from itertools import izip as zip
except ImportError:
    # in python3, we don't need itertools.izip since zip is izip
    pass
import time
import matplotlib.pyplot as plt
from pprint import pprint
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
try:
    # py3
    from urllib.request import urlopen
    from urllib.request import HTTPError
except ImportError:
    # py2
    from urllib2 import urlopen
    from urllib2 import HTTPError


def case_incensitive_overlap(all_terms):
    all_lower_to_term = [{t.lower(): t for t in terms} for terms in all_terms]
    all_lowers = [set(l.keys()) for l in all_lower_to_term]
    anchor = all_lowers[0].intersection(*all_lowers[1:])

    return [[lower_to_term[l] for l in anchor]
        for lower_to_term in all_lower_to_term]

def zip_qd(out_zip):
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
            '\*aux_data/yolo9k/\*',
            '-x',
            '\*visualization\*',
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
            '\*src/CCSCaffe/docs/tutorial/\*',
            '-x',
            '\*src/CCSCaffe/matlab/\*',
            '-x',
            '\*.git\*']
    cmd_run(cmd, working_dir=os.getcwd(), shell=True)

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
            result[p[0]] = p[1][i]
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
    if op.isdir(d):
        shutil.rmtree(d)

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
        from pprint import pformat
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
        shell=False):
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
    return json.dumps(obj, sort_keys=True)

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
    parser = argparse.ArgumentParser(description='Train a Yolo network')
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
    report_file = predict_file
    fname = op.join('output', full_expid, 'snapshot', report_file +
            '.class_ap.json')
    if op.isfile(fname):
        return json.loads(read_to_buffer(fname))
    else:
        return None

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

def parse_test_data(predict_file):
    # e.g. 'model_iter_368408.caffemodel.Tax1300V14.1_OpenImageV4_448Test_with_bb.train.maintainRatio.OutTreePath.TreeThreshold0.1.ClsIndependentNMS.predict'
    pattern = 'model(?:_iter)?_[0-9]*[e]?\.(?:caffemodel|pth\.tar|pth)\.(.*)\.(train|trainval|test)\..*\.predict'
    match_result = re.match(pattern, predict_file)
    if match_result and len(match_result.groups()) == 2:
        return match_result.groups()
    # the following will be deprecated gradually
    parts = predict_file.split('.')
    idx_caffemodel = [i for i, p in enumerate(parts) if 'caffemodel' in p]
    if len(idx_caffemodel) == 1:
        idx_caffemodel = idx_caffemodel[0]
        test_data = parts[idx_caffemodel + 1]
        test_data_split = parts[idx_caffemodel + 2]
        if test_data_split in ['train', 'trainval', 'test']:
            return test_data, test_data_split
    all_data = os.listdir('data/')
    candidates = [data for data in all_data if '.caffemodel.' + data in predict_file]
    assert len(candidates) > 0
    max_length = max([len(c) for c in candidates])
    test_data = [c for c in candidates if len(c) == max_length][0]
    test_data_split = 'test' if 'testOnTrain' not in predict_file else 'train'
    return test_data, test_data_split

def parse_data(full_expid):
    all_data = os.listdir('data/')
    candidates = [data for data in all_data if full_expid.startswith(data)]
    max_length = max([len(c) for c in candidates])
    return [c for c in candidates if len(c) == max_length][0]

def parse_iteration(file_name):
    r = re.match('.*model_iter_([0-9]*)\..*', file_name)
    if r is None:
        r = re.match('.*model_iter_([0-9]*)e\..*', file_name)
        if r is None:
            return -1
    return int(float(r.groups()[0]))

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
    found = glob.glob(op.join(model_folder, '*.report'))
    predict_files.extend([op.basename(f) for f in found])

    found = glob.glob(op.join(model_folder, '*.report.v[0-9]'))
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
    with open(filein,'r') as tsvin, open(idxout,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0;
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n");
            tsvin.readline()
            fpos = tsvin.tell();

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
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except OSError:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise

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
    with open(file_name, 'r') as fp:
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
    if all(isinstance(x, str) or isinstance(x, unicode) for x in xs):
        xs2 = range(len(xs))
        #plt.xticks(xs2, xs, rotation=15, ha='right')
        plt.xticks(xs2, xs, rotation='vertical')
        xs = xs2
    if type(ys) is dict:
        for key in ys:
            plt.plot(xs, ys[key], '-o')
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
    with open(file_name, 'w') as fp:
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

    return [new_x, new_y, new_x + new_w, new_y + new_h]

def is_valid_rect(rect):
    return len(rect) == 4 and rect[0] < rect[2] and rect[1] < rect[3]


if __name__ == '__main__':
    init_logging()
    kwargs = parse_general_args()
    from pprint import pformat
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

