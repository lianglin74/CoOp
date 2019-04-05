# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import os.path as op
import glob

def get_qd_root():
    return op.dirname(op.dirname(op.dirname(op.realpath(__file__))))


import os.path as op
sys.path.append(op.join(get_qd_root(), 'scripts'))
from tsv_io import tsv_reader
from qd_common import write_to_file
from process_tsv import TSVFile
from process_tsv import ImageTypeParser
from process_tsv import visualize_box
from process_image import draw_bb, show_image, save_image
import base64
from yolotrain import CaffeWrapper
from tsv_io import load_labels
from process_tsv import update_confusion_matrix
from pprint import pformat
import os
from yolotrain import get_confusion_matrix
from qd_common import readable_confusion_entry
from process_tsv import gt_predict_images
from qd_common import get_target_images
from tsv_io import get_all_data_info
from tsv_io import get_all_data_info2
from qd_common import get_all_model_expid
from qd_common import get_parameters_by_full_expid
from process_tsv import get_confusion_matrix_by_predict_file
from qd_common import get_all_predict_files
from qd_common import parse_data, parse_test_data
from tsv_io import load_labelmap
from process_tsv import gen_html_tree_view
from qd_common import get_all_tree_data
from qd_common import init_logging
from django.http import HttpResponseRedirect
from django.urls import reverse
import cv2
from django import template
import simplejson as json
from process_tsv import build_taxonomy_impl
import sys
import traceback2 as traceback
from .models import *
import django.core.files
import logging
import uuid
from qd_common import load_class_ap, worth_create
from process_tsv import visualize_predict
from process_tsv import visualize_predict_no_draw
from process_tsv import get_class_count
from process_tsv import visualize_box_no_draw
from process_tsv import populate_dataset_details
from qd_common import write_to_yaml_file, load_from_yaml_file
from build_compare import build_side_by_side_compare
import copy

init_logging()

def hash_sha1(s):
    import hashlib
    if type(s) is not str:
        from pprint import pformat
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def run_in_qd(func, *args, **kwargs):
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    r = func(*args, **kwargs)
    os.chdir(curr_dir)
    return r


def view_tree(request):
    if 'data' in request.GET:
        data = request.GET.get('data')
        # provide the accuracy for each category
        full_expid = request.GET.get('full_expid')
        predict_file = request.GET.get('predict_file')
        s = run_in_qd(gen_html_tree_view, data, full_expid,
                      predict_file)
        context = {'li_ul_tree_str': s}
        return render(request, 'detection/view_tree.html', context)
    else:
        data_list = run_in_qd(get_all_tree_data)
        context = {'data_names': data_list}
        return render(request, 'detection/view_list_has_tree.html', context)


def view_tree_test(request):
    if 'data' in request.GET:
        data = request.GET.get('data')
        # provide the accuracy for each category
        full_expid = request.GET.get('full_expid')
        predict_file = request.GET.get('predict_file')
        s = run_in_qd(gen_html_tree_view, data, full_expid,
                      predict_file)
        context = {'li_ul_tree_str': s}
        return render(request, 'detection/view_tree.html', context)
    else:
        data_list = run_in_qd(get_all_tree_data)
        context = {'data_names': data_list}
        return render(request, 'detection/view_list_has_tree.html', context)


def view_exp_list(request):
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    full_expids = get_all_model_expid()
    os.chdir(curr_dir)
    full_expids.sort()
    context = {'full_expids': full_expids}
    return render(request, 'detection/exp_list.html', context)


class VisualizationDatabaseByFileSystem():
    def query_predict_precision(self, full_expid, pred_file, class_name, threshold, start_id, max_item):
        pairs = visualize_predict_no_draw(full_expid, pred_file, class_name,
                                          start_id, threshold)
        result = []
        for i, (key, im_origin, rects_gt, rects_pred, ap) in enumerate(pairs):
            path_origin = save_image_in_static(im_origin, '{}/{}_origin.png'.format(
                pred_file, key))
            c = {'url': op.join('/static/', path_origin),
                 'gt': rects_gt, 'pred': rects_pred, 'key': key}
            result.append(c)
        return result


def CreateVisualizationDatabase():
    try:
        # return VisualizationDatabaseByFileSystem()
        from process_tsv import VisualizationDatabaseByMongoDB
        MongoClient().admin.command('ismaster')
        logging.info('creating mongodb database')
        return VisualizationDatabaseByMongoDB()
    except:
        logging.info('no mongodb is availalbe. Reverse back to file system')
        return VisualizationDatabaseByFileSystem()


_db = CreateVisualizationDatabase()


def submit_pipeline(request):
    if request.method == 'GET':
        return render(request, 'detection/submit_pipeline.html')
    else:
        pipeline = request.POST['pipeline']
        task_id = _db.insert('pipeline', pipeline)
        return HttpResponseRedirect('/detection/process_pipeline/')


def view_model_by_predict_file4(request, full_expid, predict_file, label, start_id, threshold):
    '''
    use js to render the box compared with version 2
    '''
    # start_time = time.time()
    start_id = int(float(start_id))
    threshold = float(threshold)

    # qd code
    curr_dir_backup = os.getcwd()
    os.chdir(get_qd_root())

    max_pairs = 50
    if request.GET.get('acc_type', 'precision') == 'precision':
        pairs = _db.query_predict_precision(full_expid,
                                            predict_file, label, threshold, start_id, max_pairs)
    else:
        pairs = _db.query_predict_recall(full_expid,
                                         predict_file, label, threshold, start_id, max_pairs)
    # pairs = visualize_predict_no_draw(full_expid, predict_file, label, start_id, threshold)
    all_type_to_rects = []
    all_url = []
    all_key = []
    for i, info in enumerate(pairs):
        key, image_url, rects_gt, rects_pred = info['key'], info['url'], \
            info['gt'], info['pred']
        all_url.append(image_url)
        all_type_to_rects.append({'gt': rects_gt,
                                  'pred': rects_pred})
        all_key.append(key)
        if i >= max_pairs - 1:
            break

    common_param = [('full_expid', full_expid),
                    ('predict_file', predict_file),
                    ('filter_label', label),
        ('threshold', threshold)]
    previous_param = []
    previous_param.extend(common_param)
    previous_start_id = start_id - max_pairs
    previous_param.append(('start_id', previous_start_id))
    next_param = []
    next_param.extend(common_param)
    next_param.append(('start_id', str(start_id + len(all_url))))
    previous_button_param = '&'.join(['{}={}'.format(key, value) for key, value
                                      in previous_param])
    previous_link = reverse('detection:view_model') + \
        '?' + previous_button_param
    next_button_param = '&'.join(['{}={}'.format(key, value) for key, value in
                                  next_param])
    next_link = reverse('detection:view_model') + '?' + next_button_param

    context = {'all_type_to_rects': json.dumps(all_type_to_rects),
               'target_label': label,
               'all_key': json.dumps(all_key),
               'all_url': json.dumps(all_url),
               'previous_link': previous_link,
               'next_link': next_link}
    # logging.info('time cost: {}'.format(time.time() - start_time))
    return render(request, 'detection/images_js2.html', context)


def view_model_by_predict_file3(request, full_expid, predict_file,
                                label, start_id, threshold):
    '''
    use js to render the box compared with version 2
    '''
    start_id = int(float(start_id))
    threshold = float(threshold)

    # qd code
    curr_dir_backup = os.getcwd()
    os.chdir(get_qd_root())

    pairs = visualize_predict_no_draw(
        full_expid, predict_file, label, start_id, threshold)
    max_pairs = 50
    all_type_to_rects = []
    all_url = []
    all_key = []
    for i, (key, im_origin, rects_gt, rects_pred, ap) in enumerate(pairs):
        path_origin = save_image_in_static(im_origin, '{}/{}_origin.png'.format(
            predict_file, key))
        all_url.append(op.join('/static/', path_origin))
        all_type_to_rects.append({'gt': rects_gt,
                                  'pred': rects_pred})
        all_key.append(key)
        if i >= max_pairs - 1:
            break

    common_param = [('full_expid', full_expid),
                    ('predict_file', predict_file),
                    ('filter_label', label),
        ('threshold', threshold)]
    previous_param = []
    previous_param.extend(common_param)
    previous_start_id = start_id - max_pairs
    previous_param.append(('start_id', previous_start_id))
    next_param = []
    next_param.extend(common_param)
    next_param.append(('start_id', str(start_id + len(all_url))))
    previous_button_param = '&'.join(['{}={}'.format(key, value) for key, value
                                      in previous_param])
    previous_link = reverse('detection:view_model') + \
        '?' + previous_button_param
    next_button_param = '&'.join(['{}={}'.format(key, value) for key, value in
                                  next_param])
    next_link = reverse('detection:view_model') + '?' + next_button_param

    context = {'all_type_to_rects': json.dumps(all_type_to_rects),
               'target_label': label,
               'all_key': json.dumps(all_key),
               'all_url': json.dumps(all_url),
               'previous_link': previous_link,
               'next_link': next_link}
    return render(request, 'detection/images_js2.html', context)


def view_model_by_predict_file(request, full_expid, predict_file,
                               label, start_id, threshold):
    start_id = int(float(start_id))
    threshold = float(threshold)

    # qd code
    curr_dir_backup = os.getcwd()
    os.chdir(get_qd_root())
    data = parse_data(full_expid)
    test_data, test_data_split = parse_test_data(predict_file)
    x = get_confusion_matrix_by_predict_file(full_expid, predict_file,
                                             threshold)
    predicts, gts, label_to_idx = x['predicts'], x['gts'], x['label_to_idx']
    confusion_pred_gt = x['confusion_pred_gt']
    confusion_gt_pred = x['confusion_gt_pred']
    os.chdir(curr_dir_backup)

    image_pairs = []
    target_images, image_aps = get_target_images(
        predicts, gts, label, threshold)
    pairs = gt_predict_images(predicts, gts, test_data, target_images,
                              label,
                              start_id,
                              threshold, label_to_idx, image_aps, test_data_split)
    max_pairs = 10
    for i, (key, im_origin, im_gt_target, im_pred_target, im_gt, im_pred, ap) in enumerate(pairs):
        path_origin = save_image_in_static(im_origin, '{}/{}_origin.png'.format(
            data, key))
        path_gt_target = save_image_in_static(im_gt_target, '{}/{}_gt_target.png'.format(
            data, key))
        path_gt = save_image_in_static(im_gt, '{}/{}_gt.png'.format(
            data, key))
        path_pred = save_image_in_static(im_pred, '{}/{}/{}_pred.png'.format(
            full_expid, predict_file, key))
        path_pred_target = save_image_in_static(im_pred_target,
                                                '{}/{}/{}_pred_target.png'.format(full_expid, predict_file, key))
        image_pairs.append((path_origin, path_gt_target,
                            path_pred_target, path_gt, path_pred, ap))
        if i >= max_pairs - 1:
            break

    common_param = [('full_expid', full_expid),
                    ('predict_file', predict_file),
                    ('filter_label', label),
        ('threshold', threshold)]
    previous_param = []
    previous_param.extend(common_param)
    previous_start_id = start_id - max_pairs
    if previous_start_id < 0:
        total_images = len(target_images)
        previous_start_id = previous_start_id + total_images
    previous_param.append(('start_id', str(max(0, previous_start_id))))
    next_param = []
    next_param.extend(common_param)
    next_param.append(('start_id', str(start_id + len(image_pairs))))
    previous_button_param = '&'.join(['{}={}'.format(key, value) for key, value
                                      in previous_param])
    next_button_param = '&'.join(['{}={}'.format(key, value) for key, value in
                                  next_param])

    if label not in confusion_pred_gt:
        html_confusion_pred_gt = []
    else:
        html_confusion_pred_gt = readable_confusion_entry(
            confusion_pred_gt[label])
    if label not in confusion_gt_pred:
        html_confusion_gt_pred = []
    else:
        html_confusion_gt_pred = readable_confusion_entry(
            confusion_gt_pred[label])
    context = {
        'previous_button_param': previous_button_param,
        'next_button_param': next_button_param,
            'label': label,
            'confusion_pred_gt': html_confusion_pred_gt,
            'confusion_gt_pred': html_confusion_gt_pred,
            'image_pairs': image_pairs}

    return render(request, 'detection/predict_result.html', context)


def view_model_prediction_labelmap(request, full_expid, predict_file):
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    data = parse_data(full_expid)
    labelmap = load_labelmap(data)
    class_ap = run_in_qd(load_class_ap, full_expid, predict_file)
    labelmap_ap = None
    if class_ap:
        if '0.3' in class_ap['overall']:
            class_ap = class_ap['overall']['0.3']['class_ap']
        elif '0' in class_ap['overall']:
            class_ap = class_ap['overall']['0']['class_ap']
        elif '-1' in class_ap['overall']:
            class_ap = class_ap['overall']['-1']['class_ap']
        else:
            assert False
        labelmap_ap = [
            [l, '{:.2f}'.format(class_ap.get(l, -1))] for l in labelmap]
    else:
        labelmap_ap = [[l, None] for l in labelmap]
    if 'data' in request.GET:
        class_count = get_class_count(request.GET['data'], ['train', 'test'])
    else:
        class_count = {'train': {}, 'test': {}}
    for l in labelmap_ap:
        l.append(class_count['train'].get(l[0]))
        l.append(class_count['test'].get(l[0]))
    os.chdir(curr_dir)
    labelmap_ap = sorted(labelmap_ap, key=lambda x: x[1])
    for i, l in enumerate(labelmap_ap):
        l.insert(0, i)
    context = {'prediction_file': predict_file,
               'labelmap_ap': labelmap_ap,
               'data': request.GET.get('data', None),
               'class_count': class_count,
               'full_expid': full_expid}
    return render(request, 'detection/view_model_prediction_labelmap.html',
                  context)



def view_test_model(request, full_expid, predict_file):
    context = {'full_expid': full_expid,
               'predict_file': predict_file}
    return render(request, 'detection/test_model.html',
                  context)


def test_model(request):
    assert request.method == 'POST'
    coded = request.FILES['image_file'].read()
    import cv2
    import numpy as np
    nparr = np.frombuffer(coded, np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if max(im.shape[:2]) > 600:
        im_scale = float(600) / float(max(im.shape[:2]))
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
    from yolotrain import predict_one_view
    predict_file = request.POST['predict_file']
    all_bb, all_label, all_conf = run_in_qd(predict_one_view, im,
                                            request.POST['full_expid'],
                                            predict_file
                                            )
    infos = []
    info = {}
    info['pred'] = [{'rect': bb, 'class': l, 'conf': conf}
                    for bb, l, conf in zip(all_bb, all_label, all_conf)]
    infos.append(info)
    import random
    html_path = save_image_in_static(
        im, 'predict_request/{}.png'.format(int(1000 * random.random())))
    context = {'all_type_to_rects': json.dumps(infos),
               'all_url': json.dumps(['/static/' + html_path])}
    return render(request, 'detection/images_js3.html', context)


def view_model(request):
    if 'full_expid' in request.GET and \
            'predict_file' in request.GET and \
            'filter_label' in request.GET and \
            'start_id' in request.GET and \
            'threshold' in request.GET:
        return view_model_by_predict_file4(request,
                                           request.GET['full_expid'],
                                           request.GET['predict_file'],
            request.GET['filter_label'],
            request.GET['start_id'],
            request.GET['threshold'])
        # return view_model_by_predict_file3(request,
              # request.GET['full_expid'],
              # request.GET['predict_file'],
              # request.GET['filter_label'],
              # request.GET['start_id'],
              # request.GET['threshold'])
        # return view_model_by_predict_file(request,
              # request.GET['full_expid'],
              # request.GET['predict_file'],
              # request.GET['filter_label'],
              # request.GET['start_id'],
              # request.GET['threshold'])
    elif 'full_expid' in request.GET and \
            'predict_file' in request.GET and \
            'test_model' in request.GET:
        return view_test_model(request,
                               request.GET['full_expid'], request.GET['predict_file'])
    elif 'full_expid' in request.GET and 'predict_file' in request.GET:
        return view_model_prediction_labelmap(request,
                                              request.GET['full_expid'], request.GET['predict_file'])
    elif 'full_expid' in request.GET:
        full_expid = request.GET['full_expid']
        predict_files = run_in_qd(get_all_predict_files, full_expid)
        context = {'prediction_files': predict_files,
                   'full_expid': full_expid}
        return render(request, 'detection/view_model_prediction.html',
                      context)
    else:
        return view_exp_list(request)


def edit_model_label(request):
    if request.method == 'POST':
        all_valid_label = request.POST.getlist('valid')
        write_to_file('\n'.join(all_valid_label),
                      '/tmp/valid2.csv')
        return HttpResponseRedirect('/detection/confirm/')


def confirm(request):
    return HttpResponse('confirm')


def save_image_in_static(im, rel_path):
    html_path = 'detection/{}'.format(rel_path)
    disk_path = op.join(op.dirname(__file__), 'static', html_path)
    # if im.shape[0] > 400:
      # factor = 400. / im.shape[0]
      # im = cv2.resize(im, (int(im.shape[1] * factor), 400))
    save_image(im, disk_path)
    return html_path


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def view_image_compare_test(request, data, split, version, label, start_id, min_conf=None):
    '''
    use js to render the box in the client side
    '''
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    start_id = int(float(start_id))
    images = visualize_box_no_draw(data, split, version, label, start_id)
    all_type_to_rects = []
    all_url = []
    max_image_shown = 40
    all_key = []
    for i, (fname, origin, gt) in enumerate(images):
        if i >= max_image_shown:
            break

        origin_html_path = save_image_in_static(
            origin, '{}/{}/{}/{}.jpg'.format(data, split, version, fname))

        all_key.append(fname)
        all_url.append('/static/' + origin_html_path)
        all_type_to_rects.append({'gt': gt})

    os.chdir(curr_dir)

    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(max(0, start_id - max_image_shown))
    previous_link = reverse('detection:view_compare_test')
    previous_link = previous_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])
    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(start_id + len(all_type_to_rects))
    next_link = reverse('detection:view_compare_test')
    next_link = next_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])

    context = {'all_type_to_rects': json.dumps(all_type_to_rects),
               'target_label': label,
               'all_url': json.dumps(all_url),
               'all_key': json.dumps(all_key),
               'previous_link': previous_link,
               'next_link': next_link}

    return render(request, 'detection/images_compare.html', context)

def view_image_compare_test(request, data, split, version, label, start_id, min_conf=None):
    '''
    use js to render the box in the client side
    '''
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    start_id = int(float(start_id))
    images = visualize_box_no_draw(data, split, version, label, start_id)
    all_type_to_rects = []
    all_url = []
    max_image_shown = 40
    all_key = []

    blkLstFilename = './data/{}/blacklist.txt'.format(data)
    black_list = []
    if os.path.isfile(blkLstFilename):
        with open(blkLstFilename, 'r') as f:
            lines = f.readlines()
            black_list = [line.strip('\n') for line in lines]
            black_list = [blk.lower() for blk in black_list]

    # black_list.append('person')

    thresholdFilename = './data/{}/threshold.tsv'.format(data)
    dict_threshold = {}
    if os.path.isfile(thresholdFilename):
        with open(thresholdFilename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]

            for line in lines:
                cols = [x.strip() for x in line.split('\t')]
                dict_threshold[cols[0]] = float(cols[1])

    black_list_set = set(black_list)

    for i, (fname, origin, gt) in enumerate(images):
        if i >= max_image_shown:
            break

        origin_html_path = save_image_in_static(
            origin, '{}/{}/{}/origin_{}.jpg'.format(data, split, version, fname))

        all_key.append(fname)
        all_url.append('/static/' + origin_html_path)

        gt1 = []
        for item in gt:
            if item['class'].lower() not in black_list_set:
                if i % 2 == 1:
                    gt1.append(item)
                else:
                    if min_conf == None:
                        if item['conf'] >= 0.5:
                            gt1.append(item)
                    else:
                        if item['conf'] >= min_conf:
                            gt1.append(item)

        all_type_to_rects.append({'gt': gt1})

    os.chdir(curr_dir)

    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(max(0, start_id - max_image_shown))
    previous_link = reverse('detection:view_compare')
    previous_link = previous_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])
    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(start_id + len(all_type_to_rects))
    next_link = reverse('detection:view_compare')
    next_link = next_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])

    context = {'all_type_to_rects': json.dumps(all_type_to_rects),
               'target_label': label,
               'all_url': json.dumps(all_url),
               'all_key': json.dumps(all_key),
               'previous_link': previous_link,
               'next_link': next_link,
               'black_list': json.dumps(black_list)}

    return render(request, 'detection/images_compare_test.html', context)

def read_threshold(thresholdFilename):
    dict_threshold  = {}
    if os.path.isfile(thresholdFilename):
        with open(thresholdFilename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]

            for line in lines:
                cols = [x.strip() for x in line.split('\t')]
                dict_threshold[cols[0]] = float(cols[1])
    return dict_threshold

def view_image_compare(request, data, split, version, label, start_id, min_conf_left, min_conf_right):
    '''
    use js to render the box in the client side
    '''
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    start_id = int(float(start_id))
    images = visualize_box_no_draw(data, split, version, label, start_id)
    all_type_to_rects = []
    all_url = []
    max_image_shown = 40
    all_key = []

    blkLstFilename = './data/{}/blacklist.txt'.format(data)
    black_list = []
    if os.path.isfile(blkLstFilename):
        with open(blkLstFilename, 'r') as f:
            lines = f.readlines()
            black_list = [line.strip('\n') for line in lines]
            black_list = [blk.lower() for blk in black_list]

    # black_list.append('person')

    # print black_list

    thresholdFilename = './data/{}/threshold_left.tsv'.format(data)
    dict_threshold_left = read_threshold(thresholdFilename)

    thresholdFilename = './data/{}/threshold_right.tsv'.format(data)
    dict_threshold_right = read_threshold(thresholdFilename)

    black_list_set = set(black_list)

    for i, (fname, origin, gt) in enumerate(images):
        if i >= max_image_shown:
            break

        origin_html_path = save_image_in_static(
            origin, '{}/{}/{}/{}.jpg'.format(data, split, version, fname))

        all_key.append(fname)
        all_url.append('/static/' + origin_html_path)

        gt1 = []
        for item in gt:
            if item['class'].lower() not in black_list_set:
                if i % 2 == 1:
                    if item['class'] in dict_threshold_left:
                        if item['conf']>=dict_threshold_left[item['class']]:
                            if min_conf_right == None:
                                if item['conf'] >= 0.5:
                                    gt1.append(item)
                            else:
                                if item['conf'] >= min_conf_right:
                                    gt1.append(item)
                    else:
                        if min_conf_right == None:
                            if item['conf'] >= 0.5:
                                gt1.append(item)
                        else:
                            if item['conf'] >= min_conf_right:
                                gt1.append(item)
                else:
                    if item['class'] in dict_threshold_right:
                        if item['conf']>=dict_threshold_right[item['class']]:
                            if min_conf_left == None:
                                if item['conf'] >= 0.5:
                                    gt1.append(item)
                            else:
                                if item['conf'] >= min_conf_left:
                                    gt1.append(item)
                    else:
                        if min_conf_left == None:
                            if item['conf'] >= 0.5:
                                gt1.append(item)
                        else:
                            if item['conf'] >= min_conf_left:
                                gt1.append(item)

       
        all_type_to_rects.append({'gt': gt1})

    os.chdir(curr_dir)

    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(max(0, start_id - max_image_shown))
    previous_link = reverse('detection:view_compare')
    previous_link = previous_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])
    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(start_id + len(all_type_to_rects))
    next_link = reverse('detection:view_compare')
    next_link = next_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])

    # all_type_to_rects = []
    context = {'all_type_to_rects': json.dumps(all_type_to_rects),
               'target_label': label,
               'all_url': json.dumps(all_url),
               'all_key': json.dumps(all_key),
               'previous_link': previous_link,
               'next_link': next_link,
               'black_list': json.dumps(black_list)}

    return render(request, 'detection/images_compare.html', context)


def view_image_result(request, data, split, version, label, start_id):
    '''
    use js to render the box in the client side
    '''
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    start_id = int(float(start_id))
    images = visualize_box_no_draw(data, split, version, label, start_id)
    all_type_to_rects = []
    all_url = []
    max_image_shown = 50
    all_key = []
    for i, (fname, origin, gt) in enumerate(images):
        if i >= max_image_shown:
            break
        origin_html_path = save_image_in_static(origin, '{}/{}/{}/origin_{}.jpg'.format(data, split,
                                                                                        version,
                                                                                        fname))
        all_key.append(fname)
        all_url.append('/static/' + origin_html_path)
        all_type_to_rects.append({'gt': gt})
    os.chdir(curr_dir)

    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(max(0, start_id - max_image_shown))
    previous_link = reverse('detection:view_image3')
    previous_link = previous_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])
    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(start_id + len(all_type_to_rects))
    next_link = reverse('detection:view_image3')
    next_link = next_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])

    context = {'all_type_to_rects': json.dumps(all_type_to_rects),
               'target_label': label,
               'all_url': json.dumps(all_url),
               'all_key': json.dumps(all_key),
               'previous_link': previous_link,
               'next_link': next_link}

    return render(request, 'detection/images_js3.html', context)


def view_image_js(request, data, split, version, label, start_id, imKey=None):

    '''
    use js to render the box in the client side
    '''
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    start_id = int(float(start_id))
    if imKey is not None:
        start_id = 0

    images = visualize_box_no_draw(data, split, version, label, start_id)

    blkLstFilename = "../data/{}/blacklist.txt".format(data)
    black_list = []
    if os.path.isfile(blkLstFilename):
        with open(blkLstFilename, 'r') as f:
            lines = f.readlines()
            black_list = [line.strip('\n') for line in lines]

    all_type_to_rects = []
    all_url = []
    max_image_shown = 50
    all_key = []

    if imKey is None:
        for i, (fname, origin, gt) in enumerate(images):
            
            if i >= max_image_shown:
                break
            
            filename, file_extension = os.path.splitext(fname)
            
            from qd_common import hash_sha1
            origin_html_path = save_image_in_static(origin, '{}/{}/{}/origin_{}.jpg'.format(data, split,
                                                                                            version,
                                                                                            hash_sha1(fname)))
            all_key.append(fname)
            all_url.append('/static/' + origin_html_path)
            all_type_to_rects.append({'gt': gt})
    else:
        for i, (fname, origin, gt) in enumerate(images):
            # print "imKey", imKey, fname
            if imKey in fname:
                
                filename, file_extension = os.path.splitext(fname)
                
                from qd_common import hash_sha1
                origin_html_path = save_image_in_static(origin, '{}/{}/{}/origin_{}.jpg'.format(data, split,
                                                                                                version,
                                                                                                hash_sha1(fname)))
                all_key.append(fname)
                all_url.append('/static/' + origin_html_path)
                all_type_to_rects.append({'gt': gt})

    
    os.chdir(curr_dir)

    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(max(0, start_id - max_image_shown))
    previous_link = reverse('detection:view_image2')
    previous_link = previous_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])
    kwargs = copy.deepcopy(request.GET)
    kwargs['start_id'] = str(start_id + len(all_type_to_rects))
    next_link = reverse('detection:view_image2')
    next_link = next_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])

    context = {'all_type_to_rects': json.dumps(all_type_to_rects),
               'target_label': label,
               'all_url': json.dumps(all_url),
               'all_key': json.dumps(all_key),
               'previous_link': previous_link,
               'next_link': next_link,
               'black_list': json.dumps(black_list)}

    return render(request, 'detection/images_js2.html', context)

def view_compare_test(request):
    if request.GET.get('data', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        # name_splits_labels = get_all_data_info()
        names = get_all_data_info2()
        os.chdir(curr_dir)
        context = {'names': names}
        return render(request, 'detection/data_list_result.html', context)
    elif request.GET.get('split', '') == '' and request.GET.get('label', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        name_splits_labels = get_all_data_info2(request.GET['data'])
        os.chdir(curr_dir)
        context = {'name_splits_label_counts': name_splits_labels}
        return render(request, 'detection/image_overview_result.html', context)
    else:
        data = request.GET.get('data')
        split = request.GET.get('split')
        if split == 'None':
            split = None
        version = request.GET.get('version')
        if version == 'None':
            version = None
        version = int(version) if type(version) is str or \
            type(version) is unicode else version
        label = request.GET.get('label')
        start_id = request.GET.get('start_id')

        min_conf = request.GET.get('min_conf')
        if min_conf == 'None':
            min_conf = None

        result = view_image_compare_test(
            request, data, split, version, label, start_id)
        return result

def view_compare_test(request):
    data = request.GET.get('data')
    split = request.GET.get('split')
    if split == 'None':
        split = None
    version = request.GET.get('version')
    if version == 'None':
        version = None
    version = int(version) if type(version) is str or \
        type(version) is unicode else version
    label = request.GET.get('label')
    start_id = request.GET.get('start_id')

    min_conf = request.GET.get('min_conf')
    if min_conf == 'None':
        min_conf = None
    else:
        min_conf = float(0.6)

    result = view_image_compare_test(
        request, data, split, version, label, start_id, min_conf)

    return result


def findVersion(longversion):
    pos0 = longversion.find(".")
    pos1 = longversion.rfind(".")

    return longversion[pos0+1:pos1]

def get_compare_all_info(config_path=None):
    if config_path is None:
        cmp_config = './data/compare_config.yaml'
    else:
        cmp_config = config_path
    
    configs = load_from_yaml_file(cmp_config)
    cmp_tasks = "compare"
    tasks = configs[cmp_tasks]

    # print "tasks", tasks
    results = []
    for cmp_task in tasks:
        # print cmp_task
        config_name = cmp_task["datapath"]
        task_name = cmp_task["task"]
        task_config =  op.join(config_name, "config.yaml")
        if op.isfile(task_config):
            db_list = cmp_task["dataset"]
            # print db_list
            results.extend(get_compare_data_info(task_name, config_name, db_list))
    # print results
    return results

def get_compare_data_info(task_name, config_path=None, dblist=None):
    if config_path is None:
        predictConfig = './data/compare/config.yaml'
    else:
        predictConfig = op.join(config_path, "config.yaml")

    configs = load_from_yaml_file(predictConfig)
    results = []

    if dblist:
    # dataset_name_01 = 'Instagram'
    # dataset_name_02 = 'MIT1K'

        for db in dblist:
            
            configs_dict = configs[db]
            configs_bl = configs_dict['baselines']
            
            for bl1 in configs_bl:
            # if (bl1['name'].find("google"))>=0 or (bl1['name'].find("amazon"))>=0:
                for bl2 in configs_bl:
                    if bl1['name'].find(bl2['name'])<0:
                        if (bl2['name'].find("google"))<0 and (bl2['name'].find("amazon"))<0 and (bl2['name'].find("aws"))<0:

                            if "conf_threshold" in bl2:
                                min_conf_left = bl2["conf_threshold"]
                            else:
                                min_conf_left = 0
                            
                            if "conf_threshold" in bl1:
                                min_conf_right = bl1["conf_threshold"]
                            else:
                                min_conf_right = 0

                            compare_name = task_name + "_" + db + "_" + bl2['name'] + "_vs_" + bl1['name']
                            # print compare_name
                            results.append({"name": compare_name, "min_conf_left": min_conf_left, "min_conf_right": min_conf_right })
    return results

def parse_compare_data(data_source_name):
    
    items = [x.strip() for x in data_source_name.split('_')]

    # print "items", items
    task = items[0]
    dataSource = items[1]
    leftLabel = items[2]
    rightLabel = items[4]

    os.chdir(get_qd_root())

    all_config  =  './data/compare_config.yaml'
    all_configs = load_from_yaml_file(all_config)
    cmp_configs = all_configs["compare"]
    
    

    predictConfigPath = ""
    for cmp_config in cmp_configs:
        if cmp_config["task"] == task:
            predictConfigPath = cmp_config["datapath"]
            

    predictConfig = op.join(predictConfigPath, "config.yaml")

    configs = load_from_yaml_file(predictConfig)

    configs_data = configs[dataSource]
    configs_data_bl = configs_data['baselines']

    gd = configs_data["groundtruth"]
    # print gd
    orig = gd["original"]

    image_db = ""
    image_db = orig["dataset"]

    thresholdFileLeft = None
    displayFileLeft = None
    thresholdFileRight = None
    displayFileRight = None
    
    for bl in configs_data_bl:
        if bl['name'] == leftLabel:
            leftFile = bl['result']
            if 'threshold' in bl:
                thresholdFileLeft = bl['threshold']
            if 'display' in bl:
                displayFileLeft = bl['display']
        if bl['name'] == rightLabel:
            rightFile = bl['result']
            if 'threshold' in bl:
                thresholdFileRight = bl['threshold']
            if 'display' in bl:
                displayFileRight = bl['display']
  
    return leftFile, rightFile, image_db, leftLabel, rightLabel, thresholdFileLeft, thresholdFileRight, displayFileLeft, displayFileRight

def view_compare_all(request):

    if request.GET.get('data', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        results = get_compare_all_info()
        os.chdir(curr_dir)
        context = {'results': results}
        return render(request, 'detection/compare_list_result.html', context)
    return

def view_compare(request):
    import shutil
    if request.GET.get('data', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        results = get_compare_all_info()
        os.chdir(curr_dir)
        context = {'results': results}
        return render(request, 'detection/compare_list_result.html', context)
    else:
        # task = request.GET.get('task')

        data = request.GET.get('data')
        split = request.GET.get('split')
        if split == 'None':
            split = None
        version = request.GET.get('version')
        if version == 'None':
            version = None
        version = int(version) if type(version) is str or \
            type(version) is unicode else version

        label = request.GET.get('label')
        start_id = request.GET.get('start_id')

        min_conf_left = request.GET.get('min_conf_left')
        if min_conf_left != None:
            min_conf_left = float(min_conf_left)
        
        min_conf_right = request.GET.get('min_conf_right')
        if min_conf_right != None:
            min_conf_right = float(min_conf_right)
        
        # if data isn't exist
        dataPath = "./data/" + data
        os.chdir(get_qd_root())

        print "dataPath:", dataPath
        # print data
        leftFile, rightFile, data_source_name, leftLabel, rightLabel, leftThreshold, rightThreshold, leftDisplay, rightDisplay = parse_compare_data(data)

        items = [x.strip() for x in data.split('_')]
        # print "items", items
        task = items[0]
        os.chdir(get_qd_root())

        all_config  =  './data/compare_config.yaml'
        all_configs = load_from_yaml_file(all_config)
        cmp_configs = all_configs["compare"]

        predictConfigPath = ""
        for cmp_config in cmp_configs:
            if cmp_config["task"] == task:
                predictConfigPath = cmp_config["datapath"]


        leftFile = op.join(predictConfigPath, leftFile)
        rightFile = op.join(predictConfigPath, rightFile)


        imageFile = "./data/" + data_source_name +  "/" + split + ".tsv"

        print leftFile
        print rightFile
        print imageFile

        if os.path.isfile(dataPath +"/test.tsv"):
            if worth_create(leftFile, dataPath +"/test.tsv") or worth_create(rightFile, dataPath +"/test.tsv"):
                shutil.rmtree( dataPath )

        if not op.isdir(dataPath):
            os.mkdir(dataPath, 0755)

            if leftThreshold != None:
                leftThreshold = op.join(predictConfigPath, leftThreshold)

            if rightThreshold != None:
                rightThreshold = op.join(predictConfigPath, rightThreshold)
            
            if leftDisplay != None:
                leftDisplay = op.join(predictConfigPath, leftDisplay)
            
            if rightDisplay != None:
                rightDisplay = op.join(predictConfigPath, rightDisplay)
            
            # print leftFile, rightFile, imageFile, leftLabel, rightLabel

            build_side_by_side_compare(leftFile,
                rightFile,
                imageFile,
                dataPath +"/test.tsv",
                leftThreshold,
                rightThreshold,
                leftLabel,
                rightLabel,
                leftDisplay,
                rightDisplay)

            black_list_filename = op.join(predictConfigPath, "blacklist.txt")

            if op.isfile(black_list_filename):
                shutil.copy2(black_list_filename, "./data/"+data)

            populate_dataset_details(data)

        # print request, data, split, version
        result = view_image_compare(
            request, data, split, version, label, start_id, min_conf_left, min_conf_right)

        return result


def view_result(request):
    if request.GET.get('data', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        # name_splits_labels = get_all_data_info()
        names = get_all_data_info2()
        os.chdir(curr_dir)
        context = {'names': names}
        return render(request, 'detection/data_list_result.html', context)
    elif request.GET.get('split', '') == '' and request.GET.get('label', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        name_splits_labels = get_all_data_info2(request.GET['data'])
        os.chdir(curr_dir)
        context = {'name_splits_label_counts': name_splits_labels}
        return render(request, 'detection/image_overview_result.html', context)
    else:
        data = request.GET.get('data')
        split = request.GET.get('split')
        if split == 'None':
            split = None
        version = request.GET.get('version')
        if version == 'None':
            version = None
        version = int(version) if type(version) is str or \
            type(version) is unicode else version
        label = request.GET.get('label')
        start_id = request.GET.get('start_id')
        result = view_image_js3(request, data, split, version, label, start_id)
        return result


def view_verify_data(request, data, split, version, label, start_id):
    start_id = int(float(start_id))
    proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    collage_folder = "detection/static/detection/{}/{}/collage".format(
        data, split)

    # filelist = glob.glob(os.path.join(proj_root, collage_folder))
    filelist_fname = os.path.join(collage_folder, "filenamelist.txt")
    if os.path.isfile(filelist_fname):
        with open(filelist_fname, 'r') as f:
            lines = f.readlines()
            filelist = [line.strip('\n') for line in lines]
        # [s.strip() for s in self._fp.readline().split('\t')]
    # filelist.sort()
    term_number = len(filelist)
    all_url = []
    all_key = []
    fname = filelist[start_id]

    base = os.path.basename(fname)
    basename = os.path.splitext(base)[0]
    term_names = [s.strip() for s in basename.split('_')]
    term_name = term_names[0]

    j = 1
    while not isfloat(term_names[j]):
        term_name = term_name + " " + term_names[j]
        j = j+1

    origin_html_path = 'detection/{}/{}/collage/{}.jpg'.format(
        data, split, basename)

    all_key.append(term_name)
    all_url.append('/static/' + origin_html_path)

    label_fname = os.path.join(
        proj_root, "detection/static", os.path.splitext(origin_html_path)[0] + '.txt')
    # option = label_fname #for debug
    comment = ""
    if os.path.isfile(label_fname):
        with open(label_fname, 'r') as f:
            line = f.readline()
            option = line.rstrip('\n')
            comment = f.read()
    else:
        option = None

    # os.chdir(curr_dir)

    kwargs = copy.deepcopy(request.GET)
    if start_id == 0:
        kwargs['start_id'] = str(term_number-1)
    else:
        kwargs['start_id'] = str(max(0, start_id - 1))

    previous_link = reverse('detection:verify_data')
    previous_link = previous_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])

    kwargs = copy.deepcopy(request.GET)

    if start_id == term_number-1:
        kwargs['start_id'] = str(0)
    else:
        kwargs['start_id'] = str(min(term_number-1, start_id + 1))

    next_link = reverse('detection:verify_data')
    next_link = next_link + '?' + \
        '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs])

    context = {
        'all_url': json.dumps(all_url),
        'all_key': json.dumps(all_key),
        'option': json.dumps(option),
        'next_id': kwargs['start_id'],
        'comment': comment,
        'label_fname': label_fname,
        'data': data,
        'split': split,
        'previous_link': previous_link,
        'next_link': next_link
    }

    return render(request, 'detection/verify_collage.html', context)


def verify_data(request):

    if request.method == 'POST':
        label_fname = request.POST.get('label_fname')

        option = request.POST.get('radio_option')
        # option_2 = request.POST.get('radio_option')
        if option != '1' and option != '2':
            option = '3'

        comment = request.POST.get('comment')

        with open(label_fname, 'w') as f:
            f.write("{}\n".format(str(option)))
            f.write(comment)

        start_id = request.POST.get('next_id')
        data = request.POST.get('data')
        split = request.POST.get('split')
    else:
        start_id = request.GET.get('start_id')
        data = request.GET.get('data')
        split = request.GET.get('split')
    if start_id == None:
        start_id = '0'

    if split == None:
        split = 'train'

    version = request.GET.get('version')
    if version == 'None':
        version = None
    version = int(version) if type(version) is str or \
        type(version) is unicode else version

    if data == None:
        data = 'Tax1300V14_1_with_bb'

    label = request.GET.get('label')
    if label == 'None':
        label = 'aardwolf'

    return view_verify_data(request, data, split, version, label, start_id)

def view_image(request):
    if request.GET.get('data', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        # name_splits_labels = get_all_data_info()
        names = get_all_data_info2()
        os.chdir(curr_dir)
        context = {'names': names}
        return render(request, 'detection/data_list.html', context)
    elif request.GET.get('split', '') == '' and request.GET.get('label', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())

        data = request.GET.get('data')
        name_splits_labels = get_all_data_info2(data)
        os.chdir(curr_dir)

        populate_dataset_details(data)
        context = {'name_splits_label_counts': name_splits_labels}
        return render(request, 'detection/image_overview.html', context)
    else:
        data = request.GET.get('data')
        split = request.GET.get('split')
        key = None
        key = request.GET.get('key')

        populate_dataset_details(data)

        if split == 'None':
            split = None
        version = request.GET.get('version')
        if version == 'None':
            version = None
        version = int(version) if type(version) is str or \
            type(version) is unicode else version
        
        label = request.GET.get('label')
        start_id = request.GET.get('start_id')
        
        result = view_image_js(request, data, split, version, label, start_id, key)
        return result


def get_data_sources_for_composite():
    datas = ['coco2017',
            'voc0712',
            'brand1048Clean',
            'imagenet3k_448Clean',
            'imagenet22k_448',
            'imagenet1kLocClean',
            'mturk700_url_as_keyClean',
            'crawl_office_v1',
            'crawl_office_v2',
            'Materialist',
            'VisualGenomeClean',
            'Naturalist',
            '4000_Full_setClean',
            'MSLogoClean',
            'clothingClean',
            'open_images_clean_1',
            'open_images_clean_2',
            'open_images_clean_3',
            ]
    return datas


@csrf_exempt
def input_taxonomy(request):
    print 'in input taxonomy'
    if request.method == 'POST':
        return validate_taxonomy2(request)
    datas = get_data_sources_for_composite()
    str_datas = ','.join(datas)
    return render(request,
                  'detection/get_tax_info.html',
                  {'str_datas': str_datas})


@csrf_exempt
def validate_taxonomy2(request):
    from vis_bkg import push_task
    task = push_task(request)
    messages = []
    messages.append('We received your request and will process soon. ')
    messages.append('Please check the log \\\\viggpu01.redmond.corp.microsoft.com\\glusterfs\\jianfw\\work\\qd_output\\vis_bkg\\{}\\log.txt to monitor the process'.format(
        task['task_id']))

    return render(request, 'detection/display_tax_results.html',
                  {'error': '\n'.join(messages)})


@csrf_exempt
def validate_taxonomy(request):
    logging.info('In validate taxonomy')
    logging.info(request)
    logging.info(request.POST.get('file_input'))
    out_data = request.POST.get('out_data')
    if out_data is None or len(out_data) == 0:
        return render(request, 'detection/display_tax_results.html',
                      {'error': 'Please specify the out dataset name'})
    handle_uploaded_file(request.FILES['file_input'])
    kwargs_mimic = dict()
    kwargs_mimic['type'] = 'taxonomy_to_tsv'
    kwargs_mimic['input'] = op.join(get_qd_root(),
                                    'visualization/taxonomy/')
    kwargs_mimic['data'] = out_data
    kwargs_mimic['datas'] = [s.strip() for s in
                             request.POST.get('str_datas').split(',')]
    if len(kwargs_mimic['datas']) <= 0 or \
            len(kwargs_mimic['datas']) == 1 and kwargs_mimic['datas'][0] == '':
        return render(request, 'detection/display_tax_results.html',
                      {'error': 'Please specify at least one data source'})
    try:
        build_taxonomy_impl(kwargs_mimic['input'], **kwargs_mimic)
    except Exception, e:
        print str(e)
        context = dict()
        context['files'] = []
        if str(e) == "":
            context['error'] = 'Taxonomy successfully verified'
            files = return_download_list(op.join(get_qd_root(),
                                                 'data/{}/'.format(out_data)))
            context['files'] = files
        else:
            trace = (traceback.format_exc())
            context['error'] = trace
        return render(request, 'detection/display_tax_results.html', context)
    context = {'error': 'Taxonomy successfully verified'}
    files = return_download_list(op.join(get_qd_root(),
                                         'data', out_data))
    context['files'] = files
    return render(request, 'detection/display_tax_results.html', context)


def handle_uploaded_file(f):
    fname = op.join(get_qd_root(), 'visualization/taxonomy/taxonomy.yaml')
    with open(fname, 'w') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def return_download_list(folder):
    import os
    import shutil
    files = []
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if not op.isfile(file_path):
            continue
        new_folder = op.join(get_qd_root(), 'visualization/mysite/media/')
        shutil.copy(file_path, new_folder)
        file_path = os.path.join(new_folder, the_file)
        fp = open(file_path, 'rb')
        f = File.objects.create(name=the_file, file = django.core.files.File(fp))
        f.url = '/media/' + f.file.name
        f.save()
        files.append(f)
    return files


def clean_up_taxonomy_folders(folder):
    import os
    import shutil
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.islink(file_path):
                continue
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def download_file(request, *callback_args, **callback_kwargs):
    request_url = ' %s' % (request.get_full_path)
    print request_url
    file_path = request_url.split('\'')[1].split(
        'http://10.137.68.61:8000/detection/media/')[0].replace('/detection/media', '')
    file_type = file_path.split('/')[-1].split('.')[-1]
    file_name = file_path.split('/')[-1]
    print file_path
    print file_type
    FilePointer = open(file_path, "r")
    response = HttpResponse(FilePointer, content_type='application/'+file_type)
    response['Content-Disposition'] = 'attachment; filename=' + file_name
    return response
