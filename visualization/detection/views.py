# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import sys
import os
import os.path as op
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
from process_tsv import build_taxonomy_impl
import sys
import traceback2 as traceback
from .models import *
import django.core.files
import logging
import uuid
from qd_common import load_class_ap

init_logging()

def run_in_qd(func, *args, **kwargs):
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    r = func(*args, **kwargs)
    os.chdir(curr_dir)
    return r

def view_tree(request):
    if 'data' in request.GET:
        data = request.GET.get('data')
        s = run_in_qd(gen_html_tree_view, data)
        context = {'li_ul_tree_str': s}
        return render(request, 'detection/view_tree.html', context)
    else:
        data_list = run_in_qd(get_all_tree_data)
        context = { 'data_names': data_list }
        return render(request, 'detection/view_list_has_tree.html', context)

def view_single_model(request):
    '''
    deprecated
    '''
    data = request.GET.get('data', 'office_v2.12')
    net = request.GET.get('net', 
            'darknet19_448')
    test_data = request.GET.get('test_data', 
            data)
    expid = request.GET.get('expid', 
            'A_noreorg_burnIn5e.1_tree_initFrom.imagenet.A_bb_nobb')
    label = request.GET.get('filter_label', 'any')
    start_id = int(float(request.GET.get('start_id', '0')))
    threshold = 0.1
    
    extra_param = {}
    if 'test_input_sizes' in request.GET:
        extra_param['test_input_sizes'] = [int(float(s)) for s in
            request.GET.getlist('test_input_sizes')]
    # qd code
    curr_dir_backup = os.getcwd()
    os.chdir(get_qd_root())
    x = get_confusion_matrix(data, 
            net, test_data, expid, threshold, **extra_param)
    predicts, gts, label_to_idx = x['predicts'], x['gts'], x['label_to_idx']
    confusion_pred_gt = x['confusion_pred_gt']
    confusion_gt_pred = x['confusion_gt_pred']
    os.chdir(curr_dir_backup)
    image_pairs = []
    target_images, image_aps = get_target_images(predicts, gts, label, threshold)
    pairs = gt_predict_images(predicts, gts, test_data, target_images,
            start_id,
            threshold, label_to_idx, image_aps)
    max_pairs = 10
    for i, (key, im_gt, im_pred, ap) in enumerate(pairs):
        path_gt = save_image_in_static(im_gt, '{}/{}/{}/{}/{}_gt.png'.format(
            data, net, expid, test_data, key))
        path_pred = save_image_in_static(im_pred, '{}/{}/{}/{}/{}_pred.png'.format(
            data, net, expid, test_data, key))
        image_pairs.append((path_gt, path_pred, ap))
        if i >= max_pairs - 1:
            break
    
    common_param = [('data', data), ('net', net), ('expid', expid),
            ('filter_label', label), ('test_data', test_data)]
    for key in extra_param:
        value = extra_param[key]
        if type(value) is list:
            for v in value:
                common_param.append((key, str(v)))
        else:
            assert False
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
    context = {
            'previous_button_param': previous_button_param,
            'next_button_param': next_button_param,
            'label': label,
            'confusion_pred_gt': html_confusion_pred_gt,
            'confusion_gt_pred': readable_confusion_entry(
                confusion_gt_pred[label]),
            'image_pairs': image_pairs}

    return render(request, 'detection/predict_result.html', context)


def view_exp_list(request):
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    full_expids = get_all_model_expid()
    os.chdir(curr_dir)
    full_expids.sort()
    context = {'full_expids': full_expids}
    return render(request, 'detection/exp_list.html', context)

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
            label,
            threshold)
    predicts, gts, label_to_idx = x['predicts'], x['gts'], x['label_to_idx']
    confusion_pred_gt = x['confusion_pred_gt']
    confusion_gt_pred = x['confusion_gt_pred']
    os.chdir(curr_dir_backup)

    image_pairs = []
    target_images, image_aps = get_target_images(predicts, gts, label, threshold)
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
        image_pairs.append((path_origin, path_gt_target, path_pred_target, path_gt, path_pred, ap))
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
        class_ap = class_ap['overall']['0.3']['class_ap']
        labelmap_ap = [(l, '{:.2f}'.format(class_ap.get(l, -1))) for l in labelmap]
    else:
        labelmap_ap = [(l, None) for l in labelmap]
    os.chdir(curr_dir)
    context = {'prediction_file': predict_file,
            'labelmap_ap': labelmap_ap,
            'full_expid': full_expid}
    return render(request, 'detection/view_model_prediction_labelmap.html',
            context)

def view_model(request):
    if 'full_expid' in request.GET and \
            'predict_file' in request.GET and \
            'filter_label' in request.GET and \
            'start_id' in request.GET and \
            'threshold' in request.GET:
        return view_model_by_predict_file(request, 
                request.GET['full_expid'],
                request.GET['predict_file'],
                request.GET['filter_label'],
                request.GET['start_id'],
                request.GET['threshold'])
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
        #return view_single_model(request)

def save_image_in_static(im, rel_path):
    html_path = 'detection/{}'.format(rel_path)
    disk_path = op.join(op.dirname(__file__), 'static', html_path)
    #if im.shape[0] > 400:
        #factor = 400. / im.shape[0]
        #im = cv2.resize(im, (int(im.shape[1] * factor), 400))
    save_image(im, disk_path)
    return html_path

def view_image(request, data, split, version, label, start_id):
    curr_dir = os.curdir
    os.chdir(get_qd_root())
    start_id = int(float(start_id))
    images = visualize_box(data, split, version, label, start_id)
    html_image_paths = []
    max_image_shown = 10
    has_next = False
    for i, (fname, origin, origin_label, im) in enumerate(images):
        if i >= max_image_shown:
            has_next = True
            break 
        origin_html_path = save_image_in_static(origin, '{}/{}/{}/origin_{}.jpg'.format(data, split,
            version,
            fname))
        origin_label_html_path = save_image_in_static(origin_label, '{}/{}/{}/origin_label_{}.jpg'.format(data, split,
            version,
            fname))
        html_path = save_image_in_static(im, '{}/{}/{}/bb_{}.jpg'.format(data, split,
            version,
            fname))
        html_image_paths.append((origin_html_path, origin_label_html_path , html_path))
    os.chdir(curr_dir)

    context = {'images': html_image_paths,
            'data': data,
            'split': split,
            'version': version,
            'label': label,
            'next_id': str(start_id + len(html_image_paths)),
            'previous_id': str(max(0, start_id - max_image_shown))}
    return render(request, 'detection/images.html', context)

def view_image2(request):
    if request.GET.get('data', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        #name_splits_labels = get_all_data_info()
        names = get_all_data_info2()
        os.chdir(curr_dir)
        context = {'names': names}
        return render(request, 'detection/data_list.html', context)
    elif request.GET.get('split', '') == '' and request.GET.get('label', '') == '':
        curr_dir = os.curdir
        os.chdir(get_qd_root())
        name_splits_labels = get_all_data_info2(request.GET['data'])
        os.chdir(curr_dir)
        context = {'name_splits_labels': name_splits_labels}
        return render(request, 'detection/image_overview.html', context)
    else:
        data = request.GET.get('data')
        split = request.GET.get('split')
        version = request.GET.get('version')
        version = int(version) if type(version) is str or type(version) is \
                unicode else version
        label = request.GET.get('label')
        start_id = request.GET.get('start_id')
        result = view_image(request, data, split, version, label, start_id)
        return result

def get_data_sources_for_composite():
    datas=['coco2017',
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
    	context=dict()
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
    context ={'error': 'Taxonomy successfully verified'}
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
    import os, shutil
    files = []
    for the_file in os.listdir(folder):
    	file_path = os.path.join(folder, the_file)
        if not op.isfile(file_path):
            continue
    	new_folder = op.join(get_qd_root(), 'visualization/mysite/media/')
    	shutil.copy(file_path, new_folder)
    	file_path = os.path.join(new_folder, the_file)
    	fp = open(file_path, 'rb')
    	f = File.objects.create(name = the_file, file = django.core.files.File(fp))
    	f.url = '/media/' + f.file.name
    	f.save()
    	files.append(f)
    return files

def clean_up_taxonomy_folders(folder):
    import os, shutil
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
    request_url =' %s' % (request.get_full_path)
    print request_url
    import ipdb;ipdb.set_trace(context=15)
    file_path = request_url.split('\'')[1].split('http://10.137.68.61:8000/detection/media/')[0].replace('/detection/media','')
    file_type = file_path.split('/')[-1].split('.')[-1]
    file_name = file_path.split('/')[-1]
    print file_path
    print file_type
    FilePointer = open(file_path,"r")
    response = HttpResponse(FilePointer,content_type='application/'+file_type)
    response['Content-Disposition'] = 'attachment; filename=' + file_name
    return response
