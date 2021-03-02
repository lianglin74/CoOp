#!/usr/bin/env python
import torch
import pickle
import copy
from qd.qd_common import encode_np
import csv
import glob
import logging
import math
import multiprocessing as mp
import base64
import os
import os.path as op
import random
import re
import shutil
import sys
import warnings
import time
import pickle
from collections import OrderedDict, defaultdict
from qd.qd_common import print_trace
from pprint import pformat, pprint
from qd.process_tsv import create_versions_from_uncertain_label
from qd.qd_common import execute_func
from qd.qd_common import get_frame_info
from qd.qd_common import ensure_copy_folder
from qd.qd_common import get_file_size
from qd.qd_common import decode_np
from qd.qd_common import dict_get_all_path
from qd.gpucluster.job_scheduler import JobScheduler
from qd.process_tsv import convertcomposite_to_standard
from qd.pipeline import run_training_pipeline
from qd.qd_common import inject_maskrcnn_log_to_board
from qd.torch_common import describe_tensor
from qd.qd_pytorch import ensure_init_process_group
from qd.process_tsv import find_best_matched_rect
from qd.gpucluster import create_aml_client
from qd.process_tsv import expand_nested_splitX
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import json
import six
import torch
from future.utils import viewitems
from pymongo import MongoClient
from sklearn.manifold import TSNE
from qd.qd_common import qd_tqdm as tqdm

from qd.batch_process import remote_run_func
from qd.cloud_storage import CloudStorage, create_cloud_storage
from qd.cocoeval import coco_eval_json
from qd.db import (AnnotationDB, create_annotation_db,
                   create_bbverification_db, create_bbverificationdb_client,
                   create_mongodb_client)
from qd.deteval import deteval_iter
from qd.evaluate.evaluate_openimages_google import parallel_oi5c_to_submit_csv
from qd.hnms import SingleHashNMS, calc_lower_iou_bound
from qd.philly import create_philly_client
from qd.pipeline import (aml_func_run, get_all_test_data, load_pipeline,
                         philly_func_run, pipeline_demo, pipeline_eval_multi,
                         pipeline_pred_eval)
from qd.pipelines.auto_param import AutoParam
from qd.process_image import (draw_bb, draw_rects, load_image,
                              network_input_to_image, put_text, save_image,
                              show_image, show_images, show_net_input)
from qd.process_tsv import (CogAPI, TSVFile, build_tax_dataset_from_db,
                            build_taxonomy_from_single_source,
                            build_taxonomy_impl, concat_tsv_files,
                            convert_one_label, ensure_inject_dataset,
                            ensure_inject_decorate, ensure_inject_expid_pred,
                            ensure_inject_gt, ensure_inject_image,
                            ensure_inject_pred, ensure_upload_image_to_blob,
                            find_best_matched_rect_idx,
                            find_same_location_rects, find_same_rects,
                            get_data_sources, get_taxonomy_path, hash_sha1,
                            img_from_base64, inc_one_dic_dic, load_key_rects,
                            load_labels, normalize_to_str,
                            parallel_multi_tsv_process, parallel_tsv_process,
                            parse_combine, populate_dataset_details,
                            populate_dataset_hw, rect_in_rects,
                            regularize_data_sources, softnms_row_process,
                            update_confusion_matrix, visualize_tsv2)
from qd.qd_common import (calc_mean, calculate_image_ap2, calculate_iou,
                          check_best_iou, cmd_run, concat_files, copy_file,
                          dict_ensure_path_key_converted, dict_get_path_value,
                          dict_has_path, dict_set_path_if_not_exist,
                          dict_to_list, dict_update_nested_dict,
                          dict_update_path_value, encode_expid,
                          encoded_from_img, ensure_copy_file, ensure_directory,
                          ensure_remove_dir, find_float_tolorance_unequal,
                          float_tolorance_equal, get_current_time_as_str,
                          get_mpi_rank, get_mpi_size, init_logging,
                          iter_swap_param, json_dump, list_to_dict,
                          load_from_yaml_file, load_list_file, parse_yolo_log,
                          parse_yolo_log_acc, parse_yolo_log_st,
                          print_offensive_folder, print_table, read_to_buffer,
                          remove_dir, run_if_not_cached, set_if_not_exist,
                          softnms, softnms_c, try_delete, url_to_str,
                          write_to_file, write_to_yaml_file, zip_qd)
from qd.qd_pytorch import calc_neg_aware_gmap, torch_load, torch_save
from qd.taxonomy import (Taxonomy, get_nick_name, load_label_parent,
                         noffset_to_synset, synset_to_noffset)
from qd.tsv_io import (TSVDataset, csv_reader, is_verified_rect, tsv_reader,
                       tsv_writer, tsv_writers)
from scripts.qd_util import multi_softnms_best_xi

from qd.process_tsv import duplicate_balance_fg_classes
from qd.pipeline import get_pipeline_waiting_full_expid

import pickle as pkl
try:
    from demo_detection import test_demo
    from caffe.proto import caffe_pb2
except:
    pass

sys.path.append('./scripts')
try:
    from qd_util import draw_gt, draw_circle
    from qd_util import BatchProcess
    from qd_util import parse_test_data, dataset_op_select
    from qd_util import url_to_image
    from qd_util import read_blob
    from qd_util import lmdb_reader
    from qd_util import philly_upload
    from qd_util import philly_mkdir
    from qd_util import philly_upload_dir
    from qd_util import philly_ls
    from qd_util import update_rects_within_image
    from qd_util import add_prediction_into_train
    from qd_util import upload_qdoutput
    from qd_util import yolo_predict
    from qd_util import philly_upload_qdoutput
    from qd_util import ocr_request
except:
    pass
try:
    from cStringIO import StringIO
except:
    pass
try:
    from yolodet import im_detect, result2bblist
    from yolodet import tsvdet, prepare_net_input
except:
    pass
try:
    from quickcaffe.modelzoo import VGGStyle
    from qd.yolotrain import yolotrain
    from qd.yolotrain import CaffeWrapper
    import caffe
except:
    pass

try:
    from itertools import izip as zip
except:
    pass

from qd.qd_common import execute_pipeline
from bson import ObjectId
from qd.db import query_acc_by_full_expid
from qd.db import query_job_acc
from qd.qd_common import try_once
from qd.pipeline_runner import PipelineRunner
from qd.qd_common import query_path_by_suffix
from qd.pipeline import verify_param
from qd.process_tsv import merge_dataset, merge_pred_label_to_gt
from qd.pipeline import env_run
from qd.qd_common import write_to_file
from qd.tsv_io import tsv_reader
from qd.qd_common import qd_tqdm as tqdm
from qd.tsv_io import TSVDataset
import os.path as op


def test_merge_open_image_caption_prod114():
    dataset = TSVDataset('TaxOpenImageSplit64')
    for i in range(64):
        split = 'train_{}_64'.format(i)
        logging.info(split)
        iter1 = dataset.iter_data(split, 'caption')
        iter2 = dataset.iter_data(split, 'caption', 'prod114')
        def gen_rows():
            for r1, r2 in tqdm(zip(iter1, iter2)):
                assert r1[0] == r2[0]
                x = json.loads(r1[1])
                y = json.loads(r2[1])
                x.extend(y)
                yield r1[0], json_dump(x)
        def gen_info():
            info = get_frame_info()
            for k, v in info.items():
                yield k, v
        dataset.safe_write_data(
            gen_rows(), split, 'caption', 'prod114merge',
            generate_info=gen_info(),
            force=True,
        )

    all_f = []
    for i in range(64):
        split = 'train_{}_64'.format(i)
        all_f.append(dataset.get_data(split, 'caption', 'prod114merge'))
    write_to_file('\n'.join(all_f),
                  dataset.get_data('trainX', 'caption', 'prod114merge')
                  )

def test_distribute_caption114_to_64_splits():
    from qd.process_tsv import split_dataset_one
    split_dataset_one(
        data='OpenimageV6', num_split=64, out_data='TaxOpenImageSplit64',
        split_type='caption', split='train', version='prod114',
    )

def test_create_openimage_v6_caption114():
    folder = 'output/Tax-captionbot-office40k_clean_train.fea.penzhan2.lab.oid_X152_min10_db67c_ImLn1_BS256_EP60_LR1e-05_LabConf0.2_LbSm0.1/checkpoint-60-122340/'
    tsv_files = ['pred.openimages_v6tag.train.{}.fea.penzhan2.lab.gt-pred-tag-rmp-cl.beam1.max20.wlabels.tsv'.format(i)
                 for i in range(28)]
    tsv_files = [op.join(folder, t) for t in tsv_files]
    dataset = TSVDataset('OpenimageV6')
    iter_hw = dataset.iter_data('train', 'hw')
    def iter_caption():
        for t in tsv_files:
            for row in tqdm(tsv_reader(t)):
                yield row
    def gen_caption():
        for row_hw, row_caption in zip(iter_hw, iter_caption()):
            assert row_hw[0] == row_caption[0]
            yield row_caption
    dataset.safe_write_data(gen_caption(), 'train', 'caption', 'prod114')

def test_create_openimage_v6_caption():
    folder = 'datasets/openimages_v6tag/cap_pred/'
    tsv_files = ['pred.openimages_v6tag.train.{}.fea.vg.lab.gt.beam1.max20.wlabels.tsv'.format(i)
                 for i in range(28)]
    tsv_files = [op.join(folder, t) for t in tsv_files]
    dataset = TSVDataset('OpenimageV6')
    iter_hw = dataset.iter_data('train', 'hw')
    def iter_caption():
        for t in tsv_files:
            for row in tqdm(tsv_reader(t)):
                yield row
    def gen_caption():
        for row_hw, row_caption in zip(iter_hw, iter_caption()):
            assert row_hw[0] == row_caption[0]
            yield row_caption
    dataset.write_data(
        gen_caption(),
        'train', 'caption')

def test_create_splitx():
    data = 'TaxOpenImageSplit64'
    #version = 'pos'
    version = 'eff4PlusGt'
    version = 'eff3PlusGt'
    dataset = TSVDataset(data)
    all_f = []
    for i in range(64):
        split = 'train_{}_{}'.format(i, 64)
        f = dataset.get_data(split, 'label', version)
        all_f.append(f)
    write_to_file(
        '\n'.join(all_f),
        dataset.get_data('trainX', 'label', version)
    )

def test_write_fix_label():
    detector = 'TaxVisualGenome_frcnn_eff_attr_FCOS_basemodel139ee_RemoveEmptyFalse_Warm5e_BoxAgnostic_C3_BS256_MaxIter200e_LR0.1_NoARG_BNToSBN_IMIN448.1344_WD1e-05_IterResize_RC_AMP_mm_cut896_MB100_TV_EachSelect'
    in_version = 'eff3PlusGt'
    out_version = 'eff3PlusGt'
    extra_args = {
        'INPUT$MAX_SIZE_TEST': 896,
        'INPUT$MIN_SIZE_TEST': 800,
    }

    #detector = 'TaxVisualGenome_frcnn_eff_attr_FCOS_basemodelc8229_Warm5e_C4_BS256_MaxIter200e_LR0.2_NoARG_SBN_IMIN512.1536_WD1e-05_IR_RC_AMP_mm_cut1024_MB100_Roi1000_TV_EachSelect_noEmptyFilter_BoxAgnostic'
    #in_version = 'eff4PlusGt'
    #out_version = 'eff4PlusGt'
    #extra_args = {
        #'INPUT$MAX_SIZE_TEST': 1024,
    #}

    all_info = []

    from_data = []
    from_data.extend([
        {
            'test_data': 'TaxGoogleCCSplit64',
            'test_split': 'train_{}_64'.format(i),
        } for i in range(64)
    ])
    from_data.extend([
        {
            'test_data': 'TaxSBUSplit16',
            'test_split': 'train_{}_16'.format(i),
        } for i in range(16)
    ])
    from_data.extend([
        {
            'test_data': 'TaxCocoCaption',
            'test_split': 'train',
        },
        {
            'test_data': 'TaxGQABalanced',
            'test_split': 'train',
        },
        {
            'test_data': 'TaxFlickr30K',
            'test_split': 'train',
        },
        {
            'test_data': 'TaxVQA',
            'test_split': 'train',
        },
        {
            'test_data': 'TaxVisualGenomeQA',
            'test_split': 'train',
        },
        {
            'data': 'TaxOpenImageSplit64',
            'split': 'train',
            't': 'label',
            'version': in_version,
        },
    ])
    data = 'TaxCCSBUCocoGQAFlk30VqaVGqaOpenImageSplit'
    split = 'train'
    all_info.append({
        'from_data': from_data,
        'data': data,
        'split': split,
    })

    split_type = 'label'
    for info in all_info:
        from_data = info['from_data']
        data = info['data']
        split = info['split']
        all_pred = []
        for f in from_data:
            if 'test_data' in f:
                pip = load_pipeline(
                    full_expid=detector,
                    test_data=f['test_data'],
                    test_split=f['test_split'],
                    **extra_args,
                )
                pred = pip._get_predict_file()
                all_pred.append(pred)
            else:
                pred = load_list_file(TSVDataset(f['data']).get_data(
                    f['split'] + 'X',
                    f['t'],
                    f['version']
                ))
                all_pred.extend(pred)
        dataset = TSVDataset(data)
        if dataset.has(
            split + 'X',
            split_type,
            version=out_version):
            r = read_to_buffer(dataset.get_data(
                split + 'X',
                split_type,
                version=out_version)).decode()
            assert r == '\n'.join(all_pred)
        else:
            write_to_file(
                '\n'.join(all_pred),
                dataset.get_data(split + 'X', split_type, version=out_version)
            )

def test_merge_pred_label_to_gt_open_image():
    detector = 'TaxVisualGenome_frcnn_eff_attr_FCOS_basemodelc8229_Warm5e_C4_BS256_MaxIter200e_LR0.2_NoARG_SBN_IMIN512.1536_WD1e-05_IR_RC_AMP_mm_cut1024_MB100_Roi1000_TV_EachSelect_noEmptyFilter_BoxAgnostic'
    version = 'eff4PlusGt'
    extra_args = {
        'INPUT$MAX_SIZE_TEST': 1024,
    }
    from qd.prep_dataset.vlp_version import merge_pred_label_to_gt_open_image
    merge_pred_label_to_gt_open_image(detector, version, extra_args)

    detector = 'TaxVisualGenome_frcnn_eff_attr_FCOS_basemodel139ee_RemoveEmptyFalse_Warm5e_BoxAgnostic_C3_BS256_MaxIter200e_LR0.1_NoARG_BNToSBN_IMIN448.1344_WD1e-05_IterResize_RC_AMP_mm_cut896_MB100_TV_EachSelect'
    version = 'eff3PlusGt'
    extra_args = {
        'INPUT$MAX_SIZE_TEST': 896,
        'INPUT$MIN_SIZE_TEST': 800,
    }
    merge_pred_label_to_gt_open_image(detector, version, extra_args)

def merge_pred_label_to_gt_open_image(detector, version, extra_args):
    test_data = 'TaxOpenImageSplit64'
    c = create_aml_client(cluster='eu')
    from qd.qd_common import parallel_map
    all_param = []
    for i in range(64):
        test_split = 'train_{}_{}'.format(i, 64)
        pip = load_pipeline(
            full_expid=detector,
            test_data=test_data,
            test_split=test_split,
            **extra_args)
        pred = pip._get_predict_file()
        c.download(pred)
        from qd.tsv_io import get_tsv_lineidx
        c.download(get_tsv_lineidx(pred))
        all_param.append((pred, test_data, test_split, 'pos', version))
    from qd.process_tsv import tuple_merge_pred_label_to_gt
    parallel_map(tuple_merge_pred_label_to_gt, all_param, 64)

