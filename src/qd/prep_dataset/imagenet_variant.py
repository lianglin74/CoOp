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


def create_imagenet_caption_by_name_def():
    version = 'photo_def'
    pattern = 'a photo of a {}; defined as {}'
    dataset = TSVDataset('TaxImageNet2012CapDef')
    labelmap = load_list_file(dataset.get_txt('simple_labelmap'))
    origin_labelmap = dataset.load_labelmap()
    origin_cap_to_idx = {l: i for i, l in enumerate(origin_labelmap)}
    assert len(origin_cap_to_idx) == len(origin_labelmap)
    #split = 'test'
    #for split in ['test', 'train']:
    for split in ['train']:
        def gen_rows():
            for key, str_rects in dataset.iter_data(split, 'caption'):
                rects = json.loads(str_rects)
                for r in rects:
                    new_label = labelmap[origin_cap_to_idx[r['caption']]]
                    r['caption'] = pattern.format(new_label, r['caption'])
                yield key, json_dump(rects)

        dataset.write_data(gen_rows(), split, 'caption', version)
        defs = [k for k, in dataset.iter_data(split, 'captionmap')]
        caption_map = [pattern.format(l, d) for l, d in zip(labelmap, defs)]
        dataset.write_data(
            ((l, ) for l in caption_map),
            split, 'captionmap', version=version
        )

def create_imagenet_caption_by_name():
    version = 'photo_name'
    prefix = 'a photo of a '
    dataset = TSVDataset('TaxImageNet2012CapDef')
    labelmap = load_list_file(dataset.get_txt('simple_labelmap'))
    origin_labelmap = dataset.load_labelmap()
    origin_cap_to_idx = {l: i for i, l in enumerate(origin_labelmap)}
    assert len(origin_cap_to_idx) == len(origin_labelmap)
    #for split in ['train', 'test']:
    for split in ['train']:
        def gen_rows():
            for key, str_rects in dataset.iter_data(split, 'caption'):
                rects = json.loads(str_rects)
                for r in rects:
                    new_label = labelmap[origin_cap_to_idx[r['caption']]]
                    r['caption'] = prefix + new_label
                yield key, json_dump(rects)

        dataset.write_data(gen_rows(), split, 'caption', version)
        caption_map = [prefix + l for l in labelmap]
        dataset.write_data(
            ((l, ) for l in caption_map),
            split, 'captionmap', version=version
        )

def create_imagenet_caption_by_def():
    dataset = TSVDataset('imagenet2012Full')
    out_data = 'TaxImageNet2012CapDef'
    out_dataset = TSVDataset(out_data)

    noffsets = load_list_file(dataset.get_noffsets_file())
    ss = [noffset_to_synset(n) for n in noffsets]
    #names = [get_nick_name(s) for s in ss]
    #assert len(set(names)) == len(names)
    for split in ['train', 'test']:
        # raw image
        num_image = dataset.num_rows(split)
        tsv_writer(
            ((0, i) for i in range(num_image)),
            out_dataset.get_shuffle_file(split)
        )
        write_to_file(
            dataset.get_data(split),
            out_dataset.get_data(split + 'X'))

        # raw label
        def gen_rows():
            for key, r in tqdm(dataset.iter_data(split, 'label')):
                s = ss[int(r)]
                yield key, json_dump([{'class': s.definition()}])
        out_dataset.write_data(gen_rows(), split, 'label')

        # caption file
        def gen_cap_rows():
            for k, s in out_dataset.iter_data(split, 'label'):
                rects = json.loads(s)
                for r in rects:
                    c = r['class']
                    del r['class']
                    r['caption'] = c
                yield k, json_dump(rects)
        out_dataset.write_data(gen_cap_rows(), split, 'caption')

    # labelmap
    write_to_file(
        '\n'.join((s.definition() for s in ss)),
        out_dataset.get_labelmap_file()
    )
    ensure_copy_file(out_dataset.get_labelmap_file(),
                     out_dataset.get_data('test', 'captionmap'))


