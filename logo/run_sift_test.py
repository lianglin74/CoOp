import os
import os.path as op
import time
import numpy as np
import re

import cv2

from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import pickle
from pyflann import *
from numpy import *

import cPickle
import copy
from tqdm import tqdm

import _init_paths
from logo.tagging.lib.dataset import TSVDataset
from logo.tagging.utils.averagemeter import AverageMeter

from logo.utils import check_dir, crop_tsv, get_feature_set, dumps_feature_set, loads_feature_set, dumps_simple, loads_simple, sift_matching_opencv

import logging
from scripts.qd_common import init_logging, worth_create
# import ipdb
import glob

from scripts import tsv_io
from scripts.yolotrain import yolo_predict

def read_sift_features(db_name, db_split, feature_set_cache_folder, feature_set_cache_filename, is_saving_images = False, enlarge_factor = 1.0, fix_max_edge = -1):

    if not op.isfile(feature_set_cache_filename):

        logging.info('read tsv and crop image from {}'.format(feature_set_cache_filename))
        image_set = crop_tsv(db_name, db_split, feature_set_cache_folder, is_saving_images, enlarge_factor, fix_max_edge)

        logging.info('get sift feature_set')
        feature_set = get_feature_set(image_set)

        logging.info('dump feature_set to file:{}'.format(feature_set_cache_filename))
        dumps_feature_set(feature_set_cache_filename, feature_set)

    else:
        logging.info('load feature_set to file:{}'.format(feature_set_cache_filename))
        feature_set = loads_feature_set(feature_set_cache_filename)
    
    return feature_set

def test_sift(train_name, train_set, test_name, test_set, result_folder, cache_folder, train_feature_set_cache_filename, test_feature_set_cache_filename, is_forth_test=False):

    result_filename = op.join(result_folder, "{}_{}_result.json".format(train_name, test_name))
    if worth_create(train_feature_set_cache_filename, result_filename) or worth_create(test_feature_set_cache_filename, result_filename) or is_forth_test:
        logging.info('Testing... result will dump to {}'.format(result_filename))
        # feature_set format:
        #   result['id_idx'][bbox_id] = image_id
        #   result['rect_idx'][bbox_id] = rect
        #   result['class_idx'][bbox_id] = class_name
        #   result['data'][bbox_id] = sift
        #   result['images'][bbox_id] = image
        #   result['class_map'][class_name].append(bbox_id)

        # Test
        result = {}
        test_result_folder = op.join(cache_folder, "{}_{}".format(train_name, test_name))
        
        if not op.isdir(test_result_folder):
            os.mkdir(test_result_folder)
        else:
            files = glob.glob(op.join(test_result_folder, '*.jpg'))
            for f in files:
                os.remove(f)
        # cnt = 0        

        sorted_test_keys = sorted(test_set['data'].iterkeys())

        for key_test in tqdm(sorted_test_keys):
            # logging.info("Testing Key={}".format(key_test))
            result_per_class = []
            for key_class in train_set['class_map']:
                result_per_image_in_class = [] # [(class, conf), (class, conf), ...]
                key_class_correct = key_class
                for key_train in train_set['class_map'][key_class]:
                    # logging.info("key_train={}".format(key_train))
                    # import ipdb; ipdb.set_trace()
                    
                    matching_result = sift_matching_opencv(train_set['data'][key_train], test_set['data'][key_test], 
                        train_set['images'][key_train], test_set['images'][key_test], op.join(test_result_folder, "{}_{}.jpg".format(key_test, key_train)))

                    # TODO: matching based on vlfeat
                    if matching_result < np.Inf:
                        result_per_image_in_class.append(matching_result)

                # logging.info("match result:{}".format(matching_result))

                if len(result_per_image_in_class)>0:
                    result_per_class.append((max(result_per_image_in_class), key_class_correct))
            if len(result_per_class) > 0:
                result[key_test]= sorted(result_per_class)
        dumps_simple(result_filename, result)
    else:
        logging.info('Result is exist! Result load from {}.'.format(result_filename))
        result = loads_simple(result_filename)

    return result

def main():
    
    # import ipdb; ipdb.set_trace = lambda: None

    # Global variables
    cache_folder = "/raid/data/logo_result_cs/cache"
    result_folder = "/raid/data/logo_result_cs"

    det_expid = "TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only"

    init_logging()
    
    is_forth_test = True

    check_dir(cache_folder)

    # sift parameter
    # compara parameter
    train_db_name = "logo40"
    train_db_split = "train"
    train_name = "{}_{}".format(train_db_name, train_db_split)

    test_db_name = "logo40"
    # test_db_split = "testsmall"
    test_db_split= "test"
    test_name = "{}_{}".format(test_db_name, test_db_split)

    pred_file, _ = yolo_predict(full_expid=det_expid, test_data=test_db_name, test_split=test_db_split)
    logging.info("\npred_file:{}".format(pred_file))

    train_feature_set_cache_filename = op.join(cache_folder, train_name +'.feature_set')
    train_feature_set_cache_folder = op.join(cache_folder, train_name)
    train_set = read_sift_features(train_db_name, train_db_split, train_feature_set_cache_folder, train_feature_set_cache_filename, True, 1, 224)

    test_feature_set_cache_filename = op.join(cache_folder, test_name +'.feature_set')
    test_feature_set_cache_folder = op.join(cache_folder, test_name)
    test_set = read_sift_features(test_db_name, test_db_split, test_feature_set_cache_folder, test_feature_set_cache_filename, True, 1)

    result = test_sift(train_name, train_set, test_name, test_set, result_folder, cache_folder, train_feature_set_cache_filename, test_feature_set_cache_filename, is_forth_test)

    test_feature_set_cache_filename = op.join(cache_folder, test_name +'.feature_set')
    test_feature_set_cache_folder = op.join(cache_folder, test_name)
    test_set = read_sift_features(test_db_name, test_db_split, test_feature_set_cache_folder, test_feature_set_cache_filename, True, 1)

    result = test_sift(train_name, train_set, test_name, test_set, result_folder, cache_folder, train_feature_set_cache_filename, test_feature_set_cache_filename, is_forth_test)

    # import ipdb; ipdb.set_trace()
    total_number = len(test_set['data'])
    total_result_count = len(result)
    top1_correct_count = 0
    # logging.info("\nlen(result):{}".format(len(result)))
    # logging.info("\ntest_set['class_idx']:\n{}".format(test_set['class_idx']))
    
    class_idx = test_set['class_idx']

    for key in sorted(result.iterkeys()):
    #     logging.info("\nkey:\n{}".format(key))
    #     logging.info("\nclass_idx:\n{}".format(class_idx[key]))
        
        class_result = result[key][0][1]

        class_gt = class_idx[key]

        if class_result in class_gt:
            top1_correct_count += 1
        else:
            logging.info("key:{}".format(key))
            logging.info("{},{}".format(class_result, class_gt))
    
    logging.info("\nTotal result count={}, Top1_correct_count={}".format(total_result_count, top1_correct_count))

    top1_Acc = float(top1_correct_count)/float(total_result_count)
    top1_Recall = float(top1_correct_count) / float(total_number)

    logging.info("\nTop1_Acc={}, Top1_Recall={}".format(top1_Acc, top1_Recall))
    # TODO:
    # log result and parameter


if __name__ == "__main__":
    main()
