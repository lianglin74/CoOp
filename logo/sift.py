import os
import os.path as op
import time
import numpy as np
import re
import sys
import cv2

from numpy import *

import copy
from tqdm import tqdm

import _init_paths

from tagging.utils.averagemeter import AverageMeter

import json
import base64

import logging
from itertools import combinations
import struct

from scripts.tsv_io import tsv_reader, tsv_writer, TSVDataset, TSVFile
from scripts.process_tsv import try_json_parse

import multiprocessing as mp
import pathos.multiprocessing
from scripts.qd_common import init_logging, worth_create, json_dump, img_from_base64, encoded_from_img

from logo.utils import do_sth_through_tsv, decode_array, encode_array, resize_img, point_in_rect, getRectHW, is_convex, is_inside, calc_area
import shutil

from logo.tagging.lib.dataset import int_rect

from skimage.measure import ransac
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
from skimage.transform import ProjectiveTransform

def keypoints_to_array(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response,
                point.octave, point.class_id, descriptors[i])
        i = i + 1
        temp_array.append(temp)
    return temp_array


def array_to_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        # print point
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1],
                                    _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
        # break
    return keypoints, np.array(descriptors)

# sift feature set
def get_sift_feature(image, method='opencv'):

    # TODO: check img type to decide whether convert to gray
    image = image.astype('uint8')

    # logging.info("image shape:{}".format(image.shape))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'opencv':
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(gray, None)

    # debug
    # img = cv2.drawKeypoints(gray2,kp,img2)
    # cv2.imwrite('sift_keypoints.jpg',img)
    return kp, des

def dump_sift_to_nparray(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = [point.pt[0], point.pt[1], point.size, point.angle,
                point.response, point.octave, point.class_id]
        temp_array.extend(temp)
        temp_array.extend(descriptors[i])
        i += 1
    return np.array(temp_array)

def load_nparray_to_sift(nparray):
    keypoints = []
    descriptors = []

    # 7+128
    # print nparray.shape[0]
    total = range(nparray.shape[0])
    for i in total[::135]:
        # print nparray[i:i+7]
        temp_kp = cv2.KeyPoint(x=nparray[i], y=nparray[i+1], _size=nparray[i+2], _angle=nparray[i+3],
                               _response=nparray[i+4], _octave=int(nparray[i+5]), _class_id=int(nparray[i+6]))
        temp_descriptor = nparray[i+7:i+135]

        keypoints.append(temp_kp)
        descriptors.append(temp_descriptor)

    return keypoints, np.array(descriptors)


def dumps_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    # import ipdb; ipdb.set_trace()
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response,
                point.octave, point.class_id, descriptors[i])
        i += 1
        temp_array.append(temp)
    return temp_array

def loads_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        # print point
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1],
                                    _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
        # break
    return keypoints, np.array(descriptors)


def compare_sift(sift_feat1, sift_feat2, rect1_HW, rect2, _debug=False, img1=None, img2=None, resultImgFilename=None):
    # import ipdb; ipdb.set_trace()

    kp1, des1 = sift_feat1
    kp2, des2 = sift_feat2

    # img_temp = cv2.drawKeypoints(img1, kp1, outImage=np.array([]))
    # # cv2.rectangle(img_temp,(int(rect2[0]),int(rect2[1])),(int(rect2[2]),int(rect2[3])),(255,0,0),2)
    # # cv2.putText(img_temp,label['class'], ( max(0,int(rect2[0]-10)), max(0, int(rect2[1]-10))),cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255,0,0),2)
    # cv2.imwrite('sift_keypoints_1.jpg', img_temp)

    # img_temp = cv2.drawKeypoints(img2, kp2, outImage=np.array([]))
    # cv2.rectangle(img_temp,(int(rect2[0]),int(rect2[1])),(int(rect2[2]),int(rect2[3])),(255,0,0),2)
    # # cv2.putText(img_temp,label['class'], ( max(0,int(rect2[0]-10)), max(0, int(rect2[1]-10))),cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255,0,0),2)
    # cv2.imwrite('sift_keypoints_2.jpg', img_temp)
    # import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()
    MIN_MATCH_COUNT = len(kp1)*0.2
    KNN_GOOD_RATIO = 0.8
    RANSAC_GOOD_RATIO = 0.8
    FLANN_INDEX_KDTREE = 0
    KNN_SECOND_RATIO = 0.8

    score = np.Inf

    if len(kp1) > MIN_MATCH_COUNT and len(kp2) > MIN_MATCH_COUNT:
        # import ipdb; ipdb.set_trace()

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # logging.info("len(des1)={},len(des2){}".format(len(des1), len(des2)))
        matches = []
        matchesMask = None

        try:
            # import ipdb; ipdb.set_trace()
            matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
        except Exception as e:
            # import ipdb; ipdb.set_trace()
            logging.info('error:{}'.format(e))

        # import ipdb; ipdb.set_trace()
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < KNN_SECOND_RATIO * n.distance:
                good.append(m)

        matched_index = -1

        # import ipdb; ipdb.set_trace()

        if len(good) > int(MIN_MATCH_COUNT*KNN_GOOD_RATIO):
            # import ipdb; ipdb.set_trace()
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # method 2 
            transform, inliers = ransac(
                    (src_pts, dst_pts),
                    AffineTransform,
                    min_samples=3,
                    residual_threshold=10,
                    max_trials=2000)

            if M is None:
                # print
                # "Can not find valid Homography Transform"
                matchesMask = None
            else:
                lst = [m.distance for m in good]

                # import ipdb; ipdb.set_trace()

                matchesMask = mask.ravel().tolist()

                # logging.info("matchesMask:{}".format(matchesMask))

                if sum(matchesMask) > int(MIN_MATCH_COUNT*RANSAC_GOOD_RATIO):
                    h1, w1 = rect1_HW
                    # logging.info("image1,h:{},w:{}".format(h1,w1))
                    pts = np.float32(
                        [[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)

                    # logging.info("pts.shape:{}".format(pts.shape))

                    h2, w2 = getRectHW(rect2)

                    # logging.info("image2,h:{},w:{}".format(h2,w2))

                    dst = cv2.perspectiveTransform(pts, M)

                    # logging.info("dst:{}".format(dst))
                    # logging.info("pts:{},{};{},{};{},{};{},{}".format(pts[0,0,0], pts[0,0,1], pts[1,0,0], pts[1,0,1],
                    #                             pts[2,0,0], pts[2,0,1], pts[3,0,0], pts[3,0,1]))
                    # logging.info("dst:{},{};{},{};{},{};{},{}".format(dst[0,0,0], dst[0,0,1], dst[1,0,0], dst[1,0,1],
                    #                             dst[2,0,0], dst[2,0,1], dst[3,0,0], dst[3,0,1]))
                    rule1 = is_convex(dst[0, 0, 0], dst[0, 0, 1], dst[1, 0, 0], dst[1, 0, 1],
                                      dst[2, 0, 0], dst[2, 0, 1], dst[3, 0, 0], dst[3, 0, 1])
                    rule2 = (
                        sum([int(is_inside(dst[0, 0, 0], dst[0, 0, 1], w2, h2)) for i in range(4)]) > 0)

                    # area large than 16 pixels
                    rule3 = calc_area(dst[0, 0, 0], dst[0, 0, 1], dst[1, 0, 0], dst[1, 0, 1],
                                      dst[2, 0, 0], dst[2, 0, 1], dst[3, 0, 0], dst[3, 0, 1]) > 16

                    if rule1:
                        if matchesMask is not None:
                            score = sum(lst)
                        if _debug:
                            img2_temp = copy.deepcopy(img2)
                            cv2.rectangle(img2_temp,(int(rect2[0]),int(rect2[1])),(int(rect2[2]),int(rect2[3])),(0,0,255),2)
                            img2_temp = cv2.polylines(img2_temp, [np.int32(
                                dst)], True, (0, 255, 255), 2, cv2.LINE_AA)

                            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                            singlePointColor=(0, 0, 255),
                                            matchesMask=matchesMask,  # draw only inliers
                                            flags=0)

                            img3 = cv2.drawMatches(
                                img1, kp1, img2_temp, kp2, good, None, **draw_params)

                            # logging.info('{} is written!'.format(resultImg))
                            cv2.imwrite(resultImgFilename, img3)
                            # import ipdb; ipdb.set_trace()

    return score

def get_sift_in_rect(kps, des, rect):
    r_kp = []
    r_des = []
    for k, d in zip(kps, des):
        if (point_in_rect(k, rect)):
            r_kp.append(k)
            r_des.append(d)

    return (r_kp, r_des)

def sift_test_sp(filter_idx, f_params):

    test_data = f_params[0]
    split = f_params[1]
    np_feature_set = f_params[2]
    based_on_det = f_params[3]
    _debug = f_params[4]
    train_data = f_params[5]
    train_split = f_params[6]
    top_k = f_params[7]

    dataset = TSVDataset(test_data)
    rows_image = dataset.iter_data(split, progress=True, filter_idx = filter_idx)
    rows_label = dataset.iter_data(split, based_on_det, filter_idx = filter_idx, version = -1)
    # rows_image = dataset.iter_data(test_split, progress=True)
    # rows_label = dataset.iter_data(test_split, based_on_det)
    r = []
    sift = cv2.xfeatures2d.SIFT_create()
    for row_image, row_label in tqdm(zip(rows_image, rows_label)):
        img_key = row_image[0]
        img_key_label = row_label[0]
        img2 = img_from_base64(row_image[-1])

        height, width, _ = img2.shape

        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray, None)
        labels = json.loads(row_label[-1])

        # logging.info("labels:{}".format(labels))
        new_labels = []
        if 'rp' in based_on_det:
            labels = sorted(labels, key=lambda k: k['conf'], reverse=True)
            if top_k > len(labels):
                top_k = len(labels)
        else:
            top_k = len(labels)

        for label in labels[0:top_k]:
            if "rect" in label:
                # logging.info("label:{}".format(label))
                rect2 = label["rect"]

                left, top, right, bot = int_rect(rect2, 2.5)
                left = np.clip(left, 0, width)
                right = np.clip(right, 0, width)
                top = np.clip(top, 0, height)
                bot = np.clip(bot, 0, height)
                if bot <= top or right <= left:
                    # logging.info("skip invalid bbox in {}".format(img_key))
                    continue
                rect2 = [left, top, right, bot]

                objectness = 0
                det_conf = 0

                if based_on_det == "label":
                    objectness = 1
                    det_conf = 1
                else:
                    objectness = label["obj"]
                    det_conf = label["conf"]

                if (rect2[2] - rect2[0]>0) and (rect2[3] - rect2[1]>0) and (((rect2[2] - rect2[0]) * (rect2[3] - rect2[1]))>16):

                    kp_label = []
                    des_label = []
                    #  = get_sift_in_rect(kp2, des2, rect2)
                    # def get_sift_in_rect(kps, des, rect):
                    # r_kp = []
                    # r_des = []
                    for k, d in zip(kp2, des2):
                        if (point_in_rect(k, rect2)):
                            kp_label.append(k)
                            des_label.append(d)

                    if len(kp_label)>10:
                        # import ipdb; ipdb.set_trace()
                        temp_r = []

                        # img_temp = cv2.drawKeypoints(img2, kp_label, outImage=np.array([]))
                        # cv2.rectangle(img_temp,(int(rect2[0]),int(rect2[1])),(int(rect2[2]),int(rect2[3])),(255,0,0),2)
                        # cv2.putText(img_temp,label['class'], ( max(0,int(rect2[0]-10)), max(0, int(rect2[1]-10))),cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255,0,0),2)
                        # cv2.imwrite('sift_keypoints_2.jpg', img_temp)
                        
                        # import ipdb; ipdb.set_trace()
                        # train_feature format: (imgID, (kp, des), train_class, hw, img)

                        for np_train_feature in np_feature_set:
                            
                            id1 = np_train_feature[0]
                            train_class = np_train_feature[1]
                            hw = np_train_feature[2]
                            img1 = np_train_feature[3]
                            kp1, des1 = load_nparray_to_sift(np_train_feature[-1])
                            # import ipdb; ipdb.set_trace()
                            # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                            # logging.info(len(kp1))
                            # logging.info(kp1[0])
                            # img_temp = cv2.drawKeypoints(img1, kp1, outImage=np.array([]))
                            # cv2.imwrite('sift_keypoints_1.jpg',img_temp)
                            # return

                            # img_temp = cv2.drawKeypoints(img2, kp_label, outImage=np.array([]))
                            # cv2.rectangle(img_temp,(int(rect2[0]),int(rect2[1])),(int(rect2[2]),int(rect2[3])),(255,0,0),2)
                            # cv2.putText(img_temp,label['class'], ( max(0,int(rect2[0]-10)), max(0, int(rect2[1]-10))),cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255,0,0),2)
                            # cv2.imwrite('sift_keypoints_2.jpg', img_temp)
                            # import ipdb; ipdb.set_trace()

                            score = compare_sift((kp1, des1), (kp_label, des_label), hw, rect2, _debug, img1, img2, 
                                "data/logo_result_cs/debug/match/{}_{}_{}_{}/{}_{}.jpg".format(train_data, train_split, test_data, split, id1, img_key))

                            # score = compare_sift((kp1, des1), (kp_label, des_label), hw, rect2)
                            temp_r.append((train_class, score))

                        if len(temp_r)>0:
                            from operator import itemgetter
                            temp_r = sorted(temp_r, key=itemgetter(1))

                            # if temp_r[-1][1]<np.inf:
                            #     import ipdb; ipdb.set_trace()

                            if temp_r[0][1] is not np.inf:
                                # import ipdb; ipdb.set_trace()
                                label["class"] = temp_r[0][0]
                                
                                score = temp_r[0][1]
                                # new_score = objectness /(np.exp(score) + 1)
                                # new_score = objectness /(np.exp(score) + 1)
                                # new_score = 1.0 /(np.exp(score) + 1)
                                new_score = -score
                                label["conf"] = -score

                                new_labels.append(label)
                                # breaktemp
        r.append((img_key, json_dump(new_labels), len(labels), len(new_labels)))
    return r

def sift_test_process(train_feature_set, train_data, train_split, region_proposal_file, test_data, 
    test_split, based_on_det, _debug = False, is_mp = True, top_k = 3):
    if _debug:
        debug_path = "data/logo_result_cs/debug/match/{}_{}_{}_{}".format(train_data, train_split, test_data, test_split)
        if op.isdir(debug_path):
            shutil.rmtree(debug_path)
        os.mkdir(debug_path)

    # for train_feature in train_feature_set:
    #     img1 = train_feature[4]
    #     kp1, des1 = train_feature[1]
    #     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #     img_temp = cv2.drawKeypoints(img1, kp1, outImage=np.array([]))
    #     cv2.imwrite('sift_keypoints_1.jpg',img_temp)
    #     break
    #     import ipdb; ipdb.set_trace()
    # train_feature_set[0][1][0][0] = list(map(float, train_feature_set[0][1][0][0].pt))

    # # train_feature format: (imgID, (kp, des), train_class, hw, img)
    # np_feature_set = []
    # for train_feature in train_feature_set:
    #     imgID = train_feature[0]
    #     kp, des = train_feature[1]
    #     train_class = train_feature[2]
    #     hw = train_feature[3]
    #     img = train_feature[4]
    #     feature = dump_sift_to_list(kp, des)
    #     np_feature_set.append((imgID, np.array(feature), train_class, hw, img))

        # nparray = decode_array(train_feature[1])
        # kp, des = load_list_to_sift(nparray)
        # r.append((train_feature[0], (kp, des), train_feature[2], train_feature[3], train_feature[4]))

    outFilename = op.join('data', test_data, "{}.{}.{}.tsv".format(test_split, based_on_det, 'sift'))

    # if not worth_create(region_proposal_file, outFilename):
    #     return outFilename

    last_tic = time.time()

    result = do_sth_through_tsv(test_data, test_split, sift_test_sp, [train_feature_set, based_on_det, _debug, train_data, train_split, top_k], is_mp)
    # dataset = TSVDataset(test_data)
    # num_images = dataset.num_rows(test_split)
    # params = [test_data, test_split, train_feature_set, based_on_det, _debug]
    # result = sift_test_sp(range(num_images), params)

    sum_rp = 0
    sum_result = 0
    for r in result:
        sum_rp += r[2]
        sum_result += r[3]
    
    logging.info("Total get {} result from {} region proposals.".format(sum_result, sum_rp))

    logging.info("Elapsed time for {}:{} seconds".format(
        sys._getframe().f_code.co_name, time.time() - last_tic))

    # import ipdb; ipdb.set_trace()
    dataset = TSVDataset(test_data)
    dataset.write_data(result, test_split, '{}.sift'.format(based_on_det))

    return outFilename

def sift_load_process(data, split, is_mp = True, _debug = False, ):
    last_tic = time.time()

    dataset = TSVDataset(data)
    results = []
    if dataset.has(split, 'sift'):
        results = do_sth_through_tsv(data, split, sift_load_sp, [], is_mp)
        logging.info("Elapsed time for {}:{}".format(sys._getframe().f_code.co_name, time.time() - last_tic))
    else:
        logging.info("sift feature file doesn't exist!")

    return results

# def sift_index_extract_sp(data, split, _debug=False):
#     last_tic = time.time()

#     dataset = TSVDataset(data)

#     rows_image = dataset.iter_data(split, progress=True)
#     rows_label = dataset.iter_data(split, 'label')

#     r = []
#     feature = []

#     sift = cv2.xfeatures2d.SIFT_create()
#     for row_image, row_label in tqdm(zip(rows_image, rows_label)):
#         img_key = row_image[0]
#         img_key_dup = row_label[0]

#         # assert(img_key, img_key_dup)

#         label_str = row_label[-1]
#         labels = try_json_parse(label_str)
#         im = img_from_base64(row_image[-1])

#         im = im.astype('uint8')
#         # logging.info("image shape:{}".format(image.shape))
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#         kp, des = sift.detectAndCompute(gray, None)

#         for label in labels:
#             if 'rect' in label:
#                 rect = label['rect']

#                 if 'sift' not in label:
#                     label['sift'] = []

#                 for kpIdx, point in enumerate(kp):
#                     if point_in_rect(point.pt, rect):
#                         label['sift'].append(kpIdx)

#         # feature = dump_sift_to_list(kp, des)
#         r.append((img_key, json_dump(labels)))
#         feature.append((img_key, json_dump(labels), (kp, des)))

#         if _debug:
#             img = cv2.drawKeypoints(im, kp, outImage=np.array([]))
#             cv2.imwrite('sift_keypoints_orig.jpg', img)

#     dataset.write_data(r, split, 'siftIndex')

#     logging.info("Elapsed time for {}:{}".format(
#         "extract sift and index by BBox", time.time() - last_tic))
#     return feature

def sift_load_sp(filter_idx, f_params):
    data = f_params[0]
    split = f_params[1]

    dataset = TSVDataset(data)
    rows = dataset.iter_data(split=split, t='sift', filter_idx = filter_idx)
    r = []
    for row in rows:
        nparray = decode_array(row[-1])
        im = img_from_base64(row[3])
        r.append((row[0], row[1], json.loads(row[2]), im, nparray))
    return r

def sift_extract_sp(filter_idx, f_params):

    data = f_params[0]
    split = f_params[1]
    _debug = f_params[2]
    
    fix_max_edge = -1
    fix_max_edge = f_params[3]

    # logging.info("_debug:{}".format(_debug))

    dataset = TSVDataset(data)
    rows_image = dataset.iter_data(split, progress=True, filter_idx = filter_idx)
    rows_label = dataset.iter_data(split, 'label', filter_idx = filter_idx, version = -1)

    r = []
    sift = cv2.xfeatures2d.SIFT_create()
    for row_image, row_label in zip(rows_image, rows_label):
        imgKey = row_image[0]
        img = img_from_base64(row_image[-1])
        img = resize_img(img, fix_max_edge)
        hw = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        labels = json.loads(row_label[-1])

        if _debug:
            img_temp = cv2.drawKeypoints(img, kp, outImage=np.array([]))
            _debug_folder = "data/logo_result_cs/debug"
            fname = op.join(_debug_folder, "{}/{}/{}.jpg".format(data, split, imgKey))
            # logging.info(fname)
            cv2.imwrite(fname, img_temp)

        train_class = ""
        if (len(labels)>0):
            train_class = labels[0]['class']

        feature = dump_sift_to_nparray(kp, des)
        r.append((imgKey, train_class, hw, img, feature))
    return r

def sift_extract_process(data, split, fix_max_edge = -1, _debug=False, is_mp = True, is_force_extract_feature=True):

    if _debug:
        debug_path = "data/logo_result_cs/debug/{}".format(data)
        if not op.isdir(debug_path):
            os.mkdir(debug_path)

        debug_path = "data/logo_result_cs/debug/{}/{}".format(data, split)
        if op.isdir(debug_path):
            shutil.rmtree(debug_path)
        os.mkdir(debug_path)

    last_tic = time.time()

    dataset = TSVDataset(data)
    results = []
    if not dataset.has(split, 'sift') or is_force_extract_feature:
        results = do_sth_through_tsv(data, split, sift_extract_sp, [_debug, fix_max_edge], is_mp)
        disk_r = []
        for r in results:
            disk_r.append((r[0], r[1], json_dump(r[2]), encoded_from_img(r[3]), encode_array(r[4])))
        dataset.write_data(disk_r, split, 'sift')
    else:
        results = do_sth_through_tsv(data, split, sift_load_sp, [], is_mp)

    # i = 0
    # for r in results:
    #     kp, des = load_nparray_to_sift(r[-1])
    #     img_temp = cv2.drawKeypoints(r[3], kp, outImage=np.array([]))
    #     cv2.imwrite("temp2/{:06}.jpg".format(i), img_temp)
    #     i += 1

    logging.info("Elapsed time for {}:{}".format(
        sys._getframe().f_code.co_name, time.time() - last_tic))

    return results

def test_nparray_dump_load():
    feature = [
        [(16.161951065063477, 90.72784423828125), 14.192916870117188, 129.22503662109375, 0.023665310814976692, 15860481, -1,
            np.array([50., 134.,  17.,   0.,   0.,   0.,   1.,   3., 134., 134.,   2.,
                      0.,   0.,   0.,   0.,   0.,  26., 134.,   9.,   0.,   7.,  39.,
                      1.,   1.,   5.,  22.,   7.,   2.,  34., 134.,  16.,   4.,  69.,
                      64.,   1.,   0.,   0.,  12., 134.,  53., 134.,  96.,   3.,   0.,
                      0.,   3.,  18.,  45.,  56., 104.,  44.,  17.,  11.,   1.,   1.,
                      8.,   1.,  70.,  48.,   8.,  23.,  25.,   0.,   0.,   5.,   1.,
                      3.,  54.,  28.,  49., 134.,  14.,  82., 102.,  77.,  37.,   7.,
                      10.,  20.,  26.,  12.,  75.,  55.,  14.,   7.,   0.,   0.,   1.,
                      0.,   5.,   4.,   2.,   2.,   0.,   0.,   0.,   0.,   0.,   2.,
                      131.,  49.,   1.,   0.,   0.,   0.,  11.,  27.,  49.,   5.,   0.,
                      0.,   0.,   0.,   8.,   5.,   0.,   0.,   0.,   0.,   0.,   0.,
                      0.,   0.,   0.,   0.,   0.,   0.,   0.])
         ]
    ]

    nparray1 = []
    for f in feature:
        nparray1.extend(f[0][:])
        nparray1.extend(f[1:5])
        nparray1.extend(f[6])

    nparray1 = np.array(nparray1)
    logging.info("{}:{}".format("nparray1", nparray1))

    # b64str = encode_array(nparray1)
    # nparray2 = decode_array(b64str)

    logging.info("{}:{}".format(" nparray1.shape",  nparray1.shape))
    shapestr = ",".join([str(x) for x in nparray1.shape])
    array_binary = nparray1.tobytes()
    # b64str =  base64.b64encode(array_binary).decode()
    b64str = base64.b64encode(array_binary)
    bufferstr = ";".join([shapestr, b64str])

    (shapestr, b64str) = [x.strip() for x in bufferstr.split(";")]
    logging.info("{}:{}".format(" shapestr",  shapestr))
    arrayshape = [int(x) for x in shapestr.split(",")]
    array_binary = base64.b64decode(b64str)
    # nparray = np.fromstring(array_binary, dtype=np.dtype('float64'))
    nparray = np.fromstring(array_binary, dtype=np.dtype('float64'))
    nparray2 = nparray.reshape(arrayshape)

    logging.info("feature:{}".format(feature))
    logging.info("arr1:{}".format(nparray1[-20:]))
    logging.info("arr2:{}".format(nparray2[-20:]))

def test():
    data = "logo40"
    split = "test"

    sift_extract_sp(data, split)

    # sift_extract_mp(data, split)
    # logo40_test = sift_load_mp(data, split)
    # # import ipdb; ipdb.set_trace()

    # sift_number = AverageMeter()
    # for feature in logo40_test:
    #     sift_number.update(len(feature[1][0]))

    # dataset = TSVDataset(data)
    # for imKey, _, imStr in dataset.iter_data(split):
    #     im = img_from_base64(imStr)
    #     assert(imKey == logo40_test[0][0])

    #     img = cv2.drawKeypoints(im, logo40_test[0][1][0], outImage=np.array([]))
    #     cv2.imwrite('sift_keypoints.jpg',img)
    #     break

    # sift_extract('brand1048Clean','train', '/raid/data/logo_result_cs/brand1048Clean_train_sift.tsv')
    # sift_extract('brand1048Clean','test', '/raid/data/logo_result_cs/brand1048Clean_test_sift.tsv')

if __name__ == "__main__":
    init_logging()
    test()
