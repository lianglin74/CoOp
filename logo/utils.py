import os
import os.path as op
import time
import numpy as np
import re

import cv2

from numpy import *

import cPickle
import copy
from tqdm import tqdm

import _init_paths

from logo.tagging.utils.averagemeter import AverageMeter

import json
import base64

import logging
from itertools import combinations
import struct

from scripts.tsv_io import tsv_reader, tsv_writer, TSVDataset
from scripts.process_tsv import try_json_parse

# 
# IO
# 
def get_dir_files_by_type(folder_name, file_type='.jpg'):
    images = []
    for root, dirs, files in os.walk(folder_name):
        for f in files:
            fullpath = op.join(root, f)
            if os.path.splitext(fullpath)[1] == file_type:
                images.append(fullpath)
    return images

def get_file_basename(full_filename):
    return op.splitext(op.basename(full_filename))[0]

def create_matching_list(src_dir, dest_dir, result_dir, file_ext='jpg'):
    # TODO: check dirs existing
    left_images = get_dir_files_by_type(src_dir)
    right_images = get_dir_files_by_type(dest_dir)

    return_list = []
    for i, left_image in enumerate(left_images):
        for j, right_image in enumerate(right_images):
            left_name = op.splitext(op.basename(left_image))[0]
            right_name = op.splitext(op.basename(right_image))[0]

            return_list.append((left_image, right_image,
                op.join(result_dir, "{0}_{1}.{2}".format(left_name,right_name, file_ext))))

    return return_list

# 
# string
# 
def get_file_ext(filename, with_dot=False):
    result = ''
    if with_dot:
        file_suffix = ".*(\..*)"
        result = re.search(file_suffix, filename)
        result.group(1)
        result = op.splitext(op.basename(filename))[0]
    else:
        if len(result)>0:
            result = result[1:-1]
    return result


# parse tsv

def read_map(filename, split='\t', regstr = "*"):
    # TODO: add regstring supporting
    mapping = {}
    if filename != None:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cols = [x.strip() for x in line.split('\t')]
                key = cols[0]
                values = [x.strip() for x in cols[1].split('_')]

                if values[3].lower() == 'text':
                    mapping[key] = values[2] + "-" + "text"
                else:
                    mapping[key] = values[2]
    return mapping

# 
# print 3rd party class
# 
def print_keypoint(point):
    logging.info('pt:{},size:{}, angle:{}, reponse:{}, octave:{}, class_id:{}'.format(point.pt, point.size, point.angle, point.response, point.octave, point.class_id))

# =============================================================================
# 
# checking & confirmation
# 
# =============================================================================
def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 
# is what
# 

def is_convex(x1, y1, x2, y2, x3, y3, x4, y4):
    s1 = map(np.array, [(x1, y1), (x3, y3)])
    s2 = map(np.array, [(x2, y2), (x4, y4)])
    if intersection(s1, s2) is None:
        return False
    else:
        return True

def intersection(s1, s2):
    """
    Return the intersection point of line segments `s1` and `s2`, or
    None if they do not intersect.
    """
    p, r = s1[0], s1[1] - s1[0]
    q, s = s2[0], s2[1] - s2[0]
    rxs = float(np.cross(r, s))
    if rxs == 0: return None
    t = np.cross(q - p, s) / rxs
    u = np.cross(q - p, r) / rxs
    if 0 < t < 1 and 0 < u < 1:
        return p + t * r
    return None

def convex_quadrilaterals(points):
    """
    Generate the convex quadrilaterals among `points`.
    """
    segments = combinations(points, 2)
    # import ipdb; ipdb.set_trace()
    for s1, s2 in combinations(segments, 2):
        if intersection(s1, s2) != None:
            yield s1, s2

def is_quadrilateral_convex(x1, y1, x2, y2, x3, y3, x4, y4):
    z1 = ((x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1))
    z2 = ((x4 - x1) * (y3 - y1) - (x3 - x1) * (y4 - y1))
    z3 = ((x4 - x2) * (y3 - y2) - (x3 - x2) * (y4 - y2))
    z4 = ((x3 - x2) * (y1 - y2) - (x1 - x2) * (y3 - y2))
    return (z1 * z2 > 0) and (z3 * z4 > 0)

def dist(x1, y1, x2, y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def calc_area(x1, y1, x2, y2, x3, y3, x4, y4):
    d12 = dist(y1, x1, y2, x2) 
    d23 = dist(y2, x2, y3, x3) 
    d34 = dist(y3, x3, y4, x4) 
    d41 = dist(y4, x4, y1, x1) 
    d24 = dist(y2, x2, y4, x4) 
    k1 = (d12+d41+d24)/2
    k2 = (d23+d34+d24)/2
    ss = (k1*(k1-d12)*(k1-d41)*(k1-d24))
    
    if ss>0:
        s1 = ss**0.5
    else:
        s1 = -1

    ss = (k2*(k2-d23)*(k2-d34)*(k2-d24))
    if ss>0:
        s2 = ss**0.5
    else:
        s2 = -1

    if s1<0 or s2<0:
        s = -1
    else:
        s = s1+s2

    return s 

def is_inside(x, y, w, h):
    return x>=0 and x<w-1 and y>=0 and y<h-1
# 
# transform
# 

def keypoints_to_array(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[i])
        i = i + 1
        temp_array.append(temp)
    return temp_array

def array_to_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        # print point
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
        # break
    return keypoints, np.array(descriptors)

# 
# image processing
# 
def int_rect(rect, enlarge_factor=1.0):
    # import ipdb; ipdb.set_trace()
    left, top, right, bot = rect
    w = (right - left) * enlarge_factor / 2.0
    h = (bot - top) * enlarge_factor / 2.0
    cx = (left+right)/2.0
    cy = (top+bot)/2.0
    left = cx - w
    right = cx + w
    top = cy - h
    bot = cy + h
    return int(np.floor(left)), int(np.floor(top)), int(np.ceil(right)), int(np.ceil(bot))

def imcrop(img, bbox, enlarge_factor = 1.0):
    x1,y1,x2,y2 = int_rect(bbox, enlarge_factor)

    # if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
    #     img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    
    if x1 < 0: x1=0
    if y1 < 0: y1=0
    if x2 > img.shape[1]-1: x2 = img.shape[1] - 1
    if y2 > img.shape[0]-1: y2 = img.shape[0] - 1

    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")

    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    
    return img, intfloor(x1), intfloor(x2), intceil(y1), intceil(y2)

def getFirstItem(item):
    return item[0]
def getSecondItem(item):
    return item[1]

def sift_matching_vlfeat(sift_feat1, sift_feat2, img1, img2, resultImg):

    # import ipdb; ipdb.set_trace()

    kp1, des1 = sift_feat1
    kp2, des2 = sift_feat2

    MIN_MATCH_COUNT = 10
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # logging.info("len(des1)={},len(des2){}".format(len(des1), len(des2)))
    matches = []

    if len(kp1)>=MIN_MATCH_COUNT and len(kp2)>=MIN_MATCH_COUNT:
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            logging.info('error:{}'.format(e))
            
        # import ipdb; ipdb.set_trace()
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        matched_index = -1

        # import ipdb; ipdb.set_trace()

        if len(good) > len(kp1)*0.5:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                score = np.Inf
                # print
                # "Can not find valid Homography Transform"
                matchesMask = None
            else:
                lst = [m.distance for m in good]
                score = sum(lst)/len(lst)

                # import ipdb; ipdb.set_trace()

                matchesMask = mask.ravel().tolist()
                h,w,_ = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

                dst = cv2.perspectiveTransform(pts,M)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

                img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
                
                if matchesMask is not None:
                    # logging.info('{} is written!'.format(resultImg))
                    cv2.imwrite(resultImg, img3)
        else:
            # print
            # "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            # matchesMask = None
            score = np.Inf
    else:
        score = np.Inf
    return score

# here kps = ([kp], [description])
def sift_matching_opencv(sift_feat1, sift_feat2, img1, img2, resultImg):

    # import ipdb; ipdb.set_trace()

    kp1, des1 = sift_feat1
    kp2, des2 = sift_feat2

    MIN_MATCH_COUNT = 20
    FLANN_INDEX_KDTREE = 0
    score = np.Inf

    if len(kp1) > MIN_MATCH_COUNT and len(kp2) > MIN_MATCH_COUNT:

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # logging.info("len(des1)={},len(des2){}".format(len(des1), len(des2)))
        matches = []
        matchesMask = None

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            logging.info('error:{}'.format(e))
            
        # import ipdb; ipdb.set_trace()
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        matched_index = -1

        # import ipdb; ipdb.set_trace()
        

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                # print
                # "Can not find valid Homography Transform"
                matchesMask = None
            else:
                lst = [m.distance for m in good]

                # import ipdb; ipdb.set_trace()

                matchesMask = mask.ravel().tolist()
                
                # logging.info("matchesMask:{}".format(matchesMask))

                if sum(matchesMask) > int(MIN_MATCH_COUNT):
                    h1,w1,_ = img1.shape
                    # logging.info("image1,h:{},w:{}".format(h1,w1))
                    pts = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
                    
                    # logging.info("pts.shape:{}".format(pts.shape))

                    h2,w2,_ = img2.shape
                    # logging.info("image2,h:{},w:{}".format(h2,w2))

                    dst = cv2.perspectiveTransform(pts,M)

                    # logging.info("dst:{}".format(dst))
                    # logging.info("pts:{},{};{},{};{},{};{},{}".format(pts[0,0,0], pts[0,0,1], pts[1,0,0], pts[1,0,1],
                    #                             pts[2,0,0], pts[2,0,1], pts[3,0,0], pts[3,0,1]))
                    # logging.info("dst:{},{};{},{};{},{};{},{}".format(dst[0,0,0], dst[0,0,1], dst[1,0,0], dst[1,0,1],
                    #                             dst[2,0,0], dst[2,0,1], dst[3,0,0], dst[3,0,1]))
                    rule1 = is_convex(dst[0,0,0], dst[0,0,1], dst[1,0,0], dst[1,0,1], dst[2,0,0], dst[2,0,1], dst[3,0,0], dst[3,0,1])
                    rule2 = (sum([int(is_inside(dst[0,0,0], dst[0,0,1], w2, h2)) for i in range(4)]) > 0)

                    # area large than 64 pixels
                    rule3 = calc_area(dst[0,0,0], dst[0,0,1], dst[1,0,0], dst[1,0,1], dst[2,0,0], dst[2,0,1], dst[3,0,0], dst[3,0,1])>16

                    if rule1:
                        img2_temp = copy.deepcopy(img2)
                        img2_temp = cv2.polylines(img2_temp,[np.int32(dst)],True,(0, 255, 255),2, cv2.LINE_AA)

                        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = (0, 0, 255),
                            matchesMask = matchesMask, # draw only inliers
                            flags = 0)

                        img3 = cv2.drawMatches(img1,kp1,img2_temp,kp2,good,None,**draw_params)
                        
                        if matchesMask is not None:
                            # logging.info('{} is written!'.format(resultImg))
                            cv2.imwrite(resultImg, img3)
                            score = sum(lst)/len(lst)

    return score

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.fromstring(jpgbytestring, np.uint8)
    try:
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return None

def parse_line_tsv(line):
    cols = [x.strip() for x in line.split('\t')]
    image_id = cols[0]
    labels = json.loads(cols[1])
    image = img_from_base64(cols[-1])
    return image_id, labels, image

def crop_tsv(data, split, folder_path, save_image=False, enlarge_factor = 1.0, fit_max_edge = -1):
    logging.info("data, split:{}, {}".format(data, split))
    dataset = TSVDataset(data)
    
    result = {}
    result['id_idx'] = {}
    result['rect_idx'] = {}
    result['images'] = {}
    result['data'] = {}

    result['class_idx'] = {}
    result['class_map'] = {} #{class_name, [image_id]}
    
    # result['rect_crop_map'] = {}

    # TODO: multiple processing
    # with open(tsv_file_name,'r') as tsv_f:
    #     lines = tsv_f.readlines()

    if save_image:
        if not op.isdir(folder_path):
            os.mkdir(folder_path)

    #     flist = open('{0}/filelist.tsv'.format(folder_path), 'w')

    bbox_id = 0

    # rows = dataset.iter_data(split, progress=True, version=-1)
    rows_image = dataset.iter_data(split)
    rows_label = dataset.iter_data(split, 'label', version=-1)

    for row_image, row_label in zip(rows_image, rows_label):
        # image_id, labels, image = parse_line_tsv(line)
        imgkey = row_label[0]
        label_str = row_label[-1]
        img_str = row_image[-1]

        labels = try_json_parse(label_str)
        
        im = img_from_base64(img_str)

        for label in labels:
            rect = label['rect']
            class_name = label['class']

            if class_name not in result['class_map']:
                result['class_map'][class_name] = []

            # import ipdb; ipdb.set_trace()
            new_data = imcrop(im, rect, enlarge_factor)

            if fit_max_edge>0:
                height, width = new_data.shape[:2]
                max_height = fit_max_edge
                max_width = fit_max_edge

                # only shrink if img is bigger than required
                if max_height < height or max_width < width:
                    # get scaling factor
                    scaling_factor = max_height / float(height)
                    if max_width/float(width) < scaling_factor:
                        scaling_factor = max_width / float(width)
                    # resize image
                    new_data = cv2.resize(new_data, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

            result['id_idx'][bbox_id] = imgkey
            result['rect_idx'][bbox_id] = rect
            result['class_idx'][bbox_id] = class_name
            result['images'][bbox_id] = new_data
            result['class_map'][class_name].append(bbox_id)
            
            if save_image:
                filename = "{:08}".format(bbox_id)
                # flist.write("{}\t{}\t{}\n".format(image_id, filename, class_name))
                image_filename = '{0}/{1}'.format(folder_path, filename + '.jpg')
                cv2.imwrite(image_filename, new_data)
            bbox_id += 1

    return result

# sift
def get_sift_feature(image, method='opencv'):
    
    # TODO: check img type to decide whether convert to gray
    # image = image.astype('uint8')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'opencv':
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(gray, None)

    # debug
    # img = cv2.drawKeypoints(gray2,kp,img2)
    # cv2.imwrite('sift_keypoints.jpg',img)
    return kp, des

def get_feature_set(image_set, method='opencv'):
    
    feature_set = copy.deepcopy(image_set)
    # print image_set['data']
    for k, v in tqdm(image_set['images'].items()):
        feature_set['data'][k] = get_sift_feature(v, method)

    return feature_set

def loads_feature_set(filename):
    feat_set = cPickle.loads(open(filename).read())

    for key in tqdm(feat_set['data']):
        value = feat_set['data'][key]
        
        temp = loads_keypoints(value)
        feat_set['data'][key] = temp

    return feat_set

def dumps_feature_set(filename, feature_set):
    result = False

    feat_set_dump_copy = copy.deepcopy(feature_set)

    for key in tqdm(feature_set['data']):
        
        value = feature_set['data'][key]

        # import ipdb; ipdb.set_trace()

        temp = dumps_keypoints(value[0], value[1])
        feat_set_dump_copy['data'][key] = temp
    logging
    with open(filename, 'wb') as fout:
        fout.write(cPickle.dumps(feat_set_dump_copy))
        result = True

    return result

def dumps_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    # import ipdb; ipdb.set_trace()
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[i])
        i = i + 1
        temp_array.append(temp)
    return temp_array

def loads_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        # print point
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
        # break
    return keypoints, np.array(descriptors)

def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x

def loads_simple(filename):
    data = json.loads( open(filename, 'r').read(), object_hook=jsonKeys2int)
    return data

def dumps_simple(filename, data):
    json.dump(data, open(filename, 'w'))
    return

def intfloor(x):
    return int(np.floor(x))

def intceil(x):
    return int(np.ceil(x))

# 
# for vlfeat
# 
# 

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def SerializeFloatArray(array, fp):
    for row in range(array.shape[0]):
        buffer = struct.pack('f' * array.shape[1], *array[row, :])
        fp.write(buffer)

def DeSerializeFloatArray(fp, cols):
    buffer = fp.read()
    elements_count = len(buffer) / 4
    array = struct.unpack(str(elements_count) + "f", buffer)
    rows = elements_count / cols
    array = np.reshape(array, (rows, cols))
    return array

def ComputeElementsCount(file, element_size):
    with open(file, 'rb') as fp:
        buffer = fp.read()
        elements_count = len(buffer) / element_size
    return elements_count

def DeSerializeFloatArrayByRow(fp, rows, cols):
    array = np.zeros([rows, cols], dtype=float)
    row_size = cols * 4
    for row in range(rows):
        bytes_read = fp.read(row_size)
        array[row, :] = struct.unpack(str(cols) + "f", bytes_read)

    return array

def SerializeIntArray(array, fp):
    for row in range(array.shape[0]):
        buffer = struct.pack('i' * array.shape[1], *array[row, :])
        fp.write(buffer)

def DeSerializeIntArray(fp, cols):
    buffer = fp.read()
    elements_count = len(buffer) / 4
    array = struct.unpack(str(elements_count) + "i", buffer)
    rows = elements_count / cols
    array = np.reshape(array, (rows, cols))
    return array

def output_knn(test_file_path, top_UPC, top_score, raw_data_folder, result_folder, knn_list):
    test_image = cv2.imread(test_file_path)

    combined_image_height = 224;

    combined_image_width = int(float(test_image.shape[1]) * float(combined_image_height) / float(test_image.shape[0]))
    test_image = cv2.resize(test_image, (combined_image_width, combined_image_height), interpolation=cv2.INTER_CUBIC)

    knn_count = min(10, len(top_UPC))

    estimation_example_images = []
    for index in range(knn_count):
        example_image_file = os.path.join(raw_data_folder, top_UPC[index] + '.jpg')
        example_image = cv2.imread(example_image_file)
        example_image_width = (int)(float(example_image.shape[1]) * float(combined_image_height) / float(example_image.shape[0]))
        example_image = cv2.resize(example_image, (example_image_width, combined_image_height), interpolation=cv2.INTER_CUBIC)
        estimation_example_images.append(example_image)
        combined_image_width += example_image.shape[1]

    combined_image = np.zeros((combined_image_height, combined_image_width, 3), np.uint8)
    combined_image[0:test_image.shape[0], 0:test_image.shape[1], :] = test_image
    start_x = test_image.shape[1]
    for index in range(knn_count):
        end_x = start_x + estimation_example_images[index].shape[1]
        combined_image[0:estimation_example_images[index].shape[0], start_x:end_x,:] = estimation_example_images[index]
        start_x = end_x

    file_name = os.path.splitext(os.path.basename(test_file_path))[0]
    result_image_file = '{}/{}.jpg'.format(result_folder, file_name)
    cv2.imwrite(result_image_file, combined_image)

    result_scores_file = '{}/{}.txt'.format(result_folder, file_name)
    with open(result_scores_file, 'w') as scores_output_file:
        for upc in top_UPC:
            scores_output_file.write(upc + ' ')
        scores_output_file.write('\n')

        for score in top_score:
            scores_output_file.write('{} '.format(int(score)))

    scores_output_file.close()

    result_list_file = '{}/{}_knn.txt'.format(result_folder, file_name)
    with open(result_list_file, 'wt') as list_fp:
        for k in range(len(knn_list)):
            list_fp.write(knn_list[k] + '\n')
    list_fp.close()

def DumpMatches(cv_image1, points1, cv_image2, points2, inliers, flip_xy = False):
    dst_image_width =  cv_image1.shape[1] + cv_image2.shape[1]
    dst_image_height = max(cv_image1.shape[0], cv_image2.shape[0])
    result_image = np.zeros((dst_image_height, dst_image_width, 3), dtype=np.uint8)
    result_image[:cv_image1.shape[0], :cv_image1.shape[1], :] = cv_image1[:, :, :3]
    result_image[:cv_image2.shape[0], cv_image1.shape[1]:, :] = cv_image2[:, :, :3]

    if flip_xy:
        x_index = 1
        y_index = 0
    else:
        x_index = 0
        y_index = 1

    for row in range(points1.shape[0]):
        cv2.circle(result_image, (int(points1[row, x_index]), int(points1[row, y_index])), 5, (0, 0, 255), 2)

    for row in range(points2.shape[0]):
        assert(int(points2[row, x_index]) < cv_image2.shape[1])
        assert(int(points2[row, y_index]) < cv_image2.shape[0])
        cv2.circle(result_image, (int(points2[row, x_index] + cv_image1.shape[1]), int(points2[row, y_index])), 5, (0, 0, 255), 2)

    for row in range(points1.shape[0]):
        if inliers[row]:
            cv2.line(result_image, (int(points1[row, x_index]), int(points1[row, y_index])), (int(points2[row, x_index] + cv_image1.shape[1]), int(points2[row, y_index])), (255, 0, 0), 2)
        # else:
        #     cv2.line(result_image, (int(points1[row, x_index]), int(points1[row, y_index])),
        #              (int(points2[row, x_index] + cv_image1.shape[1]), int(points2[row, y_index])), (0, 255, 0), 2)

    return result_image

def PerspectiveTransform(src_image, src_points, dst_points):
    transform = cv2.getPerspectiveTransform(src_points, dst_points)
    dst_image = cv2.warpPerspective(src_image, transform, src_image.shape)
    return dst_image


def load_vlfeat(folder, data_name):
    # Load the train features infor
    logging.info("Load training features infor")
    features_infor_file = op.join(folder, data_name + '_features_infor.vlfeat')
    with open(features_infor_file, 'rb') as fp:
        features_infor = DeSerializeFloatArray(fp, feature_infor_dimension)
    fp.close()

    # Load the train features image indices
    features_image_index_file = os.path.join(folder, data_name, '_features_image_index.vlfeat')
    with open(features_image_index_file, 'rb') as fp:
        features_image_index = DeSerializeIntArray(fp, 1)
    fp.close()

    # Load train images list
    images_list = []
    images_list_file = os.path.join(folder, data_name, 'list_images_retrieval.txt')
    for line in open(images_list_file):
        images_list.append(line.rstrip())

def get_hw(dataset, split, filter_idx):
    rows = dataset.iter_data(split, progress=True,
            filter_idx=filter_idx)
    return [(row[0], ' '.join(map(str,
        img_from_base64(row[-1]).shape[:2]))) 
        for row in rows]

def mp_do_sth_tsv(data, split, function_f):
    from pathos.multiprocessing import ProcessingPool as Pool
    dataset = TSVDataset(data)
    num_worker = 128
    num_tasks = num_worker * 3
    num_images = dataset.num_rows(split)
    num_image_per_worker = (num_images + num_tasks - 1) // num_tasks 
    assert num_image_per_worker > 0
    all_idx = []
    for i in range(num_tasks):
        curr_idx_start = i * num_image_per_worker
        if curr_idx_start >= num_images:
            break
        curr_idx_end = curr_idx_start + num_image_per_worker
        curr_idx_end = min(curr_idx_end, num_images)
        if curr_idx_end > curr_idx_start:
            all_idx.append(range(curr_idx_start, curr_idx_end))
    logging.info('creating pool')

    m = Pool(num_worker)
    all_result = m.map(function_f, all_idx)
    x = []
    for r in all_result:
        x.extend(r)
    dataset.write_data(x, split, 'hw')

def mp_do_sth_dict(dict, function_f):
    from pathos.multiprocessing import ProcessingPool as Pool
    num_worker = 128
    num_tasks = num_worker * 3
    num_images = dataset.num_rows(split)
    num_image_per_worker = (num_images + num_tasks - 1) // num_tasks 
    assert num_image_per_worker > 0
    all_idx = []
    for i in range(num_tasks):
        curr_idx_start = i * num_image_per_worker
        if curr_idx_start >= num_images:
            break
        curr_idx_end = curr_idx_start + num_image_per_worker
        curr_idx_end = min(curr_idx_end, num_images)
        if curr_idx_end > curr_idx_start:
            all_idx.append(range(curr_idx_start, curr_idx_end))
    logging.info('creating pool')

    m = Pool(num_worker)
    all_result = m.map(pfunction, all_idx)
    x = []
    for r in all_result:
        x.extend(r)

    dataset.write_data(x, split, 'hw')

def main():
    # unit testing
    #  convex_quadrilaterals
    points = map(np.array, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
    len(list(convex_quadrilaterals(points)))
    return

if __name__ == "__main__":
    main()