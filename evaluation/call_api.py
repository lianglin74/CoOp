from __future__ import print_function

import boto3
from google.cloud import vision
from google.cloud.vision import types

import base64
import copy
import datetime
import json
import logging
import os
import sys

import _init_paths
from evaluation import eval_utils
from scripts import qd_common, tsv_io

def call_api(gt_config_file):
    """
    Gets Object Detection results from competitor's cloud vision service
    """
    gt_cfg = eval_utils.GroundTruthConfig(gt_config_file)
    rootpath = os.path.split(gt_config_file)[0]
    def get_det_path(dataset_name, api_name):
        return os.path.join(rootpath, "{}.{}.det{}.tsv"
                .format(dataset_name, api_name, datetime.datetime.today().strftime('%Y-%m-%d')))

    for dataset_name in gt_cfg.datasets():
        tsv_dataset, split_name = gt_cfg.gt_dataset(dataset_name)
        imgfile = tsv_dataset.get_data(split_name)

        for api in ["aws", "gcloud"]:
            logging.info("Calling {}...".format(api))
            det_file = get_det_path(dataset_name, api)
            if not qd_common.worth_create(imgfile, det_file):
                continue
            if api == "aws":
                response_file = det_file.rsplit('.', 1)[0] + ".response.tsv"
                tag_file = det_file.rsplit('.', 1)[0] + ".tag.tsv"
                call_aws(imgfile, det_file, response_file, tag_file)
            elif api == "gcloud":
                call_gcloud(imgfile, det_file)


def call_aws(imgfile, det_file, response_file, tag_file, key_col=0, img_col=2):
    """
    Calls AWS Rekognition API to get object detection and tagging results
    https://docs.aws.amazon.com/rekognition/latest/dg/labels-detect-labels-image.html

    imgfile: tsv file of imgkey, labels, b64string
    response_file: tsv file of imgkey, original response from aws
    post_process_file: tsv file of imgkey, list of bboxes
    """
    client=boto3.client('rekognition')
    all_det = []
    all_tag = []
    with open(response_file, 'w') as fresponse:
        for idx, cols in enumerate(tsv_io.tsv_reader(imgfile)):
            imgkey = cols[key_col]
            response = client.detect_labels(Image={'Bytes': base64.b64decode(cols[img_col])})
            fresponse.write("{}\t{}\n".format(imgkey, json.dumps(response)))
            print("Processed {}".format(idx+1), end='\r')
            sys.stdout.flush()

            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                logging.error("aws error for image: {}".format(imgkey))

            img = qd_common.img_from_base64(cols[img_col])
            im_h, im_w, im_c = img.shape
            res = post_process_aws(response, im_h, im_w)

            all_det.append([imgkey, json.dumps([b for b in res if "rect" in b])])
            all_tag.append([imgkey, json.dumps([b for b in res if "rect" not in b])])
    tsv_io.tsv_writer(all_det, det_file)
    tsv_io.tsv_writer(all_tag, tag_file)


def post_process_aws(response, im_height, im_width, prop_parents=True):
    """
    Post-processes AWS response. If prop_parents==True, parents labels will
    be added into results
    """
    all_res = []
    if "Labels" not in response:
        return all_res
    for res in response["Labels"]:
        label = res["Name"]
        cur_res = [{"class": label, "conf": res["Confidence"]/100.0}]  # tag
        for inst in res.get("Instances", []):
            # detection
            top, left, height, width = inst["BoundingBox"]["Top"], inst["BoundingBox"]["Left"], \
                                       inst["BoundingBox"]["Height"], inst["BoundingBox"]["Width"]
            cur_res.append({"class": label, "conf": inst["Confidence"]/100.0,
                            "rect": [left*im_width, top*im_height, (left+width)*im_width, (top+height)*im_height]})
        if prop_parents:
            num_cur = len(cur_res)
            for i in range(num_cur):
                for p in res.get("Parents", []):
                    tmp = copy.deepcopy(cur_res[i])
                    tmp["class"] = p["Name"]
                    cur_res.append(tmp)
        all_res.extend(cur_res)
    return all_res


def call_gcloud(imgfile, det_file, key_col=0, img_col=2):
    """
    Calls Google Could to get object detection results.
    https://cloud.google.com/vision/docs/detecting-objects

    Remember to: set GOOGLE_APPLICATION_CREDENTIALS=[PATH to auth file]
    """
    client = vision.ImageAnnotatorClient()
    all_det = []
    for idx, cols in enumerate(tsv_io.tsv_reader(imgfile)):
        imgkey = cols[key_col]
        img = types.Image(content=base64.b64decode(cols[img_col]))
        try:
            resp = client.object_localization(image=img).localized_object_annotations
            print("Processed {}".format(idx+1), end='\r')
            sys.stdout.flush()

            img = qd_common.img_from_base64(cols[img_col])
            im_h, im_w, im_c = img.shape
            res = post_process_gcloud(resp, im_h, im_w)
        except Exception as e:
            logging.error("gcloud failed for image: {}. Message: {}".format(imgkey, str(e)))
            res = []

        all_det.append([imgkey, json.dumps(res)])
    tsv_io.tsv_writer(all_det, det_file)


def post_process_gcloud(response, im_height, im_width):
    all_res = []
    for obj in response:
        vertices = obj.bounding_poly.normalized_vertices
        left = vertices[0].x * im_width
        top = vertices[0].y * im_height
        right = vertices[2].x * im_width
        bot = vertices[2].y * im_height
        all_res.append({"class": obj.name, "conf": obj.score,
                        "rect": [left, top, right, bot]})
    return all_res
