from __future__ import print_function

from google.cloud import vision
from google.cloud.vision import types

import base64
import copy
import datetime
import json
import logging
import os
import sys
import time

import _init_paths
from evaluation import eval_utils
from qd import qd_common, tsv_io

def call_api(gt_config_file, det_type):
    """
    Gets Object Detection results from competitor's cloud vision service
    """
    gt_cfg = eval_utils.GroundTruthConfig(gt_config_file)
    rootpath = os.path.split(gt_config_file)[0]
    def get_det_path(dataset_name, api_name):
        return os.path.join(rootpath, "{}.{}.{}.tsv"
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
                call_aws(imgfile, response_file, det_file, tag_file)
            elif api == "gcloud":
                call_gcloud(imgfile, det_file, detection=det_type)


def call_aws(imgfile, response_file, det_file, tag_file, key_col=0, img_col=2):
    """
    Calls AWS Rekognition API to get object detection and tagging results
    https://docs.aws.amazon.com/rekognition/latest/dg/labels-detect-labels-image.html

    imgfile: tsv file of imgkey, labels, b64string
    response_file: tsv file of imgkey, original response from aws
    post_process_file: tsv file of imgkey, list of bboxes
    """
    if not qd_common.worth_create(imgfile, response_file):
        return

    import boto3
    client=boto3.client('rekognition')
    all_det = []
    all_tag = []
    time_elapsed = 0
    num_imgs = 0
    with open(response_file, 'w') as fresponse:
        for idx, cols in enumerate(tsv_io.tsv_reader(imgfile)):
            num_imgs += 1
            imgkey = cols[key_col]
            tic = time.time()
            response = client.detect_labels(Image={'Bytes': base64.b64decode(cols[img_col])})
            time_elapsed += time.time() - tic
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
    logging.info("#imgs: {}, avg time: {}s per image".format(num_imgs, time_elapsed/num_imgs))


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


def call_gcloud(imgfile, det_file, key_col=0, img_col=2, detection="object"):
    """
    Calls Google Could to get object detection results.
    https://cloud.google.com/vision/docs/detecting-objects

    Remember to: export GOOGLE_APPLICATION_CREDENTIALS=[PATH to auth file],
    or on Windows, set GOOGLE_APPLICATION_CREDENTIALS=[PATH to auth file]
    """
    if not qd_common.worth_create(imgfile, det_file):
        return

    client = vision.ImageAnnotatorClient()
    all_det = []
    time_elapsed = 0
    num_imgs = 0
    for idx, cols in enumerate(tsv_io.tsv_reader(imgfile)):
        imgkey = cols[key_col]
        num_imgs += 1
        img = types.Image(content=base64.b64decode(cols[img_col]))
        try:
            img_arr = qd_common.img_from_base64(cols[img_col])
            im_h, im_w, im_c = img_arr.shape
            tic = time.time()
            if detection == "object":
                resp = client.object_localization(image=img).localized_object_annotations
                res = post_process_gcloud(resp, im_h, im_w)
            elif detection == "logo":
                resp = client.logo_detection(image=img).logo_annotations
                res = post_process_gcloud_logo(resp)
            elif detection == "tag":
                resp = client.label_detection(image=img).label_annotations
                res = post_process_gcloud_tag(resp)
            else:
                raise ValueError("Invalid detection type: {}".format(detection))
            time_elapsed += time.time() - tic
            print("Processed {}".format(idx+1), end='\r')
            sys.stdout.flush()
        except ValueError as e:
            raise e
        except Exception as e:
            logging.error("gcloud failed for image: {}. Message: {}".format(imgkey, str(e)))
            res = []

        all_det.append([imgkey, json.dumps(res)])
    tsv_io.tsv_writer(all_det, det_file)
    logging.info("#imgs: {}, avg time: {}s per image".format(num_imgs, time_elapsed/num_imgs))


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


def post_process_gcloud_logo(response, prefix="logo of "):
    all_res = []
    for obj in response:
        vertices = obj.bounding_poly.vertices
        left = vertices[0].x
        top = vertices[0].y
        right = vertices[2].x
        bot = vertices[2].y
        c = obj.description
        if not c.lower().startswith(prefix):
            c = prefix + c
        all_res.append({"class": c, "conf": obj.score,
                        "rect": [left, top, right, bot]})
    return all_res


def post_process_gcloud_tag(response):
    all_res = []
    for obj in response:
        all_res.append({"mid": obj.mid, "class": obj.description, "score": obj.score})
    return all_res
