from __future__ import print_function

from google.cloud import vision
from google.cloud.vision import types

import argparse
import base64
import copy
import datetime
import json
import logging
import os
import os.path as op
import requests
import sys
import time

import _init_paths
from evaluation import eval_utils
from qd import qd_common, tsv_io
from qd.qd_common import json_dump
from qd.qd_common import image_url_to_bytes
from qd.tsv_io import TSVDataset, tsv_reader, tsv_writer
from qd.process_tsv import parallel_tsv_process

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


def call_gcloud(imgfile, det_file, key_col=0, img_col=2, detection="detection"):
    """
    Calls Google Could to get object detection results.
    https://cloud.google.com/vision/docs/detecting-objects

    Remember to: export GOOGLE_APPLICATION_CREDENTIALS=[PATH to auth file],
    or on Windows, set GOOGLE_APPLICATION_CREDENTIALS=[PATH to auth file]
    """
    if not qd_common.worth_create(imgfile, det_file):
        return

    client = vision.ImageAnnotatorClient()
    def gen_rows():
        num_imgs = 0
        time_elapsed = 0
        for idx, cols in enumerate(tsv_io.tsv_reader(imgfile)):
            imgkey = cols[key_col]
            num_imgs += 1
            img = types.Image(content=base64.b64decode(cols[img_col]))
            try:
                img_arr = qd_common.img_from_base64(cols[img_col])
                im_h, im_w, im_c = img_arr.shape
                tic = time.time()
                if detection == "detection":
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
            yield imgkey, qd_common.json_dump(res)

        logging.info("#imgs: {}, avg time: {}s per image".format(num_imgs, time_elapsed/num_imgs))

    tsv_io.tsv_writer(gen_rows(), det_file)


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
        all_res.append({"mid": obj.mid, "class": obj.description, "conf": obj.score})
    return all_res

def call_cvapi(uri, is_b64=False, model='tag'):
    """
        uri could be: 1) base64 encoded image string, set is_b64=True.
        2) file path.
        3) url, starting with http
    """
    use_prod = False
    if use_prod:
        analyze_url = 'https://westus.api.cognitive.microsoft.com/vision/v2.0/models/celebrities/analyze'
        headers = {
            'Ocp-Apim-Subscription-Key': "ZTkwNTE2ZmQ4NThlNDVjMmFhNDMzMjRlZjBlOThlN2E=",
            'Content-Type': 'application/octet-stream',
        }
    else:
        analyze_url = 'https://vision-usw2-dev.westus2.cloudapp.azure.com/api/analyze'
        headers = {
            "Authorization":"Basic MmY2ODBkY2QzZDcxNDAxZDhmNTcxZGU5ODFiYmI1MzA=",
            'Content-Type': 'application/octet-stream',
        }
    if model == 'tag':
        v = 'Tags'
    elif model == 'caption':
        v = 'Description'
    elif model == 'logo':
        v = 'Brands'
    else:
        raise ValueError('unknown target model {}'.format(model))
    params = {'visualFeatures': v}

    if is_b64:
        data = base64.b64decode(uri)
    elif uri.startswith('http'):
        #data = image_url_to_bytes(uri)
        #curl -k -X POST --data '{"url":"https://tse3.mm.bing.net/th?id=OIP.M9cfa7362b791260dbfbfbb2a5810a01eo2&pid=Api"}' -H "Content-Type:application/json" -H "Ocp-Apim-Subscription-Key:a9819f53e76b40099ac8aec16257f51d" "https://westus.api.cognitive.microsoft.com/vision/v2.0/models/celebrities/analyze"
        #data = '{"url": "' + uri + '"}'
        data = image_url_to_bytes(uri)
    else:
        assert op.isfile(uri)
        with open(uri, 'rb') as fp:
            data = fp.read()
    response = requests.post(
        analyze_url, headers=headers, params=params, data=data)
    response.raise_for_status()

    # The 'analysis' object contains various fields that describe the image. The most
    # relevant caption for the image is obtained from the 'description' property.
    analysis = response.json()

    if model == 'tag':
       tags = analysis['tags']
       ret = []
       for t in tags:
        ret.append({'class': t['name'], 'conf': t['confidence']})
    elif model == 'caption':
        tags = analysis["description"]['tags']
        ret = {'tags': tags}
        caption_res = analysis["description"]["captions"]
        if len(caption_res) > 0:
            ret['caption'] = caption_res[0]["text"].capitalize()
            ret['confidence'] = caption_res[0]["confidence"]
        else:
            #logging.info(str(analysis))
            pass
        ret = [ret]
    elif model == 'logo':
        ret = []
        for it in analysis['brands']:
            rect = it['rectangle']
            x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
            ret.append({'class': it['name'], 'rect': [x, y, x+w, y+h], 'conf':
                    it['confidence']})
    return ret

def call_cvapi_parallel(imgfile, key_col, img_col, model, outfile,
        is_debug=False):
    from qd.qd_common import limited_retry_agent
    def row_proc(parts):
        key = parts[key_col]
        str_img = parts[img_col]
        try:
            res = limited_retry_agent(5, call_cvapi, str_img, is_b64=True,
                        model=model)
        except:
            res = []
        return key, json_dump(res)

    num_proc = 0 if is_debug else 16
    parallel_tsv_process(row_proc, imgfile, outfile, num_proc)


def call_clarifai(uri, is_b64, model):
    """
    uri could be: 1) base64 encoded image string, set is_b64=True.
    2) file path.
    3) url, starting with http
    """
    time.sleep(5)  # limit of 10 requests per second
    if is_b64:
        response = model.predict_by_base64(uri)
    else:
        if uri.startswith('http'):
            response = model.predict_by_url(uri)
        else:
            assert op.isfile(uri)
            response = model.predict_by_filename(uri)

    tags = []
    for it in response['outputs'][0]['data']['concepts']:
        tags.append({'class': it['name'], 'conf': it['value']})
    return tags


def call_clarifai_parallel(imgfile, key_col, img_col, model, outfile,
        is_debug=False, api_key=None):
    from qd.qd_common import limited_retry_agent
    from clarifai.rest import ClarifaiApp

    if not api_key:
        api_key_file = op.expanduser('~/auth/clarifai.json')
        with open(api_key_file, 'r') as fp:
            r = json.load(fp)
        api_key = r['api_key']

    app = ClarifaiApp(api_key=api_key)
    model = app.public_models.general_model
    if is_debug:
        for parts in tsv_reader(imgfile):
            key = parts[key_col]
            str_img = parts[img_col]
            res = call_clarifai(str_img, False, model)
            print(res)
            return

    def row_proc(parts):
        key = parts[key_col]
        str_img = parts[img_col]
        try:
            res = limited_retry_agent(5, call_clarifai, str_img, False,
                        model=model)
        except:
            res = []
        return key, json_dump(res)

    num_proc = 0 if is_debug else 16
    parallel_tsv_process(row_proc, imgfile, outfile, num_proc)


def test():
    datas = [
        ['GettyImages2k', 'test'], ['linkedin1k', 'test'],
        #['OpenImageV4LogoTest', 'test'],['OpenImageV4LogoTest1', 'test'],
        #['old_capeval_2k_test', 'test'],
    ]
    services = [
        'clarifai',
        #'google',
        #'amazon',
        #'microsoft',
    ]
    target = 'tag'
    #target = 'logo'
    #target = 'caption'
    for data, split in datas:
        dirpath = op.join('data', data, 'api')
        imgfile = op.join('data', data, '{}.tsv'.format(split))
        urlfile = op.join('data', data, '{}.key.url.tsv'.format(split))
        key_col = 0
        img_col = 2
        for service in services:
            outfile = op.join(dirpath, '{}.{}.tsv'.format(service,
                        target))
            if op.isfile(outfile):
                logging.info('{} already exist'.format(outfile))
                continue
            if service == 'microsoft':
                call_cvapi_parallel(imgfile, key_col, img_col, target,
                        outfile, is_debug=False)
            elif service == 'amazon':
                response_file = op.join(dirpath, 'amazon.resp.tsv')
                det_file = op.join(dirpath, 'amazon.detection.tsv')
                tag_file = op.join(dirpath, 'amazon.tag.tsv')
                call_aws(imgfile, response_file, det_file, tag_file, key_col,
                        img_col)
            elif service == 'google':
                call_gcloud(imgfile, outfile, key_col, img_col,
                        target)
            elif service == 'clarifai':
                assert target == 'tag'
                call_clarifai_parallel(urlfile, 0, 1, target,
                        outfile, is_debug=False)


def main():
    from pprint import pformat
    logging.info(pformat(sys.argv))

    parser = argparse.ArgumentParser(description='Compare vision API with competitors')
    parser.add_argument('--service', required=True, type=str,
                        help='Choose from microsoft, google, amazon')
    parser.add_argument('--target', required=True, type=str,
                        help='Choose from tag, detection, logo')
    parser.add_argument('--dataset', required=True, type=str,
                        help='The name of evaluation dataset. See README.md for'
                             'a list of available datasets')
    parser.add_argument('--split', default='test', type=str,
                        help='The split of evaluation dataset, default is test')
    parser.add_argument('--outfile', required=True, type=str,
                        help='The file path to save results')

    args = parser.parse_args()

    data_root = './data'
    service = args.service
    assert service in ['microsoft', 'google', 'amazon']
    target = args.target
    assert target in ['tag', 'detection', 'logo']
    data = args.dataset
    split = args.split
    outfile = args.outfile

    imgfile = op.join(data_root, data, '{}.tsv'.format(split))
    key_col = 0
    img_col = 2
    if op.isfile(outfile):
        logging.info('{} already exist'.format(outfile))
        return
    if service == 'microsoft':
        call_cvapi_parallel(imgfile, key_col, img_col, target,
                outfile, is_debug=False)
    elif service == 'amazon':
        dirpath = op.dirname(outfile)
        response_file = op.join(dirpath, 'amazon.resp.tsv')
        det_file = op.join(dirpath, 'amazon.detection.tsv')
        tag_file = op.join(dirpath, 'amazon.tag.tsv')
        call_aws(imgfile, response_file, det_file, tag_file, key_col,
                img_col)
    elif service == 'google':
        call_gcloud(imgfile, outfile, key_col, img_col,
                target)



if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    main()
    #test()

