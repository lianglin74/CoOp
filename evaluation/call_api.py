from __future__ import print_function

from google.cloud import vision
from google.cloud.vision import types

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

def main():
    datas = [
        ['GettyImages2k', 'test'], ['linkedin1k', 'test'],
        #['OpenImageV4LogoTest', 'test'],['OpenImageV4LogoTest1', 'test'],
        #['old_capeval_2k_test', 'test'],
    ]
    services = [
        'google',
        #'amazon',
        'microsoft',
    ]
    target = 'tag'
    #target = 'logo'
    #target = 'caption'
    for data, split in datas:
        dirpath = op.join('data', data)
        imgfile = op.join(dirpath, '{}.tsv'.format(split))
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

def main4():
    dirpath = 'data/GettyImages'
    fpath = 'data/GettyImages/images.json'
    contents = json.load(open(fpath))
    count = 0
    val_set = []
    test_set = []
    for c in contents:
        key = '{}/{}'.format(c['id'], c['uri'])
        cap = c['description']
        if c['reviewed'] and count < 2000:
            val_set.append([key, cap])
            count += 1
        else:
            test_set.append([key, cap])
    tsv_writer(val_set, op.join(dirpath, 'gettyimages_val.csv'), sep=',')
    tsv_writer(test_set, op.join(dirpath, 'gettyimages_test.csv'), sep=',')
    print(count)

def main3():
    data = 'OfficePPTUserInserted'
    imgfile = 'data/{}/test.tsv'.format(data)
    key_col = 0
    img_col = 2
    model = 'caption'
    outfile = 'data/{}/cvapi.caption.tsv'.format(data)
    call_cvapi_parallel(imgfile, key_col, img_col, model, outfile)

    csv_file = 'data/{}/caption_result.csv'.format(data)
    caption_iter = tsv_reader(outfile)
    #orig_iter = tsv_reader(imgfile)
    #def gen_rows():
        #for parts1, parts2 in zip(caption_iter, orig_iter):
            #url = parts1[0]
            #assert parts2[0] == url
            #orig = json.loads(parts2[1])['description']
            #resp = json.loads(parts1[1])
            #if len(resp) == 0:
                #caption = ''
                #conf = ''
            #else:
                #resp = resp[0]
                #caption = resp.get('caption', '; '.join(resp['tags']))
                #conf = resp.get('confidence', '')
            #yield url, caption, conf, orig
    def gen_rows():
        for parts1 in caption_iter:
            url = parts1[0]
            resp = json.loads(parts1[1])
            if len(resp) == 0:
                caption = ''
                conf = ''
            else:
                resp = resp[0]
                caption = resp.get('caption', '; '.join(resp['tags']))
                conf = resp.get('confidence', '')
            yield url, caption, conf

    tsv_writer(gen_rows(), csv_file, sep=',')

def main2():
    from tqdm import tqdm
    from evaluation.collect_data_for_labels import urls_to_img_file_parallel
    data = 'GettyImages'
    fpath = 'data/GettyImages/images.json'
    with open(fpath, 'r') as fp:
        contents = json.load(fp)
    fields = ['tags', 'description', 'reviewed']
    num_reviewed = 0
    all_url = []
    all_id_set = set()
    for c in contents:
        assert c['id'] not in all_id_set
        all_id_set.add(c['id'])
        url = 'https://osizewuspersimmon001.blob.core.windows.net/m365content/publish/{}/{}'.format(
            c['id'], c['uri'])
        all_url.append([url, json_dump(c)])

    url_file = 'data/{}/urls.tsv'.format(data)
    tsv_writer(all_url, url_file)
    def row_proc(parts):
        url, content = parts
        content = json.loads(content)
        try:
            resp = call_cvapi(url)
            caption = resp.get('caption', '; '.join(resp['tags']))
        except Exception as e:
            logging.info('fail to call {}, error: {}'.format(url, str(e)))
            resp = {}
            caption = ''
        return url, caption, resp.get('confidence', ''), content['description']
    outfile = 'data/{}/caption_results.csv'.format(data)
    parallel_tsv_process(row_proc, url_file,
        outfile, 0, out_sep=',')

    #outpath = 'data/{}/images.tsv'.format(data)
    #urls_to_img_file_parallel(all_url, 0, [0, 1], outpath, keep_order=False, is_debug=False)

    #out_data = 'GettyImages2k'
    #select_urls = []
    #for url, content, str_img in tsv_reader(outpath):
        #content = json.loads(content)
        #if content['reviewed']:
            #select_urls.append(url)

    #assert len(select_urls) >= 2000
    #select_urls = set(select_urls[:2000])
    #assert len(select_urls) == 2000
    #out_dataset = TSVDataset(out_data)
    #def gen_rows():
        #for url, content, str_img in tsv_reader(outpath):
            #if url in select_urls:
                #content = json.loads(content)
                #tags = [{'class': c} for c in content['tags']]
                #yield content['id'], json_dump(tags), str_img

    #out_dataset.write_data(gen_rows(), 'test')

def main1():
    from evaluation.dataproc import urls_to_img_file_parallel, scrape_image_parallel

    data = 'linkedin1k'
    label_counts = [
        ['', 1000], ['ceo', 100], ['company', 100], ['people', 100],
        ['office', 100], ['chart', 50], ['business', 100]
    ]
    outfile = 'data/{}/query.tsv'.format(data)
    #scrape_image_parallel(label_counts, outfile, ext="jpg",
            #query_format="site:linkedin.com {}")
    urls = []
    for _, term, url in tsv_reader(outfile):
        if 'linkedin.com' in url:
            urls.append(url)
    urls = set(urls)
    print(len(urls))
    in_rows = [[url] for url in urls]
    imgpath = 'data/{}/query.image.tsv'.format(data)
    #urls_to_img_file_parallel(in_rows, 0, [0],imgpath, keep_order=False, is_debug=False)
    dataset = TSVDataset(data)
    def gen_rows():
        count = 0
        for url, str_img in tsv_reader(imgpath):
            count += 1
            yield 'linkedin{}'.format(count), '[]', str_img
    dataset.write_data(gen_rows(), 'test')

if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    #url = 'https://osizewuspersimmon001.blob.core.windows.net/m365content/publish/00c70963-e397-484b-a1e3-94d83ec8ec23/901382928.jpg'
    #call_cvapi(url)
    main()
    #imgfile = 'data/openimageV4_256/trainval.tsv'
    #det_file = 'data/openimageV4_256/trainval.google.tag.tsv'
    #call_gcloud(imgfile, det_file, key_col=0, img_col=2, detection="tag")

