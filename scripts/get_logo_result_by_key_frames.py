import os
import os.path as op
import requests
import shutil
from video_indexer import VideoIndexer
from pprint import pprint
import json
import base64
import ast


def change_label_format_cvbrowser_to_vig(pred_file):
    results = []
    for cols in tsv_reader(pred_file):
        if len(cols) < 2:
            continue
        img_id = cols[0]
        
        # import ipdb; ipdb.set_trace()
        # ast to solve single quote and final comma issues
        json_data = ast.literal_eval(cols[1])
        pred_result = json.loads(json.dumps(json_data))

        bboxes = pred_result['objects']
        for b in bboxes:
            if 'rectangle' in b and not 'rect' in b:
                rectangle = b['rectangle']
                x = rectangle['x']
                y = rectangle['y']
                w = rectangle['w']
                h = rectangle['h']

                b['rect'] = [x, y, x+w-1, y+h-1]
            if 'object' in b and not 'class' in b:
                b['class'] = b['object']
            if 'confidence' in b and not 'conf' in b:
                b['conf'] = b['confidence']

        results.append((img_id, json.dumps(bboxes)))

    return results

def build_full_tsv(video_id):
    result_folder = "{}_result".format(video_id)
    if not op.isdir(result_folder):
        os.mkdir(result_folder)

    results_two_stage = []
    results_logo_nonlogo = []

    pred_two_stage_file = "{}_logo_two_stage_result.tsv".format(video_id)
    pred_logo_nonlogo_file = "{}_logo_nonlogo_result.tsv".format(video_id)

    results_two_stage = change_label_format_cvbrowser_to_vig(pred_two_stage_file)
    results_logo_nonlogo = change_label_format_cvbrowser_to_vig(pred_logo_nonlogo_file)

    results_img = []
    for cols in tsv_reader(pred_logo_nonlogo_file):
        if len(cols) < 2:
            continue
        img_id = cols[0]
        img_filename = "{}/{}.jpg".format(video_id, img_id)

        assert(op.isfile(img_filename))

        with open(img_filename, 'rb') as f:
            img_bytes = f.read()
            img_b64str =base64.b64encode(img_bytes)

        results_img.append((img_id, json.dumps([]), img_b64str))
    
    tsv_writer(results_img, "{}/test.tsv".format(result_folder))
    tsv_writer(results_logo_nonlogo, "{}/test.label.v1.tsv".format(result_folder))
    tsv_writer(results_two_stage, "{}/test.label.v2.tsv".format(result_folder))

    return


def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise
        # we should always check if it succeeds.
        assert op.isdir(op.abspath(path)), path
    return


def tsv_writer(values, tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    import sys
    is_py2 = sys.version_info.major == 2
    with open(tsv_file_name_tmp, 'wb') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            if is_py2:
                v = sep.join(map(lambda v: v.encode(
                    'utf-8') if isinstance(v, unicode) else str(v), value)) + '\n'
            else:
                v = sep.join(map(lambda v: v.decode() if type(v)
                                 == bytes else str(v), value)) + '\n'
                v = v.encode()
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)

    if os.path.isfile(tsv_file_name):
        os.remove(tsv_file_name)
    if os.path.isfile(tsv_lineidx_file):
        os.remove(tsv_lineidx_file)
    os.rename(tsv_file_name_tmp, tsv_file_name)
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)
    return


def tsv_reader(tsv_file_name, sep='\t'):
    with open(tsv_file_name, 'r') as fp:
        for line in fp:
            yield [x.strip() for x in line.split(sep)]
    return


def get_thumbnail(location, accountId, videoId, thumbnailId, accessToken, format='Jpeg'):

    thumbnail_filename = "{}/{}.jpg".format(videoId, thumbnailId)

    if not op.isfile(thumbnail_filename):
        url = "https://api.videoindexer.ai/{}/Accounts/{}/Videos/{}/Thumbnails/{}?format={}&accessToken={}".format(
            location, accountId, videoId, thumbnailId, format, accessToken)
        response = requests.get(url)
        if response.status_code == 200:
            with open(thumbnail_filename, 'wb') as f:
                f.write(response.content)

    return thumbnail_filename


def get_video_access_token(location, accountId, videoId, subscription_key='75b7ea02beac45e59aa53b2d4cd16517'):
    url = "https://api.videoindexer.ai/auth/{}/Accounts/{}/Videos/{}/AccessToken".format(
        location, accountId, videoId)
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    response = requests.get(url, headers=headers)
    accessToken = response.json()
    return accessToken


def get_logoV2():
    url = 'http://brandx2a.westus.azurecontainer.io/api/detect'
    return get_post_result(url)


def get_brand_two_stage(img_filename):
    url = "http://brandxtwostage.westus.azurecontainer.io/api/detect"
    return get_post_result(url, img_filename)


def get_brand_logo(img_filename):
    url = "http://brandxlogononlogo.westus.azurecontainer.io/api/detect"
    return get_post_result(url, img_filename)


def get_post_result(url, img_filename):
    files = {'image': open(img_filename, 'rb')}
    with requests.Session() as s:
        r = s.post(url, files=files)
    return r.json()


def get_video_info(sub_key, location, account_id, video_id):
    vi = VideoIndexer(
        vi_subscription_key=sub_key,
        vi_location=location,
        vi_account_id=account_id
    )

    video_info = vi.get_video_info(
        video_id,
        video_language='English'
    )
    return video_info


def get_video_infos(video_ids):

    CONFIG = {
        'SUBSCRIPTION_KEY': '75b7ea02beac45e59aa53b2d4cd16517',
        'LOCATION': 'eastus2',
        'ACCOUNT_ID': '75dafd2b-f0c3-4b95-99d0-4230e0a6fee8'
    }

    vi = VideoIndexer(
        vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
        vi_location=CONFIG['LOCATION'],
        vi_account_id=CONFIG['ACCOUNT_ID']
    )

    # video_ids = ['0cdc61d046',
    #              'deac295141',
    #              '996d87c6ca',
    #              '1b8f78387f',
    #              '88d997b278',
    #              'f5e2bc49c9',
    #              'f90b9436b6',
    #              'c98880a172',
    #              '541483b995']

    # video_ids = ['541483b995']

    for video_id in video_ids:
        info = vi.get_video_info(
            video_id,
            video_language='English'
        )

        with open("video_info_{}.txt".format(video_id), 'w') as f:
            f.write(json.dumps(info))
            pprint(info)

    return


def dict_writer(dict, filename):
    with open(filename, 'w') as f:
        json_str = json.dumps(dict)
        f.write(json_str)


def main():

    CONFIG = {
        'SUBSCRIPTION_KEY': '75b7ea02beac45e59aa53b2d4cd16517',
        'LOCATION': 'eastus2',
        'ACCOUNT_ID': '75dafd2b-f0c3-4b95-99d0-4230e0a6fee8'
    }

    video_ids = ['0cdc61d046',
                 'deac295141',
                 '996d87c6ca',
                 '1b8f78387f',
                 '88d997b278',
                 'f5e2bc49c9',
                 'f90b9436b6',
                 'c98880a172',
                 '541483b995']

    for video_id in video_ids:

        logo_nonlogo_result_filename = "{}_logo_nonlogo_result.tsv".format(
            video_id)
        logo_two_stage_result_filename = "{}_logo_two_stage_result.tsv".format(
            video_id)

        if op.isfile(logo_nonlogo_result_filename) and op.isfile(logo_two_stage_result_filename):
            continue

        logo_nonlogo_result = []
        logo_two_stage_result = []

        accessToken = get_video_access_token(
            CONFIG['LOCATION'], CONFIG['ACCOUNT_ID'], video_id)
        print("accessToken:\n{}".format(accessToken))
        # import ipdb; ipdb.set_trace()
        video_info_filename = "video_info_{}.txt".format(video_id)
        video_info = None
        if op.isfile(video_info_filename):
            with open(video_info_filename, 'r') as f:
                json_str = f.read()
            video_info = json.loads(json_str)
        else:
            video_info = get_video_info(
                CONFIG['SUBSCRIPTION_KEY'], CONFIG['LOCATION'], CONFIG['ACCOUNT_ID'], video_id)

        assert(video_info != None)

        thumbnailIds = get_keyframe_thumbnailIds(video_info)
        if not op.isdir(video_id):
            os.mkdir(video_id)
        # import ipdb; ipdb.set_trace()
        thumbnail_number = len(thumbnailIds)
        for i, thumbnailId in enumerate(thumbnailIds):
            print("{}/{}".format(i, thumbnail_number))
            thumbnail_filename = get_thumbnail(
                CONFIG['LOCATION'], CONFIG['ACCOUNT_ID'], video_id, thumbnailId, accessToken, format='Jpeg')

            logo_two_stage_result.append(
                (thumbnailId, get_brand_two_stage(thumbnail_filename)))
            logo_nonlogo_result.append(
                (thumbnailId, get_brand_logo(thumbnail_filename)))

        tsv_writer(logo_nonlogo_result, logo_nonlogo_result_filename)
        tsv_writer(logo_two_stage_result, logo_two_stage_result_filename)

    return


def print_dict_keys(obj, leading_str='-'):
    if isinstance(obj, dict):
        for k in obj:
            print(leading_str + k)
            print_dict_keys(obj[k], '-{}'.format(leading_str))
    elif isinstance(obj, list):
        for ob in obj:
            print_dict_keys(ob, '-{}'.format(leading_str))
            break

    return


def get_keyframe_thumbnailIds(video_info):
    '''
    https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/media-services/video-indexer/video-indexer-output-json-v2.md
    '''
    thumbnailIds = []
    videos = video_info['videos']
    # import ipdb; ipdb.set_trace()
    # print_dict_keys(video_info)
    # printthumbnailId
    # return
    for video in videos:
        insights = video['insights']

        shots = insights['shots']
        for shot in shots:
            keyFrames = shot['keyFrames']
            for keyFrame in keyFrames:
                instances = keyFrame['instances']
                for instance in instances:
                    thumbnailId = instance['thumbnailId']
                    thumbnailIds.append(thumbnailId)
    return thumbnailIds


if __name__ == '__main__':
    # main()

    video_ids = ['0cdc61d046',
                'deac295141',
                '996d87c6ca',
                '1b8f78387f',
                '88d997b278',
                'f5e2bc49c9',
                'f90b9436b6',
                'c98880a172',
                '541483b995']

    for video_id in video_ids:
        build_full_tsv(video_id)
    
