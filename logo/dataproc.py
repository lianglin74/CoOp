import base64
import collections
import cv2
import json
import logging
import os
from urllib2 import Request, urlopen

import _init_paths
from logo.classifier import BBOX_POS_PAIR_IDS, BBOX_NEG_PAIR_IDS
from scripts.qd_common import scrape_bing
from scripts.tsv_io import tsv_reader, tsv_writer


def convert_pair_to_label_file(pair_files, outfile):
    """
    pair_files: enumerable of tuple (fpath, is_pos), is_pos is 1 if fpath contains positive files, else 0
    pair_file: pair_id, pair_item1, pair_item2. each pair item is a dict of class, rect, imageId
    outfile: the label file of imageId, a list of bboxes
    """
    label_dict = collections.defaultdict(list)
    for fpath, is_pos in pair_files:
        pair_field = BBOX_POS_PAIR_IDS if is_pos else BBOX_NEG_PAIR_IDS
        for cols in tsv_reader(fpath):
            pair_id = cols[0]
            it1 = json.loads(cols[1])
            it2 = json.loads(cols[2])
            for it in [it1, it2]:
                imgkey = it["imageId"]
                target_it = None
                if imgkey in label_dict:
                    bbox_list = label_dict[imgkey]
                    idx = _retrieve_bbox_idx(it, bbox_list)
                    if idx is not None:
                        target_it = bbox_list[idx]

                # if current item does not exist in label_dict
                if not target_it:
                    label_dict[imgkey].append(it)
                    target_it = it
                if pair_field not in target_it:
                    target_it[pair_field] = []

                target_it[pair_field].append(pair_id)

    tsv_writer([[k, json.dumps(label_dict[k])] for k in label_dict], outfile)


def _retrieve_bbox_idx(bbox, bbox_list):
    """ Returns the index of bbox in bbox_list, else return None
    """
    for i, b in enumerate(bbox_list):
        if _is_bbox_same(bbox, b):
            return i
    return None

def _is_bbox_same(b1, b2):
    if b1["class"] == b2["class"] and all(b1["rect"][i]==b2["rect"][i] for i in range(4)):
        return True
    else:
        return False


def generate_canonical_tsv(dirpath, labelmap, outfile):
    """
    Saves images to TSV file: imgkey, list of bboxes, b64string
    """
    num_imgs = 0
    num_valid_imgs = 0
    with open(outfile, 'w') as fout:
        for cols in tsv_reader(labelmap):
            term = cols[0]
            imgdir = os.path.join(dirpath, term)
            if not os.path.exists(imgdir):
                logging.info("No images for {}".format(term))
                continue

            num_cur = 0
            for fname in os.listdir(imgdir):
                # if fname.rsplit('.', 1) not in ["jpg", "png", "svg"]:
                #     logging.info("invald image ext: {}".format(fname))
                #     continue
                num_imgs += 1
                im = cv2.imread(os.path.join(imgdir, fname), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                h, w, c = im.shape
                b64string = base64.b64encode(cv2.imencode('.jpg', im)[1])
                num_valid_imgs += 1
                num_cur += 1
                imgkey = '_'.join([term, fname])
                bbox = [{"class": term, "rect": [0, 0, w, h]}]
                fout.write('\t'.join([imgkey, json.dumps(bbox), b64string]))
                fout.write('\n')
            if num_cur <= 0:
                print("no enough image of {}: {}".format(term, num_cur))
    logging.info("find #img: {}, #valid: {}".format(num_imgs, num_valid_imgs))


def scrape_canonical_image(labelmap, outdir, num_imgs=30, ext=".png"):
    urlmap = os.path.join(outdir, "urlmap.tsv")
    with open(urlmap, 'w', buffering=0) as fout:
        for cols in tsv_reader(labelmap):
            term = cols[0]
            logging.info("term: {}".format(term))
            term_dir = os.path.join(outdir, term)
            if not os.path.exists(term_dir):
                os.mkdir(term_dir)
            query_term = ' '.join([term.replace('_', ' '), "logo"])
            urls = scrape_bing(query_term, num_imgs, trans_bg=True)
            for idx, url in enumerate(urls):
                if not url.endswith(ext):
                    continue
                fpath = os.path.join(term_dir, str(idx)+ext)
                if try_download_image(url, fpath):
                    fout.write('\t'.join([url, term, fpath]))
                    fout.write('\n')


def try_download_image(url, fpath):
    req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
    try:
        response = urlopen(req, None, 10)
        data = response.read()
        response.close()
        with open(fpath, 'wb') as fout:
            fout.write(data)
        im = Image.open(fpath)
        im.verify()
        return True
    except Exception as e:
        logging.info("error downloading: {}".format(e.message))
        if os.path.isfile(fpath):
            os.remove(fpath)
    return False
