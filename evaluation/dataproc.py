import base64
import collections
import cv2
import datetime
import json
import logging
import multiprocessing as mp
import os
from PIL import Image
import shutil
from tqdm import tqdm
try:
    # For Python 3.0 and later
    from urllib.request import urlopen, Request
except ImportError:
    # Python 2
    from urllib2 import urlopen, Request
import uuid

import _init_paths
import logo.constants
from qd import qd_common, tsv_io, process_tsv
import evaluation.utils
from evaluation import eval_utils, generate_task, analyze_task


def prop_pred_not_in_gt(model_name, det_expid, gt_cfg_file, conf_thres=0.4, label_prefix=""):
    from qd import yolotrain
    rootpath = "/raid/data/uhrs/"
    uhrs_new_task_dir = "/raid/data/uhrs/status/new/"

    cfg_dir, cfg_fname = os.path.split(gt_cfg_file)
    cfg_dir = os.path.realpath(cfg_dir)
    assert(cfg_dir.startswith(rootpath))
    # get relative dir, HACK to make Windows compatiable
    cfg_dir = cfg_dir[len(rootpath):]

    gt_cfg = eval_utils.GroundTruthConfig(gt_cfg_file)
    task_cfg = {"model_name": model_name, "gt_config": os.path.join(cfg_dir, cfg_fname), "task_type": "VerifyBox", "pred_files": []}
    for dataset_key in gt_cfg.datasets():
        tsv_dataset, split = gt_cfg.gt_dataset(dataset_key)
        pred_file, _ = yolotrain.yolo_predict(full_expid=det_expid, test_data=tsv_dataset.name, test_split=split, gpus=[6,7])
        _, fname = os.path.split(pred_file)

        def gen_rows():
            for cols in tsv_io.tsv_reader(pred_file):
                if len(cols) < 2:
                    continue
                bboxes = json.loads(cols[1])
                for b in bboxes:
                    if not b["class"].startswith(label_prefix):
                        b["class"] = label_prefix + b["class"]
                yield cols[0], json.dumps(bboxes, separators=(',', ':'))
        tsv_io.tsv_writer(gen_rows(), os.path.join(rootpath, cfg_dir, fname))
        task_cfg["pred_files"].append({
            "dataset": dataset_key,
            "result": fname,
            "conf_threshold": conf_thres
        })
    qd_common.write_to_yaml_file(task_cfg,
            os.path.join(uhrs_new_task_dir, "{}_{}.yaml".format(model_name, uuid.uuid4())))
    # run watchdog


def build_dataset_from_web(tax_yaml_file, out_dataset_name, data_split):
    """
    Collects images for terms in labels_with_few_images, build TSVDataset(out_dataset_name)
    """
    target_num = 300
    outdir = "/raid/data/uhrs/draw_box/tasks/{}/".format(out_dataset_name)
    qd_common.ensure_directory(outdir)
    scraped_image_file = os.path.join(outdir, "bing.scrape.tsv")

    if qd_common.worth_create(tax_yaml_file, scraped_image_file):
        label_counts = []
        for it in qd_common.load_from_yaml_file(tax_yaml_file):
            label_counts.append((it["name"], target_num-it["cum_images_with_bb"]))

        scrape_image_parallel(label_counts, scraped_image_file, ext="jpg", query_format="{} flickr")

        hp_file = "/raid/data/uhrs/draw_box/honeypot/voc0712.test.easygt.txt"
        upload_dir = os.path.join(outdir, "upload")
        qd_common.ensure_directory(upload_dir)
        generate_task.generate_task_files("DrawBox", scraped_image_file, hp_file,
                os.path.join(upload_dir, out_dataset_name))

    # TODO: submit tasks to watchdog

    download_dir = os.path.join(outdir, "download")
    qd_common.ensure_directory(download_dir)
    task_res_files = evaluation.utils.list_files_in_dir(download_dir)
    if len(task_res_files) == 0:
        return
    res_file = os.path.join(outdir, "draw_box_results_uhrs.tsv")
    url2ans = analyze_task.analyze_draw_box_task(task_res_files, "uhrs", res_file)
    url2key = {p[2]: p[0] for p in tsv_io.tsv_reader(scraped_image_file)}

    out_data_root = os.path.join("/raid/data/", out_dataset_name)
    qd_common.ensure_directory(out_data_root)
    outtsv = os.path.join(out_data_root, "{}.tsv".format(data_split))
    outinfo = os.path.join(out_data_root, "train.generate.info.tsv")

    if qd_common.worth_create(res_file, outtsv):
        def data_iter():
            for url in url2ans:
                yield [url2key[url], json.dumps(url2ans[url], separators=(',', ':')), url]
        urls_to_img_file_parallel(data_iter, 2, (0,1), outtsv)
        info_data = [("images collected from", scraped_image_file)]
        tsv_io.tsv_writer(info_data, outinfo)

    process_tsv.populate_dataset_details(out_dataset_name, check_image_details=True)
    process_tsv.upload_image_to_blob(out_dataset_name, data_split)


def align_detection(dataset_name, split, pred_file, outfile=None, min_conf=0.0):
    """
    Aligns detection imgkeys with gt for visualization
    """
    pred_res = {p[0]: json.loads(p[1]) for p in tsv_io.tsv_reader(pred_file)}
    dataset = tsv_io.TSVDataset(dataset_name)
    def gen_output():
        for k, _ in dataset.iter_data(split, 'label'):
            bbox = pred_res[k] if k in pred_res else []
            if min_conf > 0:
                bbox = [b for b in bbox if "conf" not in b or b["conf"]>=min_conf]
            yield k, json.dumps(bbox, separators=(',', ':'))
    if not outfile:
        outfile = pred_file
    tsv_io.tsv_writer(gen_output(), outfile)


def convert_pair_to_label_file(pair_files, outfile):
    """
    pair_files: enumerable of tuple (fpath, is_pos), is_pos is 1 if fpath contains positive files, else 0
    pair_file: pair_id, pair_item1, pair_item2. each pair item is a dict of class, rect, imageId
    outfile: the label file of imageId, a list of bboxes
    """
    label_dict = collections.defaultdict(list)
    for fpath, is_pos in pair_files:
        pair_field = logo.constants.BBOX_POS_PAIR_IDS if is_pos else logo.constants.BBOX_NEG_PAIR_IDS
        for cols in tsv_io.tsv_reader(fpath):
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

    tsv_io.tsv_writer([[k, json.dumps(label_dict[k])] for k in label_dict], outfile)


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


def convert_local_images_to_b64(dirpath, labelmap, outfile, max_per_class=None):
    """
    Saves images to TSV file: imgkey, list of bboxes, b64string
    """
    num_imgs = 0
    num_valid_imgs = 0
    with open(outfile, 'w') as fout:
        for cols in tsv_io.tsv_reader(labelmap):
            term = cols[0]
            imgdir = os.path.join(dirpath, term)
            if not os.path.exists(imgdir):
                logging.info("No images for {}".format(term))
                continue

            num_cur = 0
            for fname in os.listdir(imgdir):
                if max_per_class and num_cur >= max_per_class:
                    break
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


def scrape_image_parallel(label_counts, outfile, ext="jpg", query_format="{}"):
    """
    Scrapes images from Bing with terms in labelmap, formatted as query_format
    label_counts: list of list: term, num_to_scrape, query_keywords(optional)
    outfile: TSV file of img_key, json list of dict {"class":term}, url
    """
    if os.path.isfile(outfile):
        raise ValueError("already exists: {}".format(outfile))
    from pathos.multiprocessing import ProcessPool as Pool
    num_worker = 128
    num_tasks = num_worker * 3
    num_rows = len(label_counts)
    num_rows_per_task = (num_rows + num_tasks - 1) // num_tasks
    all_args = []
    for i in range(num_tasks):
        cur_idx_start = i*num_rows_per_task
        if cur_idx_start >= num_rows:
            break
        cur_idx_end = min(cur_idx_start+num_rows_per_task, num_rows)
        if cur_idx_end > cur_idx_start:
            all_args.append((label_counts[cur_idx_start: cur_idx_end], ext, query_format))

    m = Pool(num_worker)
    all_result = m.map(scrape_image, all_args)
    m.close()
    def gen_res():
        for res in all_result:
            for row in res:
                yield row
    tsv_io.tsv_writer(gen_res(), outfile)


def scrape_image(args):
    label_counts, ext, query_format = args
    ret = []
    for cols in label_counts:
        term, num_imgs = cols[0], int(cols[1])
        if len(cols) == 3:
            query_term = cols[2]
        else:
            query_term = term
        query_term = query_format.format(query_term)
        if ext and "png" in ext:
            trans_bg = True
        else:
            trans_bg = False
        urls = qd_common.scrape_bing(query_term, num_imgs*2, trans_bg=trans_bg)

        num_valid_img = 0
        for idx, url in enumerate(urls):
            if num_valid_img >= num_imgs:
                break
            if not is_url_clean(url):
                continue
            im_bytes = image_url_to_bytes(url)
            if not im_bytes:
                continue
            try:
                im = qd_common.img_from_base64(base64.b64encode(im_bytes))
                h, w, c = im.shape
                assert c == 3 or c == 4
            except:
                continue
            # avoid key collision with existing images
            # img_key = "{}_{}.{}".format(term, datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"), ext)
            img_key = "{}_{}.{}".format(query_term, idx, ext)
            img_key = img_key.replace(' ', '_')
            label = [{"class": term}]
            ret.append([img_key, json.dumps(label), url])
            num_valid_img += 1
    return ret


def save_img_to_file(img_bytes, fpath):
    try:
        with open(fpath, 'wb') as fout:
            fout.write(img_bytes)
        return True
    except Exception as e:
        logging.info("error save: {}".format(e.message))
        if os.path.isfile(fpath):
            os.remove(fpath)
    return False


def image_url_to_bytes(url):
    req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17"})
    try:
        response = urlopen(req, None, 10)
        if response.code != 200:
            return None
        data = response.read()
        response.close()
        return data
    except Exception as e:
        logging.info("error downloading: {}".format(e))
    return None

def is_url_clean(url):
    s = evaluation.utils.escape_json_obj({"url": url})
    try:
        obj = evaluation.utils.load_escaped_json(s)
        assert(obj["url"] == url)
    except:
        return False
    return True


def urls_to_img_file_parallel(in_rows, url_col_idx, out_cols_idx, outpath, keep_order=False):
    """ Given iterable of rows, writes TSV file: out_cols_idx, followed by b64_image of the given url
    """
    num_workers = mp.cpu_count() + 10
    in_queue = mp.Queue(10 * num_workers)
    out_queue = mp.Queue(10 * num_workers)

    # save the key, used for re-ranking to keep the order
    import tempfile
    key_file = os.path.join(tempfile.gettempdir(), qd_common.gen_uuid() + '.tsv')

    def reader_process(in_rows, in_queue, num_workers, url_col_idx, out_cols_idx, key_file):
        max_col_idx = max(url_col_idx, max(out_cols_idx))
        with open(key_file, 'wb') as key_fp:
            for cols in tqdm(in_rows):
                if max_col_idx >= len(cols):
                    logging.info("invalid input row: {}".format(cols))
                    continue
                key_fp.write(cols[url_col_idx] + '\n')
                in_queue.put(cols)
        for _ in range(num_workers):
            in_queue.put(None)  # ending signal for workers

    def worker_process(in_queue, out_queue, url_col_idx, out_cols_idx):
        while True:
            cols = in_queue.get()
            if cols is None:
                out_queue.put(None)
                break
            out_cols = [cols[i] for i in out_cols_idx]
            url = cols[url_col_idx]
            img_bytes = image_url_to_bytes(url)
            if img_bytes is not None:
                b64_str = base64.b64encode(img_bytes)
                im = qd_common.img_from_base64(b64_str)
                h, w, c = im.shape
                # NOTE: opencv has a bug, even set the flag cv2.IMREAD_COLOR, some
                # images still return 4 channels
                if c!=3:
                    logging.info("invalid image format: {}".format(url))
                    continue
                out_cols.append(b64_str)
                out_queue.put(out_cols)

    def writer_process(out_queue, num_workers, outpath):
        def gen_output():
            num_finished = 0
            while True:
                if num_finished == num_workers:
                    break
                cols = out_queue.get()
                if cols is None:
                    num_finished += 1
                else:
                    yield cols
        tsv_io.tsv_writer(gen_output(), outpath)

    reader = mp.Process(target=reader_process, args=(in_rows,
            in_queue, num_workers, url_col_idx, out_cols_idx, key_file))
    reader.daemon = True
    reader.start()
    worker_pool = []
    for _ in range(num_workers):
        worker = mp.Process(target=worker_process, args=(in_queue,
                out_queue, url_col_idx, out_cols_idx))
        worker.daemon = True
        worker_pool.append(worker)
        worker.start()
    writer = mp.Process(target=writer_process, args=(out_queue, num_workers, outpath))
    writer.daemon = True
    writer.start()

    reader.join()
    if reader.exitcode != 0:
        logging.info('reader failed')
        for proc in worker_pool: #wait all process finished.
            proc.terminate()
        writer.terminate()
        return
    logging.info('reader finished')
    for proc in worker_pool: #wait all process finished.
        proc.join()
    logging.info('worker finished')
    writer.join()
    logging.info('writer finished')

    # reorder results
    if keep_order:
        ordered_keys = qd_common.load_list_file(key_file)
        tsv_io.reorder_tsv_keys(outpath, ordered_keys, outpath)
    os.remove(key_file)
