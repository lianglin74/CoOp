import collections
from datetime import datetime
import logging
import json
import os
import os.path as op

import _init_paths
from generate_task import generate_task_files
from analyze_task import analyze_draw_box_task
from dataproc import scrape_image_parallel, scrape_image, urls_to_img_file_parallel
from uhrs import UhrsTaskManager

from qd.qd_common import ensure_directory, json_dump, ensure_remove_dir, calculate_iou, hash_sha1
from qd.tsv_io import tsv_reader, tsv_writer, TSVDataset


def resubmit_tasks():
    old_task_upload_dir = "//ivm-server2/IRIS/OD/xiaowh/vendor/resubmit_2k_new/upload_0807"
    old_task_id_tsv = "//ivm-server2/IRIS/OD/xiaowh/vendor/resubmit_2k_new/task_ids_real_resubmit_0807.tsv"

    rootdir = "//ivm-server2/IRIS/OD/xiaowh/vendor/resubmit_2k_new"
    new_ts = '0809'
    new_task_upload_dir = op.join(rootdir, 'upload_{}'.format(new_ts))
    new_task_id_tsv = op.join(rootdir, 'task_ids_real_resubmit_{}.tsv'.format(new_ts))

    old_download_dir = op.join(rootdir, "old_download")
    res_files = download_uhrs_by_task_ids([old_task_id_tsv], old_download_dir, use_cache=True)
    old_url_bboxes_fname = "old_url_bboxes.tsv"
    get_url_bboxes_from_uhrs_task_res(res_files, rootdir, out_real_label_file="old_real.tsv",
            out_proto_label_file="old_proto.tsv", start_time=None,
            url_bboxes_fname=old_url_bboxes_fname, remove_nopair_labels=False)

    old_url2bboxes = dict()
    for url, str_bboxes in tsv_reader(op.join(rootdir, old_url_bboxes_fname)):
        assert url not in old_url2bboxes
        old_url2bboxes[url] = json.loads(str_bboxes)
    old_upload_files = list_files_in_dir(old_task_upload_dir, sort_by_name=True)

    url2bboxes_merged = {url: _merge_labeled_bboxes(bboxes) for url, bboxes in old_url2bboxes.items()}
    new_upload_files = []
    for old_upload_file in old_upload_files:
        new_upload_file = op.join(new_task_upload_dir, op.basename(old_upload_file))
        update_bbox_in_upload_file(old_upload_file, url2bboxes_merged, new_upload_file)
        new_upload_files.append(new_upload_file)

    import ipdb; ipdb.set_trace()
    task_group_id = 91761
    upload_uhrs_tasks(new_upload_files, task_group_id, new_task_id_tsv)

def update_bbox_in_upload_file(old_upload_file, url2bboxes, new_upload_file):
    def gen_rows():
        for row_idx, parts in enumerate(tsv_reader(old_upload_file)):
            if row_idx == 0:
                assert parts[0] == "input_content"
                yield parts
            else:
                task_json = json.loads(parts[0])
                url = task_json['image_url']
                if url in url2bboxes:
                    task_json['bboxes'] = url2bboxes[url]
                else:
                    bboxes = task_json.get('bboxes', [])
                    if len(bboxes) > 1:
                        bboxes = _merge_labeled_bboxes(bboxes)
                    task_json['bboxes'] = bboxes
                yield [json.dumps(task_json)]
    tsv_writer(gen_rows(), new_upload_file)

def list_files_in_dir(dirpath, sort_by_name=False, ignore_idx=True):
    fnames = [f for f in os.listdir(dirpath)]
    if ignore_idx:
        fnames = [f for f in fnames if not f.endswith('.lineidx')]
    if sort_by_name:
        def fname2idx(fname):
            assert fname.endswith('.tsv')
            num = fname.rsplit('_', 1)[1]
            num = num[len('part'): -len('.tsv')]
            assert num.isdigit()
            return int(num)
        fname_idx_pairs = [(fname, fname2idx(fname)) for fname in fnames]
        fname_idx_pairs = sorted(fname_idx_pairs, key=lambda p: p[1])
        fnames = [p[0] for p in fname_idx_pairs]
    return [op.join(dirpath, f) for f in fnames]


def review_vendor_labels():
    timestamp = '0903'
    # rootdir = "//ivm-server2/IRIS/OD/xiaowh/vendor/resubmit_2k_new"
    # tsvdataset_dir = op.join("//vigdgx02/raid_xiaowh/visualize/", 'review_vendor_{}'.format(timestamp))
    rootdir = "/mnt/ivm_server2_od/xiaowh/vendor/resubmit_2k_new"
    tsvdataset_dir = op.join("/raid/xiaowh/visualize/", 'review_vendor_{}'.format(timestamp))
    review_dir = op.join(rootdir, timestamp)

    # task_id_tsvs = [op.join(rootdir, 'task_ids_real_resubmit_0809.tsv')]
    # task_res_dir = op.join(review_dir, '{}download'.format(timestamp))
    # task_res_files = download_uhrs_by_task_ids(task_id_tsvs, task_res_dir)
    # # task_res_files = [op.join(task_res_dir, fname) for fname in os.listdir(task_res_dir)]

    # out_real_label_file = "url_bboxes_real.tsv"
    # out_proto_label_file = "url_bboxes_proto.tsv"
    # get_url_bboxes_from_uhrs_task_res(task_res_files, review_dir, out_real_label_file,
    #         out_proto_label_file, start_time=datetime(2019, 8, 21),
    #         remove_nopair_labels=False)

    url_bboxes_to_tsvdataset(tsvdataset_dir,
            [('train', op.join(review_dir, 'url_bboxes_proto.tsv')),
            ('test', op.join(review_dir, 'url_bboxes_real.tsv'))])

def inspect_uploaded_tasks():
    upload_dir = '//ivm-server2/IRIS/OD/xiaowh/vendor/resubmit_2k_new/upload_0807'
    upload_files = list_files_in_dir(upload_dir)
    url2row = dict()
    num_overlap_urls = 0
    num_overlap_bboxes = 0
    for upload_file in upload_files:
        for row_idx, parts in enumerate(tsv_reader(upload_file)):
            if row_idx == 0:
                continue
            task_json = json.loads(parts[0])
            url = task_json['image_url']
            if url in url2row:
                print(url2row[url])
                print(parts)
                num_overlap_urls += 1
            else:
                url2row[url] = parts[0]

            bboxes = task_json['bboxes']
            for i in range(len(bboxes)):
                for j in range(i+1, len(bboxes)):
                    if calculate_iou(bboxes[i]['rect'], bboxes[j]['rect']) > 0.8:
                        num_overlap_bboxes += 1
    print(len(url2row), num_overlap_bboxes, num_overlap_urls)

def _merge_labeled_bboxes(bboxes, iou_thres=0.75):
    res = []
    num_bboxes = len(bboxes)
    for i in range(num_bboxes):
        is_overlap = False
        for j in range(i+1, num_bboxes):
            if calculate_iou(bboxes[i]['rect'], bboxes[j]['rect']) > iou_thres:
                is_overlap = True
                if bboxes[i]['class'] != bboxes[j]['class']:
                    print(bboxes[i]['class'], bboxes[j]['class'])
                break
        if not is_overlap:
            res.append(bboxes[i])
    # if len(res) < len(bboxes):
    #     print('merged {}'.format(len(bboxes) - len(res)))
    return res

def download_uhrs_by_task_ids(task_id_tsvs, outdir, use_cache=True, complete_only=False):
    if not use_cache:
        ensure_remove_dir(outdir)
    ensure_directory(outdir)
    res_files = []
    for task_id_tsv in task_id_tsvs:
        for parts in tsv_reader(task_id_tsv):
            task_group_id = int(parts[0])
            task_id = int(parts[1])
            fname = '.'.join([parts[0], parts[1], op.basename(parts[2])])
            if complete_only and not UhrsTaskManager.is_task_completed(task_group_id, task_id):
                continue
            fpath = op.join(outdir, fname)
            if not op.isfile(fpath):
                try:
                    UhrsTaskManager.download_task(task_group_id, task_id, fpath)
                except:
                    logging.info('fail to download: {} {}'.format(task_group_id, task_id))
                    continue
            res_files.append(fpath)
            logging.info('download {}'.format(fpath))
    return res_files

def upload_uhrs_tasks(upload_files, task_group_id, task_id_tsv, num_judgment=1):
    def gen_task_ids():
        for upload_file in upload_files:
            task_id = UhrsTaskManager.upload_task(task_group_id, upload_file, num_judgment=num_judgment)
            yield task_group_id, task_id, upload_file
    tsv_writer(gen_task_ids(), task_id_tsv)

def scrape_keywords_to_search_images():
    # dname = "logo200"
    dname = "2k_new"
    # ext = "png"
    ext = "jpg"
    img_type = "real"
    num_imgs_per_query = 80
    rootdir = "//ivm-server2/IRIS/OD/xiaowh/vendor/"
    # rootdir = "/mnt/ivm_server2_od/xiaowh/vendor/"
    # keywords_file = r"C:\Users\xiaowh\OneDrive - Microsoft\share\filtered_labels_to_add.tsv"
    keywords_file = op.join(rootdir, dname, "2k_labels_to_add.txt")
    outfile = op.join(rootdir, dname, "scraped_urls_{}.tsv".format(img_type + '_' + ext))
    filtered_fname1 = "{}_dedup_scraped_urls1.tsv".format(img_type + '_' + ext)
    filtered_fname2 = "{}_dedup_scraped_urls2.tsv".format(img_type + '_' + ext)
    outdir = op.join(rootdir, dname)
    log_file = "task_ids_{}.tsv".format(img_type + '_' + ext)

    out_real_label_file = "url_bboxes_real.tsv"
    out_proto_label_file = "url_bboxes_proto.tsv"
    phase = "upload"

    ranked_list_file = op.join(rootdir, dname, "2k_ranked_logo_list.txt")
    blacklist_file = op.join(rootdir, dname, "no_found_logos.txt")
    blacklist_labels_set = set(p[0] for p in tsv_reader(blacklist_file))
    ranked_labels = [p[0] for p in tsv_reader(ranked_list_file) if p[0] not in blacklist_labels_set]

    if phase == "scrape":
        label2keywords = collections.defaultdict(list)
        cur_term = None
        img_type_col_idx = 4
        brand_col_idx = 0
        for i, parts in enumerate(tsv_reader(keywords_file)):
            if i == 0:
                continue
            if i%2 == 1:
                cur_term = parts[brand_col_idx]
                assert(parts[img_type_col_idx] == "prototype logo image")
                continue
                label2keywords[cur_term].append(cur_term + " logo")
            else:
                assert cur_term is not None
                assert parts[brand_col_idx] == ""
                assert(parts[img_type_col_idx] == "real image")
                # # !!!!HACK: do not include real images for this time
                # continue
                label2keywords[cur_term].append(cur_term )
            for j in range(img_type_col_idx+1, len(parts)):
                if parts[j]:
                    label2keywords[cur_term].append(parts[j])

        all_terms = []
        for label in label2keywords:
            terms = label2keywords[label]
            terms_set = set()
            for term in terms:
                if term.lower() in terms_set:
                    continue
                terms_set.add(term.lower())
                all_terms.append([label, num_imgs_per_query, term])
        scrape_image_parallel(all_terms, outfile, ext=ext, query_format="{}")

        label2urls = collections.defaultdict(set)
        label2parts = collections.defaultdict(list)
        ranked_labels_set = set(ranked_labels)
        assert len(ranked_labels) == len(ranked_labels_set)
        for parts in tsv_reader(outfile):
            label = json.loads(parts[1])[0]["class"]
            if label not in ranked_labels_set:
                continue
            url = parts[2]
            if url in label2urls[label]:
                continue
            label2urls[label].add(url)
            label2parts[label].append(parts)

        target_size = int(len(ranked_labels) / 2)
        outdata1 = []
        outdata2 = []
        for idx, label in enumerate(ranked_labels):
            if idx < target_size:
                outdata1.extend(label2parts[label])
            else:
                outdata2.extend(label2parts[label])

        tsv_writer(outdata1, op.join(outdir, filtered_fname1))
        tsv_writer(outdata2, op.join(outdir, filtered_fname2))

    if phase == "upload":
        for label_file in [filtered_fname1, filtered_fname2]:
            ensure_directory(op.join(outdir, "upload_{}".format(img_type)))
            outbase = op.join(outdir, "upload_{}".format(img_type), label_file)
            task_files = generate_task_files("DrawBox", op.join(outdir, label_file), None, outbase,
                            None, num_tasks_per_hit=10, num_hp_per_hit=0,
                            num_hits_per_file=2000)
            task_group_id = 91761
            with open(op.join(outdir, log_file), 'a') as fp:
                for task_file in task_files:
                    task_id = UhrsTaskManager.upload_task(task_group_id, task_file, num_judgment=1)
                    fp.write("{}\t{}\t{}\n".format(task_group_id, task_id, task_file))

    if phase == "download":
        download_dir = op.join(outdir, "download")
        ensure_directory(download_dir)
        res_files = []
        # for parts in tsv_reader(op.join(outdir, log_file)):
        #     task_group_id = int(parts[0])
        #     task_id = int(parts[1])
        #     fname = op.basename(parts[2])
        #     if UhrsTaskManager.is_task_completed(task_group_id, task_id):
        #         fpath = op.join(download_dir, fname)
        #         print(fpath)
        #         UhrsTaskManager.download_task(task_group_id, task_id, fpath)
        #         res_files.append(fpath)

        res_files = [op.join(download_dir, fname) for fname in os.listdir(download_dir)]
        get_url_bboxes_from_uhrs_task_res(res_files, outdir, out_real_label_file,
            out_proto_label_file, start_time=None,
            remove_nopair_labels=True)

def get_url_bboxes_from_uhrs_task_res(res_files, outdir, out_real_label_file,
            out_proto_label_file, start_time=None,
            url_bboxes_fname="url_bboxes.tsv", remove_nopair_labels=False):
    url_ans_tsv = op.join(outdir, url_bboxes_fname)
    url2ans = analyze_draw_box_task(res_files, url_ans_tsv, start_time=start_time)
    proto_labels = []
    real_labels = []
    for url in url2ans:
        bboxes = url2ans[url]
        if not bboxes:
            continue
        is_proto = False
        for b in bboxes:
            if "IsPrototype" in b:
                is_proto = True
        if is_proto:
            c = bboxes[0]['class']
            # keurig & green mountain coffee appear at the same time
            if not all([b['class'] == c for b in bboxes]):
                continue
            proto_labels.append([url, bboxes])
        else:
            real_labels.append([url, bboxes])

    def count_labels(labels):
        label2count = collections.defaultdict(int)
        for url, bboxes in labels:
            for b in bboxes:
                label2count[b["class"]] += 1
        return label2count

    proto_label2counts = count_labels(proto_labels)
    real_label2counts = count_labels(real_labels)

    keep_labels = set()
    if remove_nopair_labels:
        for label in real_label2counts:
            if label in proto_label2counts:
                keep_labels.add(label)
    else:
        for label in real_label2counts:
            keep_labels.add(label)
        for label in proto_label2counts:
            keep_labels.add(label)
    print("Get {} categories".format(len(keep_labels)))

    def filter_labels(url_bboxes_list, keep_labels):
        for url, bboxes in url_bboxes_list:
            filtered_bboxes = [b for b in bboxes if b["class"] in keep_labels]
            if len(filtered_bboxes) == 0:
                continue
            yield url, json_dump(filtered_bboxes)
    tsv_writer(filter_labels(real_labels, keep_labels), op.join(outdir, out_real_label_file))
    tsv_writer(filter_labels(proto_labels, keep_labels), op.join(outdir, out_proto_label_file))

def url_bboxes_to_tsvdataset(tsvdataset_dir, all_split_fpath):
    # indir = "/mnt/ivm_server2_od/xiaowh/vendor/logo200"
    # tsvdataset_dir = "/mnt/gpu01_raid/data/logo200"
    # indir = "/mnt/ivm_server2_od/xiaowh/vendor/2k_new"
    # tsvdataset_dir = "/mnt/vigstandard/data/logo2k"
    ensure_directory(tsvdataset_dir)

    def gen_in_rows(fpath):
        for parts in tsv_reader(fpath):
            # parts: url, bboxes
            url = parts[0]
            bboxes = parts[1]
            yield hash_sha1(url), bboxes, url

    for split, fpath in all_split_fpath:
        outpath = op.join(tsvdataset_dir, split + ".tsv")
        urls_to_img_file_parallel(gen_in_rows(fpath), 2, [0, 1], outpath)

def draw_box_for_missing(phase, dataset_name, split, outdir, version=-1):
    from qd.process_tsv import upload_image_to_blob
    from qd.qd_common import calculate_iou

    # upload_image_to_blob(dataset_name, split)
    dataset = TSVDataset(dataset_name)
    log_file = op.join(outdir, "task_log.tsv")

    if phase == "upload":
        def gen_labels():
            url_iter = dataset.iter_data(split, t='key.url', version=version)
            label_iter = dataset.iter_data(split, t='label', version=version)
            for url_parts, label_parts in zip(url_iter, label_iter):
                assert(url_parts[0] == label_parts[0])
                yield label_parts[0], label_parts[1], url_parts[1]

        label_file = op.join(outdir, "labels.tsv")
        tsv_writer(gen_labels(), label_file)
        upload_dir = op.join(outdir, "upload")
        ensure_directory(upload_dir)
        task_files = generate_task_files("DrawBox", label_file, None, op.join(upload_dir, "relabel"),
                            None, num_tasks_per_hit=10, num_hp_per_hit=0,
                            num_hits_per_file=2000)

        task_group_id = 91761
        with open(log_file, 'a') as fp:
            for task_file in task_files:
                task_id = UhrsTaskManager.upload_task(task_group_id, task_file, num_judgment=1)
                fp.write("{}\t{}\t{}\n".format(task_group_id, task_id, task_file))

    if phase == "download":
        download_dir = op.join(outdir, "download")
        ensure_directory(download_dir)
        res_files = []
        # for parts in tsv_reader(log_file):
        #     task_group_id = int(parts[0])
        #     task_id = int(parts[1])
        #     fname = op.basename(parts[2])
        #     if UhrsTaskManager.is_task_completed(task_group_id, task_id):
        #         fpath = op.join(download_dir, fname)
        #         print(fpath)
        #         UhrsTaskManager.download_task(task_group_id, task_id, fpath)
        #         res_files.append(fpath)

        res_files = [op.join(download_dir, fname) for fname in os.listdir(download_dir)]
        url_ans_tsv = op.join(outdir, "url_bboxes.tsv")
        url2ans = analyze_draw_box_task(res_files, url_ans_tsv)

        key_url_iter = dataset.iter_data(split, t="key.url")
        key_rects_iter = dataset.iter_data(split, t='label', version=version)
        generate_info = []
        def gen_new_labels():
            for (key, url), (key1, str_rects) in zip(key_url_iter, key_rects_iter):
                assert(key == key1)
                old_rects = json.loads(str_rects)
                new_rects = url2ans[url]
                for bi in old_rects:
                    is_found = False
                    for bj in new_rects:
                        if bi["class"] == bj["class"] and calculate_iou(bi["rect"], bj["rect"]) > 0.5:
                            is_found = True
                            break
                    if not is_found:
                        generate_info.append(["remove", key, json.dumps(bi)])
                yield key, json_dump(new_rects)

        dataset.update_data(gen_new_labels(), split, t='label', generate_info=generate_info)

def download_logo2k():
    if os.name == 'nt':
        data_root = "//vigdgx02/raid_data/"
    else:
        data_root = 'data/'

    dataset_name = 'uhrs_label_logo2k'
    dirpath = op.join(data_root, dataset_name, 'uhrs')
    task_id_tsv = op.join(dirpath, 'task_ids_real_resubmit_0809.tsv')

    url_bboxes_fname = 'url_bboxes.tsv'
    url_bboxes_real_fname = 'url_bboxes_real.tsv'
    url_bboxes_proto_fname = 'uhr_bboxes_proto.tsv'
    old_dirs = [op.join(dirpath, '0920'), op.join(dirpath, '1017') ]
    cur_dir = op.join(dirpath, '1101')

    def get_task_id_from_downloaded_file(f):
        task_group_id, task_id, _ = op.basename(f).split('.', 2)
        assert task_group_id.isdigit() and task_id.isdigit()
        return task_group_id, task_id

    # download finished tasks
    if os.name == 'nt':
        task_download_dir = op.join(cur_dir, 'download')
        res_files = download_uhrs_by_task_ids([task_id_tsv], task_download_dir, use_cache=True,
                complete_only=False)
        existing_tasks = set(get_task_id_from_downloaded_file(f) for f in res_files)
        for old_dir in old_dirs:
            old_task_dl_dir = op.join(old_dir, 'download')
            for f in os.listdir(old_task_dl_dir):
                task_group_id, task_id = get_task_id_from_downloaded_file(f)
                if (task_group_id, task_id) not in existing_tasks:
                    res_files.append(op.join(old_task_dl_dir, f))
        logging.info('\n'.join(res_files))

        get_url_bboxes_from_uhrs_task_res(res_files, cur_dir, url_bboxes_real_fname,
            url_bboxes_proto_fname, url_bboxes_fname=url_bboxes_fname, start_time=None,
            remove_nopair_labels=False)
    else:
        def gen_in_rows(fpaths):
            for fpath in fpaths:
                for parts in tsv_reader(fpath):
                    # parts: url, bboxes
                    url = parts[0]
                    bboxes = parts[1]
                    yield hash_sha1(url), bboxes, url

        out_in_fpaths = [
            [op.join(data_root, dataset_name, 'train.tsv'), [op.join(cur_dir, url_bboxes_real_fname)]],
            [op.join(data_root, dataset_name, 'test.tsv'), [op.join(cur_dir, url_bboxes_proto_fname)]],
        ]

        for outpath, inpaths in out_in_fpaths:
            urls_to_img_file_parallel(gen_in_rows(inpaths), 2, [0, 1], outpath)

        from qd.process_tsv import populate_dataset_details
        populate_dataset_details(dataset_name, check_image_details=True)


if __name__ == "__main__":
    from qd.qd_common import init_logging
    init_logging()
    # scrape_keywords_to_search_images()
    # url_bboxes_to_tsvdataset()
    # outdir = "/mnt/ivm_server2_od/xiaowh/vendor/logo200/relabel/"
    # outdir = "//ivm-server2/IRIS/OD/xiaowh/vendor/logo200/relabel/"
    # draw_box_for_missing("download", "logo200", "test", outdir)

    # review_vendor_labels()
    # resubmit_tasks()
    # inspect_uploaded_tasks()
    download_logo2k()
