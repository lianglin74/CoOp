import collections
import json
import os
import os.path as op

import _init_paths
from generate_task import generate_task_files
from analyze_task import analyze_draw_box_task
from dataproc import scrape_image_parallel, scrape_image
from uhrs import UhrsTaskManager

from qd.qd_common import ensure_directory, json_dump
from qd.tsv_io import tsv_reader, tsv_writer, TSVDataset

def scrape_keywords_to_search_images():
    # dname = "logo200"
    dname = "2k_new"
    # ext = "png"
    ext = "jpg"
    img_type = "real"
    num_imgs_per_query = 80
    # rootdir = "//ivm-server2/IRIS/OD/xiaowh/vendor/"
    rootdir = "/mnt/ivm_server2_od/xiaowh/vendor/"
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
        url_ans_tsv = op.join(outdir, "url_bboxes.tsv")
        url2ans = analyze_draw_box_task(res_files, url_ans_tsv)
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
        for label in real_label2counts:
            if label in proto_label2counts:
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

def url_bboxes_to_tsvdataset():
    from dataproc import urls_to_img_file_parallel
    from qd.qd_common import hash_sha1, ensure_directory

    # indir = "/mnt/ivm_server2_od/xiaowh/vendor/logo200"
    # tsvdataset_dir = "/mnt/gpu01_raid/data/logo200"
    indir = "/mnt/ivm_server2_od/xiaowh/vendor/2k_new"
    tsvdataset_dir = "/mnt/vigstandard/data/logo2k"
    ensure_directory(tsvdataset_dir)
    split2fpath = {"test": op.join(indir, "url_bboxes_real.tsv"), "train": op.join(indir, "url_bboxes_proto.tsv")}

    def gen_in_rows(fpath):
        for parts in tsv_reader(fpath):
            # parts: url, bboxes
            url = parts[0]
            bboxes = parts[1]
            yield hash_sha1(url), bboxes, url

    for split in split2fpath:
        outpath = op.join(tsvdataset_dir, split + ".tsv")
        urls_to_img_file_parallel(gen_in_rows(split2fpath[split]), 2, [0, 1], outpath)

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


if __name__ == "__main__":
    from qd.qd_common import init_logging
    init_logging()
    # scrape_keywords_to_search_images()
    # url_bboxes_to_tsvdataset()
    outdir = "/mnt/ivm_server2_od/xiaowh/vendor/logo200/relabel/"
    # outdir = "//ivm-server2/IRIS/OD/xiaowh/vendor/logo200/relabel/"
    draw_box_for_missing("download", "logo200", "test", outdir)