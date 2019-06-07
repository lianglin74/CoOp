import json
import os

import _init_paths

from qd.qd_common import ensure_directory, json_dump
from qd.tsv_io import tsv_reader, tsv_writer

def scrape_keywords_to_search_images():
    import collections
    import os.path as op
    from dataproc import scrape_image_parallel, scrape_image
    from generate_task import generate_task_files
    from analyze_task import analyze_draw_box_task
    from uhrs import UhrsTaskManager

    dname = "logo200"
    # dname = "2k_new"
    ext = "png"
    num_imgs_per_query = 80
    rootdir = "//ivm-server2/IRIS/OD/xiaowh/vendor/"
    # rootdir = "/mnt/ivm_server2_od/xiaowh/vendor/"
    # keywords_file = r"C:\Users\xiaowh\OneDrive - Microsoft\share\filtered_labels_to_add.tsv"
    keywords_file = op.join(rootdir, dname, "2k_labels_to_add.txt")
    outfile = op.join(rootdir, dname, "scraped_urls_canonical_{}.tsv".format(ext))
    filtered_fname1 = "canonical_{}_dedup_scraped_urls1.tsv".format(ext)
    filtered_fname2 = "canonical_{}_dedup_scraped_urls2.tsv".format(ext)
    outdir = op.join(rootdir, dname)
    log_file = "task_ids.tsv"

    out_real_label_file = "url_bboxes_real.tsv"
    out_proto_label_file = "url_bboxes_proto.tsv"
    phase = "download"

    # ranked_list_file = op.join(rootdir, dname, "2k_ranked_logo_list.txt")
    # blacklist_file = op.join(rootdir, dname, "no_found_logos.txt")
    # blacklist_labels_set = set(p[0] for p in tsv_reader(blacklist_file))
    # ranked_labels = [p[0] for p in tsv_reader(ranked_list_file) if p[0] not in blacklist_labels_set]

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
                label2keywords[cur_term].append(cur_term + " logo")
            else:
                assert cur_term is not None
                assert parts[brand_col_idx] == ""
                assert(parts[img_type_col_idx] == "real image")
                # !!!!HACK: do not include real images for this time
                continue
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
                outdata1.extend(label2parts[label])

        tsv_writer(outdata1, op.join(outdir, filtered_fname1))
        tsv_writer(outdata2, op.join(outdir, filtered_fname2))

    if phase == "upload":
        for label_file in [filtered_fname1, filtered_fname2]:
            ensure_directory(op.join(outdir, "upload"))
            outbase = op.join(outdir, "upload", label_file)
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


if __name__ == "__main__":
    scrape_keywords_to_search_images()