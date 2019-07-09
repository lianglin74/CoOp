import base64
import collections
import json
import numpy as np
import os.path as op

from qd.tsv_io import TSVDataset, TSVFile, tsv_reader
from qd.qd_common import write_to_yaml_file
from qd_classifier.scripts import extract

def dedup(model_pth, dataset_name, split, version=-1, similar_thres=0.7, verbal=False):
    assert(op.isfile(model_pth))
    outdir = op.join(op.dirname(model_pth), 'eval')
    fea_file = op.join(outdir, "{}.{}.{}.{}.features.tsv".format(op.basename(model_pth), dataset_name, split, version))
    # extract_img_feature(model_pth, fea_file, dataset_name, split, version)

    label2rows = collections.defaultdict(list)
    for row_idx, parts in enumerate(tsv_reader(fea_file)):
        label = json.loads(parts[1])["class"]
        label2rows[label].append(row_idx)

    fea_tsv = TSVFile(fea_file)
    dup_imgkeys = set()
    generate_info = []
    for label, row_indices in label2rows.items():
        features = [fea_tsv.seek(i)[-1] for i in row_indices]
        dup_flags = pairwise_dedup(features, similar_thres)
        for i in range(len(features)):
            if dup_flags[i] != i:
                img1, str_bbox1 = fea_tsv.seek(row_indices[i])[:-1]
                img2, str_bbox2 = fea_tsv.seek(row_indices[dup_flags[i]])[:-1]
                assert(img1 != img2)
                dup_imgkeys.add(img1)
                generate_info.append(["dedup", img1, str_bbox1, img2, str_bbox2])
                if verbal:
                    print("detect duplication: {} {} and {} {}".format(img1, str_bbox1, img2, str_bbox2))

    if verbal:
        num_total_dups = len(dup_imgkeys)
        print(len(fea_tsv), num_total_dups, float(num_total_dups) / len(fea_tsv))

    dataset = TSVDataset(dataset_name)
    def gen_new_labels():
        for imgkey, str_rects in dataset.iter_data(split, t='label', version=version):
            if imgkey in dup_imgkeys:
                yield imgkey, "[]"
            else:
                yield imgkey, str_rects
    dataset.update_data(gen_new_labels(), split, t='label', generate_info=generate_info)

def extract_img_feature(model_pth, outfile, dataset_name, split, version):
    data_yaml_path = outfile + ".datacfg.yaml"
    dataset = TSVDataset(dataset_name)
    data_cfg = {"test":
        {
            "tsv": dataset.get_data(split),
            "label": dataset.get_data(split, t='label', version=version),
        }
    }
    write_to_yaml_file(data_cfg, data_yaml_path)
    extract.main([
            data_yaml_path,
            "--model", model_pth,
            "--output", outfile,
            "--enlarge_bbox", str(1),
            "--workers", str(4)
    ])

def pairwise_dedup(features, similar_thres):
    num_feas = len(features)
    dup_flags = list(range(num_feas))

    for i, fea in enumerate(features):
        if dup_flags[i] != i:
            continue
        normalized_fea_i = from_b64_to_fea(fea)
        for j in range(i+1, num_feas):
            if dup_flags[j] != j:
                continue
            normalized_fea_j = from_b64_to_fea(features[j])
            sim = cosine_similarity(normalized_fea_i, normalized_fea_j)
            if sim >= similar_thres:
                dup_flags[j] = i
    # is_dup = [dup_flags[i] != i for i in range(num_feas)]
    return dup_flags

def from_b64_to_fea(b64_fea, normalize=True):
    fea = base64.b64decode(b64_fea)
    fea = np.frombuffer(fea, dtype=np.float32)
    if normalize:
        norm = np.linalg.norm(fea)
        fea = fea / norm
    return fea

def cosine_similarity(fea1, fea2):
    return max(0, np.dot(fea1, fea2))

if __name__ == "__main__":
    model_path = "brand_output/brandsports_addlogo40syn/snapshot/model_best.pth.tar"
    dedup(model_path, "logo200", "train")
