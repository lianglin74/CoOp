import collections
import json
import os

from evaluation.eval_utils import DetectionFile
from qd.process_tsv import populate_dataset_details
from qd.qd_common import load_from_yaml_file
from qd.taxonomy import Taxonomy
from qd.tsv_io import tsv_reader, tsv_writer, TSVDataset


def sample_from_dataset(in_dataset_name, insplit, out_dataset_name, outsplit,
                        inversion=-1, out_num_imgs=1000, tax_tree_file=None):
    """
    Creates a new dataset by sampleing from in_dataset
    The newly created dataset will contain at most [out_num_imgs] images
    Class names will be mapped according to tax_tree_file, classes not in taxonomy
    will not appear in the new dataset.
    """
    dataset = TSVDataset(in_dataset_name)
    out_data_root = os.path.join(os.path.split(dataset._data_root)[0], out_dataset_name)
    if os.path.exists(out_data_root):
        raise Exception("{} already exists".format(out_data_root))
    os.mkdir(out_data_root)
    outfile = os.path.join(out_data_root, "{}.tsv".format(outsplit))
    sample_images(in_dataset_name, insplit, outfile, tax_tree_file=tax_tree_file,
                  version=inversion, target_num_imgs=out_num_imgs)
    populate_dataset_details(out_dataset_name, check_image_details=True)


def sample_images(dataset_name, split, outfile, target_num_imgs=1000, tax_tree_file=None, version=-1):
    """
    Sample images from dataset to cover as many classes as possible
    At most target_num_imgs will be returned
    If tax_tree_file (root.yaml) is provided, only classes in taxonomy will be returned, others will be omitted
    outfile: tsv file, [imgkey]\t[list of bboxes]\t[b64_image]
    """
    displayname_dict = None
    if tax_tree_file:
        displayname_dict = {}
        tax = Taxonomy(load_from_yaml_file(tax_tree_file))
        for n in tax.root.iter_search_nodes():
            if n == tax.root:
                continue
            # name in the specified dataset
            cur_names = getattr(n, dataset_name, n.name)
            for cur_name in cur_names.split(','):
                displayname_dict[cur_name] = n.name

    class_count = {}
    if displayname_dict:
        # class: num_pos_labels, num_neg_labels
        class_count = {displayname_dict[k]: [0, 0] for k in displayname_dict}

    def get_display_name(name):
        is_neg = 1 if name.startswith('-') else 0
        class_name = name[1:] if is_neg else name
        if displayname_dict:
            if class_name not in displayname_dict:
                return None, -1
            else:
                class_name = displayname_dict[class_name]
        return class_name, is_neg

    dataset = TSVDataset(dataset_name)
    img_label_dict = collections.defaultdict(list)
    for imgkey, coded_rects in dataset.iter_data(split, 'label', version=version):
        gt_bboxes = json.loads(coded_rects)
        for b in gt_bboxes:
            class_name, is_neg = get_display_name(b["class"])
            if not class_name:
                continue
            if class_name not in class_count:
                class_count[class_name] = [0, 0]
            b["class"] = '-' + class_name if is_neg>0 else class_name
            img_label_dict[imgkey].append(b)
            class_count[class_name][is_neg] += 1

    min_num_labels_per_class = 50
    # imgkey, num of labels
    img_label_list = [(k, len(img_label_dict[k])) for k in img_label_dict]
    img_label_list = sorted(img_label_list, key=lambda p:p[1])
    # the image keys to keep
    keep = set(p[0] for p in img_label_list)
    # start removing images that have less labels
    while len(keep) > target_num_imgs:
        for imgkey, _ in img_label_list:
            # already removed
            if imgkey not in keep:
                continue
            # reached target number
            if len(keep) <= target_num_imgs:
                break
            need_keep = False
            for bbox in img_label_dict[imgkey]:
                # skip negative labels
                if bbox["class"].startswith('-'):
                    continue
                if class_count[bbox["class"]][0] <= min_num_labels_per_class:
                    need_keep = True
                    break
            if not need_keep:
                keep.remove(imgkey)
                for bbox in img_label_dict[imgkey]:
                    is_neg = 1 if bbox["class"].startswith('-') else 0
                    cname = bbox["class"][1:] if is_neg>0 else bbox["class"]
                    class_count[cname][is_neg] -= 1
        min_num_labels_per_class -= 1

    labels = [(k, class_count[k][0], class_count[k][1]) for k in class_count if class_count[k][0]>0]
    labels = sorted(labels, key=lambda p: p[1], reverse=True)
    print("get {} images, covering {} classes".format(len(keep), len(labels)))
    nprint = min(len(labels), 3)
    print(' '.join(["{}(pos:{}, neg:{})".format(labels[i][0], labels[i][1], labels[i][2]) for i in range(nprint)+range(-nprint, 0)]))

    outdata = []
    for imgkey, _, coded_img in dataset.iter_data(split):
        if imgkey in keep:
            outdata.append([imgkey, json.dumps(img_label_dict[imgkey]), coded_img])
    tsv_writer(outdata, outfile)
