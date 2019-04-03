import random
import re
from tsv_io import TSVDataset, tsv_reader, tsv_writer
from qd_common import write_to_file
from taxonomy import synset_to_noffset, LabelToSynset
import json
import os
import os.path as op
import logging

def ensure_dataset_sample(source_dataset, sample_label, sample_image, out_data):
    if op.exists(out_data):
        logging.info('skip to generate the samle data since it exists')
        return
    logging.info('start to generate the sample data')
    assert source_dataset.name == 'imagenet2012', 'only cls dataset is supp'
    assert sample_label > 0 and sample_label <= 1
    assert sample_image > 0 and sample_image <= 1
    random.seed(777)
    labels = source_dataset.load_labelmap()
    num_labels = len(labels)
    num_sampled_labels = int(num_labels * sample_label)
    assert num_sampled_labels > 0
    label_idx = range(num_labels)
    random.shuffle(label_idx)
    sampled_labels_idx = label_idx[:num_sampled_labels]
    train_rows = tsv_reader(source_dataset.get_train_tsv())
    train_images = [row for row in train_rows if int(row[1]) in sampled_labels_idx]
    for row in train_images:
        row[1] = str(sampled_labels_idx.index(int(row[1])))
    random.shuffle(train_images)
    tsv_writer(train_images[: int(len(train_images) * sample_image)],
            op.join(out_data, 'train.tsv'))
    # process the test set
    test_rows = tsv_reader(source_dataset.get_test_tsv_file())
    test_images = [row for row in test_rows if int(row[1]) in sampled_labels_idx]
    for row in test_images:
        row[1] = str(sampled_labels_idx.index(int(row[1])))
    tsv_writer(test_images, op.join(out_data, 'test.tsv'))
    # save teh label map
    sampled_labels = [labels[i] for i in sampled_labels_idx]
    write_to_file('\n'.join(sampled_labels), op.join(out_data, 'labelmap.txt'))

def dynamic_process_tsv(source_dataset, output_data_path,
        **kwargs):
    if 'dataset_ops' not in kwargs:
        kwargs['dataset_ops'] = []
    return process_dataset(source_dataset, output_data_path,
            kwargs['dataset_ops'])

def construct_low_shot(source_dataset, labels, num_train_images, shuffle_file):
    assert len(labels) == 1
    label_imageIdx = {}
    target_images = []
    rows = tsv_reader(source_dataset.get_train_tsv())
    shuffle = []
    for i, row in enumerate(rows):
        curr_labels = json.loads(row[1])
        if any([l['class'] == labels[0] for l in curr_labels]):
            target_images.append((i, row[0]))
        else:
            shuffle.append(i)
    assert len(target_images) >= num_train_images
    random.seed(777)
    random.shuffle(target_images)
    selected = target_images[:num_train_images]
    num_duplicate = (len(target_images) + num_train_images - 1) / num_train_images
    assert num_duplicate >= 1
    for d in range(num_duplicate):
        shuffle.extend([s[0] for s in selected])

    random.shuffle(shuffle)
    write_to_file('\n'.join(map(str, selected)), shuffle_file + '.selected')
    write_to_file('\n'.join(map(str, shuffle)), shuffle_file)

def process_dataset(source_dataset, output_root_data_path, dataset_ops):
    sources, source_labels = [], []
    source_shuffles, data_batch_weights = [], []
    labelmaps = [source_dataset.get_labelmap_file()]
    tree_files = [source_dataset.get_tree_file()]

    source = source_dataset.get_train_tsvs()
    sources.append(source)
    train_label_files = source_dataset.get_train_tsvs('label')
    if len(train_label_files) != 0 and all(op.isfile(f) for f in train_label_files):
        source_labels.append(train_label_files)
    shuffle_file = source_dataset.get_train_shuffle_file()
    def append_shuffle_file_if_valid(shuffle_file, source):
        if op.isfile(shuffle_file):
            if len(source) == 1:
                # we need to check if the shuffle file is the multi-version
                is_invalid = any(len(x) == 2 and int(x[0]) != 0 for x in
                        tsv_reader(shuffle_file))
                if not is_invalid:
                    source_shuffles.append(shuffle_file)
            else:
                source_shuffles.append(shuffle_file)
    append_shuffle_file_if_valid(shuffle_file, source)

    for i, operator in enumerate(dataset_ops):
        if operator['op'] == 'remove':
            # remove the current dataset. That is, we don't want to use the
            # current dataset at all
            sources = []
            source_labels = []
            data_batch_weights = []
            source_shuffles = []
            labelmaps = []
            tree_files = []
        elif operator['op'] == 'select_top':
            assert False, 'no longer support -> create a new dataset pls'
            # dataset_ops = [{'op':'select_top', 'num_top': 1}]
            # only select the first few training images. This is used to verify
            # if the training loss can be reduced if we only have a few
            # training samples.
            num_top = operator['num_top']
            shuffle_file = op.join(output_root_data_path, source_dataset.name,
                    'train_shuffle{}.txt'.format(num_top))
            write_to_file('\n'.join(map(str, range(num_top))), shuffle_file)
            assert len(sources) == 1
            assert len(source_shuffles) == 0
            source_shuffles.append(shuffle_file)
        elif operator['op'] == 'add':
            extra_dataset = TSVDataset(operator['name'])
            if operator['source'] == 'train':
                curr_source = extra_dataset.get_train_tsvs()
                curr_labels = extra_dataset.get_train_tsvs('label')
            elif operator['source'] == 'trainval':
                raise NotImplemented
                curr_source = extra_dataset.get_trainval_tsv()
            else:
                assert False
            #curr_labelmap = extra_dataset.load_labelmap()
            shuffle_file = extra_dataset.get_train_shuffle_file()
            source = curr_source
            append_shuffle_file_if_valid(shuffle_file, source)
            sources.append(curr_source)
            labelmaps.append(extra_dataset.get_labelmap_file())
            if len(curr_labels) == len(curr_source):
                source_labels.append(curr_labels)
            data_batch_weights.append(operator.get('weight', 1))
            tree_files.append(extra_dataset.get_tree_file())
        elif operator['op'] == 'sample':
            assert False, 'no longer support -> create a new dataset pls'
            # randomly sample the training data by a probability for all
            assert len(dataset_ops) == 1
            sample_label = operator['sample_label']
            sample_image = operator['sample_image']
            out_folder = op.join(output_root_data_path,
                    '{}_{}_{}'.format(source_dataset.name,
                        operator['sample_label'],
                        operator['sample_image']))
            ensure_dataset_sample(source_dataset, sample_label, sample_image, out_folder)
            sources = [op.join(out_folder, 'train.tsv')]
            labelmap = op.join(out_folder, 'labelmap.txt')
        elif operator['op'] == 'low_shot':
            assert False, 'no longer support -> create a new dataset pls'
            assert len(dataset_ops) == 1
            labels = operator['labels'].split(',')
            num_train = operator['num_train']
            shuffle_file = op.join(output_root_data_path, source_dataset.name,
                    'low_shot_shuffle_{}_{}.txt'.format('_'.join(labels),
                        num_train))
            source_shuffles.append(shuffle_file)
            if not op.isfile(shuffle_file):
                construct_low_shot(source_dataset, labels,
                        num_train, shuffle_file)
        elif operator['op'] == 'mask_background':
            assert False, 'no longer support -> create a new dataset pls'
            assert i == 0
            sources = []
            new_label_idx = operator['new_label_idx']
            target_folder = op.join(output_root_data_path,
                    '{}_{}_{}'.format(source_dataset.name,
                        '.'.join(map(str, operator['old_label_idx'])),
                        new_label_idx))
            if not op.exists(target_folder):
                old_labels = source_dataset.load_labelmap()
                num_label = len(old_labels)
                old_new = create_mask_label_map(operator['old_label_idx'],
                        new_label_idx, num_label)
                mask_background(source_dataset.get_train_tsv(),
                        op.join(target_folder, 'train.tsv'),
                        old_new)
                mask_background(source_dataset.get_test_tsv_file(),
                        op.join(target_folder, 'test.tsv'),
                        old_new)
                new_labels = [None] * (num_label - len(operator['old_label_idx']) + 1)
                new_labels[new_label_idx] = 'background'
                for curr_old_label_idx, curr_new_label_idx in enumerate(old_new):
                    if new_labels[curr_new_label_idx] is None:
                        if curr_new_label_idx != new_label_idx:
                            new_labels[curr_new_label_idx] = old_labels[curr_old_label_idx]
                write_to_file('\n'.join(new_labels), op.join(target_folder,
                    'labelmap.txt'))
            sources.append(op.join(target_folder, 'train.tsv'))
        else:
            assert False
    return {'sources': sources,
            'source_labels': source_labels,
            'source_shuffles': source_shuffles,
            'data_batch_weights': data_batch_weights,
            'labelmaps': labelmaps,
            'tree_files': tree_files}

def create_mask_label_map(old_label_idx,
        new_label_idx, num_label):
    old_new = [None] * num_label
    idx = 0
    for i in range(num_label):
        if i in old_label_idx:
            old_new[i] = new_label_idx
        else:
            if idx == new_label_idx:
                idx = idx + 1
            old_new[i] = idx
            idx = idx + 1

    return old_new

def mask_background(tsv_file,
                    out_file,
                    old_new):

    def gen_rows():
        rows = tsv_reader(tsv_file)
        for row in rows:
            row[1] = str(old_new[int(float(row[1]))])
            yield row

    tsv_writer(gen_rows(), out_file)

def is_noffset_list(labels):
    has = False
    for label in labels:
        if re.match('^n[0-9]{8}$', label) == None:
            return False
        else:
            return True

def map_label(source, labels, synset_tree, source_label):
    assert synset_tree
    all_idx = []

    if is_noffset_list(labels):
        noffsets = labels
    else:
        mapper = LabelToSynset()
        noffsets = [synset_to_noffset(mapper.convert(l)) for l in labels]
        for noffset in noffsets:
            assert len(synset_tree.root.search_nodes(name=noffset)) == 1

    label_noffset = { label:noffset for label, noffset in zip(labels, noffsets)}
    rows = tsv_reader(source)
    def convert_label(rows, labels, label_noffset):
        is_cls_set = False
        for i, row in enumerate(rows):
            if row[1].isdigit():
                infos = [{'class': label_noffset[labels[int(row[1])]],
                    'rect': [0,0,0,0]}]
                is_cls_set = True
            else:
                assert not is_cls_set
                infos = json.loads(row[1])
                for info in infos:
                    info['class'] = label_noffset[info['class']]
            if (i % 1000) == 0:
                logging.info(i)
            yield row[0], json.dumps(infos)
    tsv_writer(convert_label(rows, labels, label_noffset), source_label)

