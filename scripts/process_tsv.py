from qd_common import init_logging
import os.path as op
import argparse
import json
import logging
from taxonomy import gen_term_list
from taxonomy import gen_noffset
from taxonomy import load_all_tax, merge_all_tax
from taxonomy import LabelToSynset, synset_to_noffset
from taxonomy import populate_url_for_offset
from taxonomy import noffset_to_synset
from taxonomy import disambibuity_noffsets
from taxonomy import populate_cum_images
from taxonomy import child_parent_print_tree2
from taxonomy import create_markdown_url
from taxonomy import get_nick_name
from qd_common import write_to_yaml_file, load_from_yaml_file
from qd_common import read_to_buffer, load_list_file
from tsv_io import TSVDataset
from tsv_io import tsv_reader, tsv_writer
from qd_common import write_to_file
from qd_common import ensure_directory
import random
from tsv_io import get_meta_file
from tsv_io import extract_label
from tsv_io import create_inverted_tsv
import numpy as np
import shutil
import os
from qd_common import img_from_base64
from qd_common import yolo_old_to_new

def tsv_details(tsv_file):
    rows = tsv_reader(tsv_file)
    label_count = {}
    sizes = []
    for i, row in enumerate(rows):
        if (i % 1000) == 0:
            logging.info('get tsv details: {}-{}'.format(tsv_file, i))
        rects = json.loads(row[1])
        curr_labels = set(str(rect['class']) for rect in rects)
        for c in curr_labels:
            if c in label_count:
                label_count[c] = label_count[c] + 1
            else:
                label_count[c] = 1
        im = img_from_base64(row[2])
        sizes.append(im.shape[:2])
    min_size_count = sizes[0][0] * sizes[0][1]
    size_counts = [s[0] * s[1] for s in sizes]
    min_size = sizes[np.argmin(size_counts)]
    max_size = sizes[np.argmax(size_counts)]
    min_size = map(float, min_size)
    max_size = map(float, max_size)
    mean_size = (np.mean([s[0] for s in sizes]), 
            np.mean([s[1] for s in sizes]))
    mean_size = map(float, mean_size)
    
    return {'label_count': label_count, 
            'min_image_size': min_size, 
            'max_image_size': max_size, 
            'mean_image_size': mean_size}

def gen_tsv_from_labeling(input_folder, output_folder):
    fs = glob.glob(op.join(input_folder, '*'))
    labels = set()
    def gen_rows():
        for f in fs:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            yaml_file = op.splitext(f)[0] + '.yaml'
            if not op.isfile(yaml_file):
                logging.info('{} not exist'.format(yaml_file))
            with open(yaml_file, 'r') as fp:
                bb_labels = yaml.loads(fp.read())
            for bb_label in bb_labels:
                labels.add(bb_label['class'])
            with open(f, 'r') as fp:
                encoded_im = base64.b64encode(fp.read())
            yield op.basename(f), json.dumps(bb_labels), encoded_im

    tsv_writer(gen_rows(), op.join(output_folder, 'train.tsv'))
    write_to_file('\n'.join(labels), op.join(output_folder, 'labelmap.txt'))

class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' 
        self._fp = None
        self._lineidx = None
    
    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx) 

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]
    
    def _ensure_lineidx_loaded(self):
        if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
            generate_lineidx(self.tsv_file, self.lineidx)
        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')

class DatasetSource(object):
    def __init__(self):
        pass

    def populate_info(self, root):
        pass

    def gen_tsv_rows(self, root):
        pass

class TSVDatasetSource(TSVDataset, DatasetSource):
    def __init__(self, name, root=None):
        super(TSVDatasetSource, self).__init__(name)
        self._type = None
        self._root = root
        # the list of <datasetlabel, rootlabel>
        self._sourcelabel_targetlabel = None
        self._sourcelabel_to_targetlabels = None
        self._targetlabel_to_sourcelabels = None
        self._sourcelabel_to_imagecount = None
        self._split_label_idx = None # list of <split, label, idx>
        self._datasetlabel_to_splitidx = None
        self._initialized = False

    def populate_info(self, root):
        self._ensure_initialized()
        for node in root.iter_search_nodes():
            if root == node:
                continue
            if node.name in self._targetlabel_to_sourcelabels:
                sourcelabels = self._targetlabel_to_sourcelabels[node.name]
                for sourcelabel in sourcelabels:
                    c = self._datasetlabel_to_count[sourcelabel]
                    if self._type == 'with_bb':
                        logging.info('with_bb: {}: {}->{}'.format(self.name,
                            node.name, c))
                        node.images_with_bb = node.images_with_bb + c
                                
                    else:
                        assert self._type == 'no_bb'
                        logging.info('no_bb: {}: {}->{}'.format(self.name,
                            node.name, c))
                        node.images_no_bb = node.images_no_bb + c

    def _ensure_initialized(self):
        if self._initialized:
            return
        populate_dataset_details(self.name)
        splits = ['trainval', 'train', 'test']
        # check the type of the dataset
        for split in splits:
            label_tsv = self.get_data(split, 'label')
            if not op.isfile(label_tsv):
                continue
            for row in tsv_reader(label_tsv):
                rects = json.loads(row[1])
                if any(np.sum(r['rect']) > 1 for r in rects):
                    self._type = 'with_bb'
                else:
                    self._type = 'no_bb'
                break
        logging.info('identify {} as {}'.format(self.name, self._type))
        
        # list of <split, label, idx>
        self._split_label_idx = []
        for split in splits:
            logging.info('loading the inverted file: {}-{}'.format(self.name,
                split))
            inverted = self.load_inverted_label(split)
            label_idx = dict_to_list(inverted, 0)
            for label, idx in label_idx:
                self._split_label_idx.append((split, label, idx))

        self._datasetlabel_to_splitidx = list_to_dict(self._split_label_idx, 1)
        self._datasetlabel_to_count = {l: len(self._datasetlabel_to_splitidx[l]) for l in
                self._datasetlabel_to_splitidx}

        self._datasetlabel_to_rootlabel = self.get_label_mapper()
        self._initialized = True

    def get_label_mapper(self):
        root = self._root
        labelmap = self.load_labelmap()
        noffsets = self.load_noffsets()
        tree_noffsets = {node.noffset: node.name 
                for node in root.iter_search_nodes() 
                if node.noffset and node != root}
        tree_labels = {node.name: None
                for node in root.iter_search_nodes() 
                if node.noffset and node != root}

        sourcelabel_targetlabel = [] 

        result = {}
        for l, n in zip(labelmap, noffsets):
            if l in tree_labels:
                sourcelabel_targetlabel.append((l, l))
                result[l] = l
            elif n != '' and n in tree_noffsets:
                sourcelabel_targetlabel.append((l, tree_noffsets[n]))
                result[l] = tree_noffsets[n]

        self._sourcelabel_targetlabel = sourcelabel_targetlabel
        self._sourcelabel_to_targetlabels = list_to_dict(sourcelabel_targetlabel,
                0)
        self._targetlabel_to_sourcelabels = list_to_dict(sourcelabel_targetlabel,
                1)
    
        return result

    def select_tsv_rows(self, label_type):
        self._ensure_initialized()
        assert self._type is not None
        if label_type != self._type:
            return []
        result = []
        for datasetlabel in self._datasetlabel_to_splitidx:
            if datasetlabel in self._datasetlabel_to_rootlabel:
                split_idxes = self._datasetlabel_to_splitidx[datasetlabel]
                rootlabel = self._datasetlabel_to_rootlabel[datasetlabel]
                result.extend([(rootlabel, split, idx) for split, idx in
                    split_idxes])
        return result

    def gen_tsv_rows(self, root, label_type):
        selected_info = self.select_tsv_rows(root, label_type)
        mapper = self._datasetlabel_to_rootlabel
        for split, idx in selected_info:
            data_tsv = TSVFile(self.get_data(split))
            for i in idx:
                data_row = data_tsv.seek(i)
                rects = json.loads(data_row[1])
                convert_one_label(rects, mapper)
                assert len(rects) > 0
                data_row[1] = json.dumps(rects)
                yield data_row
        return

def populate_dataset_details(data):
    dataset = TSVDataset(data)

    def details_tsv(tsv_file):
        out_file = get_meta_file(tsv_file)
        if not op.isfile(out_file) and op.isfile(tsv_file):
            details = tsv_details(tsv_file)
            write_to_yaml_file(details, out_file)

    splits = ['trainval', 'train', 'test']
    for split in splits:
        details_tsv(dataset.get_data(split))

    # for each data tsv, generate the label tsv and the inverted file
    for split in splits:
        full_tsv = dataset.get_data(split)
        label_tsv = dataset.get_data(split, 'label')
        if not op.isfile(label_tsv) and op.isfile(full_tsv):
            extract_label(full_tsv, label_tsv)
        inverted = dataset.get_data(split, 'inverted.label')
        if not op.isfile(inverted) and op.isfile(label_tsv):
            create_inverted_tsv(label_tsv, inverted)
    
    # generate lineidx if it is not generated
    for split in splits:
        lineidx = dataset.get_lineidx(split)
        full_tsv = dataset.get_data(split)
        if not op.isfile(lineidx) and op.isfile(full_tsv):
            logging.info('no lineidx for {}. generating...'.format(split))
            generate_lineidx(full_tsv, lineidx)

    # generate the label map if there is no
    if not op.isfile(dataset.get_labelmap_file()):
        logging.info('no labelmap, generating...')
        labelmap = []
        for split in splits:
            label_tsv = dataset.get_data(split, 'label')
            if not op.isfile(label_tsv):
                continue
            for row in tsv_reader(label_tsv):
                labelmap.extend(set([rect['class'] for rect in
                    json.loads(row[1])]))
        assert len(labelmap) > 0, 'there are no labels!'
        labelmap = list(set(labelmap))
        logging.info('find {} labels'.format(len(labelmap)))
        write_to_file('\n'.join(labelmap), dataset.get_labelmap_file())

    if not op.isfile(dataset.get_noffsets_file()):
        logging.info('no noffset file. generating...')
        labelmap = dataset.load_labelmap()
        mapper = LabelToSynset()
        ambigous = []
        ss = [mapper.convert(l) for l in labelmap]
        for l, s in zip(labelmap, ss):
            if type(s) is list and len(s) > 1:
                d = create_info_for_ambigous_noffset(l, [synset_to_noffset(s1)
                    for s1 in s])
                ambigous.append(d)
        if len(ambigous) > 0:
            logging.info('ambigous term which has no exact noffset: {}'.format(
                dataset.name))
            write_to_yaml_file(ambigous, dataset.get_noffsets_file() +
                    '.ambigous.yaml')
        noffsets = []
        for s in ss:
            if type(s) is list:
                noffsets.append('')
            else:
                noffsets.append(synset_to_noffset(s))
        write_to_file('\n'.join(noffsets), dataset.get_noffsets_file())

    if not op.isfile(dataset.get_labelmap_of_noffset_file()):
        noffsets = dataset.load_noffsets()
        all_line = []
        for noffset in noffsets:
            if len(noffset) == 0:
                all_line.append('unkown')
            else:
                s = noffset_to_synset(noffset)
                all_line.append(get_nick_name(s))
        write_to_file('\n'.join(all_line),
                dataset.get_labelmap_of_noffset_file())

def initialize_images_count(root):
    for node in root.iter_search_nodes():
        node.add_feature('images_with_bb', 0)
        node.add_feature('images_no_bb', 0)

def convert_one_label(rects, label_mapper):
    to_remove = []
    for rect in rects:
        if rect['class'] in label_mapper:
            rect['class'] = label_mapper[rect['class']]
        else:
            to_remove.append(rect)
    for t in to_remove:
        rects.remove(t)

def convert_label(label_tsv, idx, label_mapper):
    '''
    '''
    tsv = TSVFile(label_tsv)
    result = None
    for i in idx:
        row = tsv.seek(i)
        rects = json.loads(row[1])
        if result is None:
            result = [len(row) * ['d']] * tsv.num_rows()
        to_remove = []
        for rect in rects:
            if rect['class'] in label_mapper:
                rect['class'] = label_mapper[rect['class']]
            else:
                to_remove.append(rect)
        for t in to_remove:
            rects.remove(t)
        assert len(rects) > 0
        row[1] = json.dumps(rects)
        result[i] = row
    return result

def create_info_for_ambigous_noffset(name, noffsets):
    definitions = [str(noffset_to_synset(n).definition()) for n in noffsets]
    de = [{'noffset': n, 'definition': d} for n, d in zip(noffsets,
            definitions)]
    d = {'name': name,
            'definitions': de,
            'noffset': None,
            'markdown_url': create_markdown_url(noffsets)}
    return d

def build_taxonomy_impl(taxonomy_folder, **kwargs):
    random.seed(777)
    dataset_name = kwargs.get('data', 
            op.basename(taxonomy_folder))
    overall_dataset = TSVDataset(dataset_name)
    if op.isfile(overall_dataset.get_labelmap_file()):
        logging.info('ignore to build taxonomy since {} exists'.format(
            overall_dataset.get_labelmap_file()))
        return
    all_tax = load_all_tax(taxonomy_folder)
    tax = merge_all_tax(all_tax)
    initialize_images_count(tax.root)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k_448', tax.root)
    if op.isfile(imagenet22k.get_labelmap_file()):
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('there is no imagenet22k_448 dataset to help identify the noffset')
    populate_url_for_offset(tax.root)

    ambigous_noffset_file = op.join('./output/', 'ambigous_noffsets',
            dataset_name + '.yaml')
    output_ambigous_noffsets(tax.root, ambigous_noffset_file)
    
    data_sources = []
    
    datas = kwargs.get('datas', ['voc20', 'coco2017', 'imagenet3k_448',
        'crawl_office_v2', 'crawl_office_v1'])
    logging.info('extract the images from: {}'.format(','.join(datas)))

    for d in datas:
        data_sources.append(TSVDatasetSource(d, tax.root))
    
    for s in data_sources:
        s.populate_info(tax.root)

    populate_cum_images(tax.root)

    labels, child_parents = child_parent_print_tree2(tax.root, 'name')

    label_map_file = overall_dataset.get_labelmap_file() 
    write_to_file('\n'.join(map(lambda l: l.encode('utf-8'), labels)), 
            label_map_file)

    out_dataset = {'with_bb': TSVDataset(dataset_name + '_with_bb'),
            'no_bb': TSVDataset(dataset_name + '_no_bb')}

    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_labelmap_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(label_map_file, target_file)

    logging.info('cum_images_with_bb: {}'.format(tax.root.cum_images_with_bb))
    logging.info('cum_images_no_bb: {}'.format(tax.root.cum_images_no_bb))

    # dump the tree to yaml format
    dest = op.join(overall_dataset._data_root, tax.root.name + '.yaml')
    d = tax.dump()
    write_to_yaml_file(d, dest)

    # write the simplified version of the tree
    dest = op.join(overall_dataset._data_root, tax.root.name + '.simple.yaml')
    write_to_yaml_file(tax.dump(['images_with_bb']), dest)

    tree_file = overall_dataset.get_tree_file()
    write_to_file('\n'.join(['{} {}'.format(c.encode('utf-8'), p) for c, p in child_parents]), 
            tree_file)

    #leaves_should_have_images(tax.root, 200)
    
    def gen_rows(label_type):
        for s in data_sources:
            for i, row in enumerate(s.gen_tsv_rows(tax.root, label_type)):
                if (i % 1000) == 0:
                    logging.info('gen-rows: {}-{}-{}'.format(s.name, label_type, i))
                yield row
    
    copy_rows = False
    if copy_rows:
        # write trainval.tsv
        for label_type in out_dataset:
            tsv_file = out_dataset[label_type].get_trainval_tsv()
            if op.isfile(tsv_file):
                continue
            tmp_tsv_file = 'tmp.' + op.basename(tsv_file)
            tmp_tsv_file = op.join(op.dirname(tsv_file), tmp_tsv_file)
            if not op.isfile(tsv_file) or True:
                tsv_writer(gen_rows(label_type), tmp_tsv_file)
            logging.info('shuffling {}'.format(tmp_tsv_file))
            rows = tsv_shuffle_reader(tmp_tsv_file)
            tsv_writer(rows, tsv_file)
            logging.info('remove the unshuffled: {}'.format(tmp_tsv_file))
            os.remove(tmp_tsv_file)
            os.remove(op.splitext(tmp_tsv_file)[0] + '.lineidx')

        # split into train and test
        for label_type in out_dataset:
            dataset = out_dataset[label_type]
            trainval_split(dataset, 50)
        
        for label_type in out_dataset:
            populate_dataset_details(out_dataset[label_type].name)
    else:
        # get the information of all train val
        train_vals = []
        ldtsi = []
        logging.info('collecting all candidate images')
        for label_type in out_dataset:
            for dataset in data_sources:
                split_idxes = dataset.select_tsv_rows(label_type)
                for rootlabel, split, idx in split_idxes:
                    ldtsi.append((rootlabel, dataset, label_type, split, idx))
        # split into train val
        num_test = 50
        logging.info('splitting the images into train and test')
        # group by label_type
        t_to_ldsi = list_to_dict(ldtsi, 2)
        train_ldtsi = [] 
        test_ldtsi = []
        for label_type in t_to_ldsi:
            ldsi= t_to_ldsi[label_type]
            l_to_dsi = list_to_dict(ldsi, 0)
            for rootlabel in l_to_dsi:
                dsi = l_to_dsi[rootlabel]
                if len(dsi) < num_test:
                    logging.info('rootlabel={}; label_type={}->less than {} images'.format(
                        rootlabel, label_type, len(dsi)))
                assert len(dsi) > 1
                curr_num_test = min(num_test, int(len(dsi) / 2))
                random.shuffle(dsi)
                test_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i
                    in dsi[:curr_num_test]])
                train_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i 
                    in dsi[curr_num_test:]])

        logging.info('creating the train data')
        t_to_ldsi = list_to_dict(train_ldtsi, 2)
        train_ldtsik = []
        shuffle_idx = []
        for label_type in t_to_ldsi:
            ldsi = t_to_ldsi[label_type]
            d_to_lsi = list_to_dict(ldsi, 1)
            k = 0
            sources = []
            for dataset in d_to_lsi:
                lsi = d_to_lsi[dataset]
                s_li = list_to_dict(lsi, 1)
                for split in s_li:
                    li = s_li[split]
                    idx_to_l = list_to_dict(li, 1)
                    idx = idx_to_l.keys()
                    # link the data tsv
                    source = dataset.get_data(split)
                    out_split = 'train{}'.format(k)
                    train_ldtsik.extend([(l, dataset, label_type, split, i,
                        k) for l, i in li])
                    k = k + 1
                    dest = out_dataset[label_type].get_data(
                            out_split)
                    sources.append(dest)
                    if op.islink(dest):
                        os.remove(dest)
                    os.symlink(op.relpath(source, op.dirname(dest)), dest)
                    # link the lineidx
                    source = dataset.get_lineidx(split)
                    dest = out_dataset[label_type].get_lineidx(out_split)
                    if op.islink(dest):
                        os.remove(dest)
                    os.symlink(op.relpath(source, op.dirname(dest)), dest)
                    # create the label tsv
                    logging.info('converting labels: {}-{}'.format(
                        dataset.name, split))
                    converted_label = convert_label(dataset.get_data(split, 'label'),
                            idx, dataset._datasetlabel_to_rootlabel)
                    tsv_writer(converted_label,
                            out_dataset[label_type].get_data(out_split, 'label'))
            write_to_file('\n'.join(sources),
                    out_dataset[label_type].get_data('trainX'))
        logging.info('duplicating or removing the train images')
        # for each label, let's duplicate the image or remove the image
        max_image = 1000
        min_image = 200
        label_to_dtsik = list_to_dict(train_ldtsik, 0)
        for label in label_to_dtsik:
            dtsik = label_to_dtsik[label]
            if len(dtsik) > max_image:
                # first remove the images with no bounding box
                num_remove = len(dtsik) - max_image
                type_to_dsik = list_to_dict(dtsik, 1)
                if 'no_bb' in type_to_dsik:
                    dsik = type_to_dsik['no_bb']
                    if num_remove >= len(dsik):
                        # remove all this images
                        del type_to_dsik['no_bb']
                        num_remove = num_remove - len(dsik)
                    else:
                        random.shuffle(dsik)
                        type_to_dsik['no_bb'] = dsik[: len(dsik) - num_remove]
                        num_remove = 0
                if num_remove > 0:
                    assert 'with_bb' in type_to_dsik
                    dsik = type_to_dsik['with_bb']
                    random.shuffle(dsik)
                    assert len(dsik) > num_remove
                    type_to_dsik['with_bb'] = dsik[: len(dsik) - num_remove]
                    num_remove = 0
                dtsik = dict_to_list(type_to_dsik, 1)
            elif len(dtsik) < min_image:
                num_duplicate = int(np.ceil(float(min_image) / len(dtsik)))
                logging.info('duplicate images for label of {}: {}->{}, {}'.format(
                    label, len(dtsik), min_image, num_duplicate))
                dtsik = num_duplicate * dtsik
            label_to_dtsik[label] = dtsik
        logging.info('# train instances before duplication: {}'.format(len(train_ldtsik)))
        train_ldtsik = dict_to_list(label_to_dtsik, 0)
        logging.info('# train instances after duplication: {}'.format(len(train_ldtsik)))

        logging.info('saving the shuffle file')
        type_to_ldsik = list_to_dict(train_ldtsik, 2)
        for label_type in type_to_ldsik:
            ldsik = type_to_ldsik[label_type]
            random.shuffle(ldsik)
            shuffle_str = '\n'.join(['{}\t{}'.format(k, i) for l, d, s, i, k in
                ldsik])
            write_to_file(shuffle_str,
                    out_dataset[label_type].get_shuffle_file('train'))

        logging.info('writing the test data')
        t_to_ldsi = list_to_dict(test_ldtsi, 2)
        for label_type in t_to_ldsi:
            def gen_test_rows():
                ldsi = t_to_ldsi[label_type]
                d_to_lsi = list_to_dict(ldsi, 1)
                for dataset in d_to_lsi:
                    lsi = d_to_lsi[dataset]
                    s_to_li = list_to_dict(lsi, 1)
                    for split in s_to_li:
                        li = s_to_li[split]
                        idx = list_to_dict(li, 1).keys()
                        tsv = TSVFile(dataset.get_data(split))
                        for i in idx:
                            row = tsv.seek(i)
                            rects = json.loads(row[1])
                            convert_one_label(rects, 
                                    dataset._datasetlabel_to_rootlabel)
                            assert len(rects) > 0
                            row[1] = json.dumps(rects)
                            row[0] = '{}_{}_{}'.format(dataset.name,
                                    split, row[0])
                            yield row
            tsv_writer(gen_test_rows(), 
                    out_dataset[label_type].get_test_tsv_file())

def dict_to_list(d, idx):
    result = []
    for k in d:
        vs = d[k]
        for v in vs:
            try:
                r = []
                # if v is a list or tuple
                r.extend(v[:idx])
                r.append(k)
                r.extend(v[idx: ])
            except TypeError:
                r = []
                if idx == 0:
                    r.append(k)
                    r.append(v)
                else:
                    assert idx == 1
                    r.append(v)
                    r.append(k)
            result.append(r)
    return result

def list_to_dict(l, idx):
    result = {}
    for x in l:
        if x[idx] not in result:
            result[x[idx]] = []
        y = x[:idx] + x[idx + 1:]
        if len(y) == 1:
            y = y[0]
        result[x[idx]].append(y)
    return result

def output_ambigous_noffsets(root, ambigous_noffset_file):
    ambigous = []
    for node in root.iter_search_nodes():
        if hasattr(node, 'noffsets') and node.noffset is None:
            noffsets = node.noffsets.split(',')
            definitions = [str(noffset_to_synset(n).definition()) for n in noffsets]
            de = [{'noffset': n, 'definition': d} for n, d in zip(noffsets,
                    definitions)]
            d = {'name': node.name,
                    'definitions': de,
                    'noffset': None,
                    'markdown_url': node.markdown_url}
            ambigous.append(d)
    if len(ambigous) > 0:
        logging.info('output ambigous terms to {}'.format(ambigous_noffset_file))
        write_to_yaml_file(ambigous, ambigous_noffset_file)
    else:
        logging.info('Congratulations on no ambigous terms.')

def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein,'r') as tsvin, open(idxout,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0;
        while fpos!=fsize:
    	    tsvout.write(str(fpos)+"\n");
            tsvin.readline()
            fpos = tsvin.tell();

def output_ambigous_noffsets_main(tax_input_folder, ambigous_file_out):
    all_tax = load_all_tax(tax_input_folder)
    tax = merge_all_tax(all_tax)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k')
    if op.isfile(imagenet22k.get_labelmap_file()):
        logging.info('remove the noffset if it is not in imagenet22k')
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('no imagenet22k data used to help remove noffset ambiguities')

    populate_url_for_offset(tax.root)

    output_ambigous_noffsets(tax.root, ambigous_file_out)

def standarize_crawled(tsv_input, tsv_output):
    rows = tsv_reader(tsv_input)
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info(i)
            image_str = row[-1]
            image_label = row[0]
            rects = [{'rect': [0, 0, 0, 0], 'class': image_label}]
            image_name = '{}_{}'.format(op.basename(tsv_input), i)
            yield image_name, json.dumps(rects), image_str
    tsv_writer(gen_rows(), tsv_output)

def process_tsv_main(**kwargs):
    if kwargs['type'] == 'gen_tsv':
        input_folder = kwargs['input']
        output_folder = kwargs['ouput']
        gen_tsv_from_labeling(input_folder, output_folder)
    elif kwargs['type'] == 'gen_term_list':
        tax_folder = kwargs['input']
        term_list = kwargs['output']
        gen_term_list(tax_folder, term_list)
    elif kwargs['type'] == 'gen_noffset':
        tax_input_folder = kwargs['input']
        tax_output_folder = kwargs['output']
        gen_noffset(tax_input_folder, tax_output_folder)
    elif kwargs['type'] == 'ambigous_noffset':
        tax_input_folder = kwargs['input']
        ambigous_file_out = kwargs['output']
        output_ambigous_noffsets_main(tax_input_folder, ambigous_file_out)
    elif kwargs['type'] == 'standarize_crawled':
        tsv_input = kwargs['input']
        tsv_output = kwargs['output']
        standarize_crawled(tsv_input, tsv_output)
    elif kwargs['type'] == 'taxonomy_to_tsv':
        taxonomy_folder = kwargs['input']
        build_taxonomy_impl(taxonomy_folder)
    elif kwargs['type'] == 'yolo_model_convert':
        old_proto = kwargs['prototxt']
        old_model = kwargs['model']
        new_model = kwargs['output']
        yolo_old_to_new(old_proto, old_model, new_model)
    else:
        logging.info('unknown task {}'.format(kwargs['type']))

def parse_args():
    parser = argparse.ArgumentParser(description='TSV Management')
    parser.add_argument('-t', '--type', help='what type it is: gen_tsv',
            type=str, required=True)
    parser.add_argument('-i', '--input', help='input',
            type=str, required=False)
    parser.add_argument('-p', '--prototxt', help='proto file',
            type=str, required=False)
    parser.add_argument('-m', '--model', help='model file',
            type=str, required=False)
    parser.add_argument('-o', '--output', help='output',
            type=str, required=False)
    parser.add_argument('-d', '--datas', 
            nargs='*',
            help='which data are used for taxonomy_to_tsv',
            type=str, 
            required=False)
    return parser.parse_args()

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    process_tsv_main(**vars(args))

