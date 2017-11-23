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
from qd_common import write_to_yaml_file, load_from_yaml_file
from tsv_io import TSVDataset
from tsv_io import tsv_reader, tsv_writer

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

class DatasetSource(object):
    def __init__(self):
        pass

    def populate_info(self, root):
        pass

    def gen_tsv_rows(self, root):
        pass

class TSVDatasetSource(TSVDataset, DatasetSource):
    def __init__(self, name):
        super(TSVDatasetSource, self).__init__(name)
        self._noffset_count = {}
        self._noffset_to_label = {}
        self._type = None

    def populate_info(self, root):
        self._ensure_noffset_count()
        for node in root.iter_search_nodes():
            if node.noffset in self._noffset_count:
                if self._type == 'with_bb':
                    c = self._noffset_count[node.noffset]
                    logging.info('with_bb: {}: {}->{}'.format(self.name,
                        self._noffset_to_label[node.noffset],
                        c))
                    node.images_with_bb = node.images_with_bb + c
                            
                else:
                    assert self._type == 'no_bb'
                    c = self._noffset_count[node.noffset]
                    logging.info('no_bb: {}: {}->{}'.format(self.name,
                        self._noffset_to_label[node.noffset],
                        c))
                    node.images_no_bb = node.images_no_bb + c
                            

    def _ensure_noffset_count(self):
        populate_dataset_details(self.name)
        self._type = read_to_buffer(op.join(self._data_root,
            'type.txt')).strip()
        if len(self._noffset_count) == 0:
            noffsets = self.load_noffsets()
            self._noffset_count = {noffset: 0 for noffset in noffsets}
        else:
            return
        labelmap = self.load_labelmap()
        label_to_noffset = {label: noffset for label, noffset in zip(labelmap, noffsets)}
        if op.isfile(self.get_labelmap_of_noffset_file()):
            labelmap = load_list_file(self.get_labelmap_of_noffset_file())
        self._noffset_to_label = {noffset: label for label, noffset in zip(labelmap, noffsets)}

        def populate_file(tsv_file):
            if op.isfile(tsv_file):
                details = load_from_yaml_file(get_meta_file(tsv_file))
                for label in details['label_count']:
                    noffset = label_to_noffset[label]
                    self._noffset_count[noffset] += details['label_count'][label]
        populate_file(self.get_trainval_tsv())
        if not op.isfile(self.get_trainval_tsv()):
            populate_file(self.get_trainval_tsv())
        populate_file(self.get_test_tsv_file())

    def gen_tsv_rows(self, root, label_type):
        if label_type == self._type:
            if op.isfile(self.get_trainval_tsv()):
                rows = tsv_reader(self.get_trainval_tsv())
                for row in rows:
                    yield row
            elif op.isfile(self.get_train_tsv()):
                for row in tsv_reader(self.get_train_tsv()):
                    yield row

            if op.isfile(self.get_test_tsv_file()):
                for row in tsv_reader(self.get_test_tsv_file()):
                    yield row

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

def parse_args():
    parser = argparse.ArgumentParser(description='TSV Management')
    parser.add_argument('-t', '--type', help='what type it is: gen_tsv',
            type=str, required=True)
    parser.add_argument('-i', '--input', help='input',
            type=str, required=False)
    parser.add_argument('-o', '--output', help='output',
            type=str, required=False)
    return parser.parse_args()

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    process_tsv_main(**vars(args))

