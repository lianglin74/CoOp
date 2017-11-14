import logging
from collections import OrderedDict
from nltk.corpus import wordnet as wn
import yaml
from ete2 import Tree
import glob
import os.path as op
from qd_common import write_to_file, read_to_buffer
from qd_common import init_logging, ensure_directory
from qd_common import load_from_yaml_file

def dump_tree(root):
    result = OrderedDict()
    for f in root.features:
        if f in ['support', 'name', 'dist', 'synset']:
            continue
        result[f] = root.__getattribute__(f)
    result[root.name] = []
    for c in root.children:
        result[root.name].append(dump_tree(c))
    return result

def synset_to_noffset(synset):
    return '{}{:0>8}'.format(synset.pos(), synset.offset())

def synonym_list():
    p = []
    p.append(('airplane', 'aeroplane'))
    #p.append(('tv', 'tvmonitor'))
    p.append(('motorcycle', 'motorbike'))
    p.append(('couch', 'sofa'))
    return p

def synonym():
    p = synonym_list()
    result = {}
    for a, b in p:
        result[a] = b
        result[b] = a
    return result

def noffset_to_synset(noffset):
    return wn.synset_from_pos_and_offset(noffset[0], int(noffset[1:]))

class LabelToSynset(object):
    def __init__(self):
        self._white_list = {'apple': wn.synset_from_pos_and_offset('n', 7739125),
                    'banana': wn.synset_from_pos_and_offset('n', 7753592),
                    'bear': wn.synset_from_pos_and_offset('n', 2131653),
                    'bed': wn.synset_from_pos_and_offset('n', 2818832),
                    'bench': wn.synset_from_pos_and_offset('n', 2828884),
                    'device': wn.synset_from_pos_and_offset('n', 3183080),
                    'chair': wn.synset_from_pos_and_offset('n', 3001627),
                    'pen': wn.synset_from_pos_and_offset('n', 3906997),
                    'marker': wn.synset_from_pos_and_offset('n', 3722007),
                    'book': wn.synset_from_pos_and_offset('n', 6410904),
                    'coke': wn.synset_from_pos_and_offset('n', 7928696),
                    'ring': wn.synset_from_pos_and_offset('n', 4092609),
                    'tv': wn.synset_from_pos_and_offset('n', 6277280),
                    'television': wn.synset_from_pos_and_offset('n', 6277280),
                    'projector': wn.synset_from_pos_and_offset('n', 4009552),
                    'telephone': wn.synset_from_pos_and_offset('n', 4401088),
                    'dish': wn.synset_from_pos_and_offset('n', 3206908),
                    'monitor': wn.synset_from_pos_and_offset('n', 3782190),
                    'eyeglasses': wn.synset_from_pos_and_offset('n', 4272054),
                    'sun glasses': wn.synset_from_pos_and_offset('n', 4356056),
                    'pencil': wn.synset_from_pos_and_offset('n', 3908204),
                    }
        self._update_by_name_map('./aux_data/label_to_noffset')
        s = synonym()
        for k1 in s:
            k2 = s[k1]
            if k1 in self._white_list:
                if k2 in self._white_list:
                    assert self._white_list[k1] == self._white_list[k2]
                else:
                    self._white_list[k2] = self._white_list[k1]
            elif k2 in self._white_list:
                self._white_list[k1] = self._white_list[k2]

        labels = self._white_list.keys()
        for label in labels:
            anchor = self._white_list[label]
            equal_ls = self._equal_labels(label)
            for l in equal_ls:
                if l in self._white_list:
                    assert self._white_list[l] == anchor 
                else:
                    self._white_list[l] = anchor 

    def convert(self, label):
        if len(label) == 9 and label[0] == 'n':
            return noffset_to_synset(label)

        label = label.lower()

        labels = self._equal_labels(label)
        for label in labels:
            if label in self._white_list:
                return self._white_list[label]
        
        result  = []
        for label in labels:
            sss = [ss for ss in wn.synsets(label) if ss.pos() == 'n']
            result.extend(sss)
        if len(result) == 1:
            return result[0]
        return result

    def _equal_labels(self, label):
        r = [label]
        if ' ' in label:
            r += [label.replace(' ', '_'), label.replace(' ', '')]
        return r

    def populate_noffset(self, root):
        if not hasattr(root, 'noffset') or root.noffset is None:
            s = self.convert(root.name)
            if s is not None and type(s) is not list:
                root.add_feature('noffset', synset_to_noffset(s))
            else:
                root.add_feature('noffset', None)
                if type(s) is list and len(s) > 1:
                    #root.add_feature('noffsets', ','.join(
                        #['[{0}](http://www.image-net.org/synset?wnid={0})'.format(
                            #synset_to_noffset(o)) for o in s]))
                    root.add_feature('noffsets', ','.join([synset_to_noffset(o) for o in
                        s]))
                else:
                    logging.info('cannot find {}'.format(root.name.encode('UTF-8')))
        for c in root.children:
            self.populate_noffset(c)

    def _update_by_name_map(self, folder):
        for yaml_file in glob.glob(op.join(folder, '*.yaml')):
            logging.info('loadding {}'.format(yaml_file))
            wl = load_from_yaml_file(yaml_file)
            for d in wl:
                if d['name'] in self._white_list:
                    assert synset_to_noffset(self._white_list[d['name']]) == \
                            d['noffset']
                elif d['noffset'] != None:
                    self._white_list[d['name']] = noffset_to_synset(d['noffset'])

def populate_url_for_offset(root):
    if not hasattr(root, 'noffset'):
        populate_noffset(root)
    if root.noffset:
        root.add_feature('url',
                'http://www.image-net.org/synset?wnid={}'.format(root.noffset))
    if hasattr(root, 'noffsets'):
        noffsets = root.noffsets.split(',')
        urls = ['[{0}](http://www.image-net.org/synset?wnid={0})'.format(noffset) for
                noffset in noffsets]
        root.add_feature('markdown_url', ','.join(urls))
    for c in root.children:
        populate_url_for_offset(c)

def populate_noffset(root):
    mapper = LabelToSynset()
    mapper.populate_noffset(root)

def disambibuity_noffsets(root, keys):
    if hasattr(root, 'noffsets'):
        noffsets = root.noffsets.split(',')
        exists = [noffset in keys for noffset in noffsets]
        left = [noffset for noffset, exist in zip(noffsets, exists) if exist]
        if len(left) == 1:
            root.del_feature('noffsets')
            root.add_feature('noffset', left[0])
        elif len(left) != noffsets and len(left) > 0:
            root.noffsets = ','.join(left)
        elif len(left) == 0:
            root.del_feature('noffsets')
    for c in root.children:
        disambibuity_noffsets(c, keys)

class Taxonomy(object):
    def __init__(self, tax, name='root'):
        self.root = None
        self.tax = tax
        self.name = name
        self.build_from_local()

    def _add_current_as_child(self, one, root):
        '''
        one has not been in root
        '''
        if type(one) is dict:
            list_value_keys = [k for k in one if type(one[k]) is list]
            if len(list_value_keys) == 1:
                n = list_value_keys[0]
                sub_root = root.add_child(name=n)
            else:
                assert 'name' in one, one
                sub_root = root.add_child(name=one['name'])
            feats = {}
            for k in one:
                v = one[k]
                if type(v) is not list and k != 'name':
                    feats[k] = v
            sub_root.add_features(**feats)
            if len(list_value_keys) == 1:
                for sub_one in one[list_value_keys[0]]:
                    self._add_current_as_child(sub_one, sub_root)
        else:
            if one is None:
                one = 'None'
            assert type(one) is str or type(one) is unicode
            root.add_child(name=one)

    def build_from_local(self):
        tax = self.tax
        assert type(tax) is list
        self.root = Tree()
        self.root.name = self.name
        for one in tax:
            self._add_current_as_child(one, self.root)

    def dump(self):
        result = []
        for c in self.root.children:
            result.append(dump_tree(c))
        return result

    def _add_children(self, one, root):
        '''
        deprecated
        one: a dictionary, whose information has already been in root
        '''
        if type(one) is str:
            return
        assert type(one) is dict, type(one)
        values = [one[k] for k in one if type(one[k]) is list]
        if len(values) == 1:
            for one_value in values[0]:
                if type(one_value) is str:
                    sub_root = root.add_child(name = one_value)
                else:
                    feats = {}
                    for k in one_value:
                        v = one_value[k]
                        if type(v) is not list:
                            feats[k] = v
                        else:
                            n = k
                    sub_root = root.add_child(name=n)
                    sub_root.add_features(**feats)
                    self._add_children(one_value, sub_root)
        else:
            assert len(values) == 0

def load_all_tax(tax_folder):
    all_yaml = glob.glob(op.join(tax_folder, '*.yaml'))
    all_tax = []
    for y in all_yaml:
        with open(y, 'r') as fp:
            config_tax = yaml.safe_load(fp)
        name = op.splitext(op.basename(y))[0]
        tax = Taxonomy(config_tax, name=name)
        all_tax.append(tax)
    return all_tax

def merge_all_tax(all_tax):
    if len(all_tax) == 0:
        logging.info('no taxonomy found')
        return
    while True:
        all_tax2 = []
        for i in range(len(all_tax)):
            found = False
            for j in range(len(all_tax)):
                if i == j:
                    continue
                tax1 = all_tax[i]
                tax2 = all_tax[j]
                nodes = tax2.root.search_nodes(name=tax1.root.name)
                if len(nodes) == 0:
                    continue
                else:
                    assert len(nodes) == 1, 'more than 1 {} in {}.yaml'.format(
                            tax1.root.name, tax2.root.name)
                    node = nodes[0]
                    node.children.extend(tax1.root.children)
                    found = True
                    break
            if not found:
                all_tax2.append(all_tax[i])
            else:
                all_tax2.extend(all_tax[i + 1:])
                break
        if len(all_tax2) == 1:
            root_tax = all_tax2[0]
            break
        else: 
            assert len(all_tax) == len(all_tax2) + 1, \
                    '{} can not be merged'.format(','.join([t.root.name for t in all_tax]))
            all_tax = all_tax2
    return root_tax


def gen_noffset(tax_input_folder, tax_output_folder):
    all_tax = load_all_tax(tax_input_folder)
    ensure_directory(tax_output_folder)
    for tax in all_tax:
        populate_noffset(tax.root)
        out_file = op.join(tax_output_folder, tax.root.name + '.yaml')
        with open(out_file, 'w') as fp:
            yaml.dump(tax.dump(), fp)

def gen_term_list(tax_folder, term_list):
    all_tax = load_all_tax(tax_folder)
    root_tax = merge_all_tax(all_tax)

    all_term = [node.name for node in root_tax.root.iter_search_nodes() if
            node.name != root_tax.root.name]

    write_to_file('\n'.join(map(lambda x: x.encode('utf-8'), all_term)),
        term_list)
