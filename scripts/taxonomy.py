import logging
from collections import OrderedDict
from nltk.corpus import wordnet as wn
import yaml
from ete2 import Tree
import glob
import os.path as op
from qd_common import load_list_file
from qd_common import write_to_file, read_to_buffer
from qd_common import init_logging, ensure_directory
from qd_common import load_from_yaml_file
import Queue

def get_nick_name(s):
    n = s.name()
    return n[: -5]

def create_markdown_url(noffsets):
    urls = ['[{0}](http://www.image-net.org/synset?wnid={0})'.format(noffset) for
            noffset in noffsets]
    return urls

def update_with_path(root, path, psudo=False):
    curr_root = root
    for i in xrange(len(path)):
        s = path[i]
        existings = curr_root.search_nodes(noffset=synset_to_noffset(s))
        assert len(existings) <= 1
        if len(existings) == 1:
            curr_root = existings[0]
            continue
        else:
            if psudo:
                return len(path) - i
            else:
                curr_root = curr_root.add_child(name=s.name())
                curr_root.add_features(nick_name=get_nick_name(s),
                        noffset=synset_to_noffset(s))
    if psudo:
        return 0

def update_with_synsets(root, ss):
    '''
    build the tree based on the method described in yolo9k paper
    '''
    ambiguities = []
    c = 0
    for i, s in enumerate(ss):
        ps = [p for p in s.hypernym_paths() if any(pn for pn in p if
            synset_to_noffset(pn)==root.noffset)]
        if len(ps) == 0:
            logging.info('ignore {}'.format(s))
            continue
        c = c + 1
        if len(ps) == 1:
            update_with_path(root, ps[0])
        else:
            ambiguities.append(ps)
        if (i % 500) == 0:
            logging.info('non-ambiguities: {}/{}'.format(i, len(ss)))

    for j, ps in enumerate(ambiguities):
        ns = [update_with_path(root, p, psudo=True) for p in ps]
        i = np.argmin(np.asarray(ns))
        update_with_path(root, ps[i])
        if (j % 500) == 0:
            logging.info('ambiguities: {}/{}'.format(j, len(ambiguities)))

def prune_root(root):
    curr = root
    while True:
        if len(curr.children) == 1:
            curr = curr.children[0]
        else:
            break
    root = curr
    return root

def prune(root):
    '''
    remove any node who has only one child
    '''
    root = prune_root(root)
    cs = []
    for c in root.children:
        cs.append(prune(c))
    root.children = cs
    return root

def check_prune(root):
    if len(root.children) == 1: 
        logging.info('{}->{}'.format(root.synset.name(),
            root.children[0].synset.name()))
    for c in root.children:
        check_prune(c)

def tree_size(root):
    if root == None:
        return 0
    else:
        s = 0
        for c in root.children:
            s = s + tree_size(c)
        return 1 + s

def populate_dataset_count(root, images_root):
    for n in root.iter_search_nodes():
        noffset = '{}{}'.format(n.synset.pos(), n.synset.offset())
        image_dir = op.join(images_root, noffset)
        if not op.exists(image_dir):
            num_image = 0
        else:
            num_image = len(glob.glob(op.join(image_dir, '*.*')))
        n.add_feature('num_image', num_image)

class LabelTree(object):
    def __init__(self, tree_file):
        r = load_label_parent(tree_file)
        self.noffset_idx, self.noffset_parentidx, self.noffsets = r
        self.root, self.noffset_node = load_label_tree(self.noffset_parentidx, self.noffsets)
        self.basefilename = op.basename(tree_file)

def synset_to_noffset(synset):
    return '{}{:0>8}'.format(synset.pos(), synset.offset())

def noffset_to_synset(noffset):
    return wn.synset_from_pos_and_offset(noffset[0], int(noffset[1:]))

def child_parent_print_tree2(root, field):
    def get_field(n):
        key = n.__getattribute__(field)
        key = key.replace(' ', '_')
        return key
    name_to_lineidx = {}
    q = Queue.Queue()
    q.put(root)
    idx = -1
    while not q.empty(): 
        n = q.get()
        key = get_field(n)
        name_to_lineidx[key] = idx
        idx = idx + 1
        for c in n.children:
            q.put(c)
    q.put(root)
    lines = []
    labels = []
    while not q.empty():
        n = q.get()
        ps = n.get_ancestors()
        if len(ps) >= 1:
            key = get_field(n) 
            lines.append((key, name_to_lineidx[get_field(ps[0])]))
            labels.append(n.__getattribute__(field))
        for c in n.children:
            q.put(c)
    return labels, lines

def populate_cum_images(root):
    cum_images_with_bb = root.images_with_bb
    cum_images_no_bb = root.images_no_bb
    for c in root.children:
        populate_cum_images(c)
        cum_images_with_bb = cum_images_with_bb + c.cum_images_with_bb
        cum_images_no_bb = cum_images_no_bb + c.cum_images_no_bb

    root.add_feature('cum_images_with_bb', cum_images_with_bb)
    root.add_feature('cum_images_no_bb', cum_images_no_bb)

def load_label_tree(noffset_parentidx, noffsets):
    root = Tree()
    root_synset = wn.synset('physical_entity.n.01')
    root.name = root_synset.name()
    root.add_feature('synset', root_synset)
    noffset_node = {}
    for noffset in noffsets:
        parientid = noffset_parentidx[noffset] 
        if parientid == -1:
            c = root.add_child(name=noffset)
        else:
            parentnode = noffset_node[noffsets[parientid]]
            c = parentnode.add_child(name=noffset)
        noffset_node[noffset] = c
    return prune_root(root), noffset_node

def load_label_parent(label_tree_file):
    buf = read_to_buffer(label_tree_file).split('\n')
    label_map = [b.split(' ') for b in buf]
    # hack: the labelmap should be in the same folder with the tree file
    # sometimes, the label in the tree file is different than the name in
    # labelmap because there should be no whitespace in the label part. Thus,
    # we have to use the labelmap to replace the labels in the tree file
    true_labels = load_list_file(op.join(op.dirname(label_tree_file), 'labelmap.txt'))
    for i, true_label in enumerate(true_labels):
        label_map[i][0] = true_label
    if len(label_map[-1]) == 1:
        label_map = label_map[:-1]
    label_idx, label_parentidx = {}, {}
    labels = []
    idx = 0
    for l_pi in label_map:
        label, pi = l_pi
        pi = int(pi)
        label_parentidx[label] = pi
        label_idx[label] = idx
        idx = idx + 1
        labels.append(label)
    return label_idx, label_parentidx, labels 


def dump_tree(root, feature_name=None):
    result = OrderedDict()
    if feature_name is None:
        for f in root.features:
            if f in ['support', 'name', 'dist', 'synset']:
                continue
            result[f] = root.__getattribute__(f)
    elif type(feature_name) == list:
        for f in feature_name:
            result[f] = root.__getattribute__(f)
    elif type(feature_name) == str:
        if feature_name != 'name':
            result[feature_name] = root.__getattribute__(feature_name)
    result[root.name] = []
    for c in root.children:
        result[root.name].append(dump_tree(c, feature_name))
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

    def dump(self, feature_name=None):
        result = []
        for c in self.root.children:
            result.append(dump_tree(c, feature_name))
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
