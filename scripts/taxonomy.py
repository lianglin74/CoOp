import yaml
from ete2 import Tree
import glob
import os.path as op
from qd_common import write_to_file, read_to_buffer

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
