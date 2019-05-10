import logging
from pprint import pformat
import glob
import json
import random
from qd.qd_common import ensure_directory
from qd.qd_common import load_list_file
from qd.qd_common import generate_lineidx
from qd.qd_common import copy_file
from qd.qd_common import worth_create
import six
import os
import os.path as op
import shutil
import re
try:
    from itertools import izip as zip
except ImportError:
    # python 3
    pass
import progressbar
from tqdm import tqdm


def rm_tsv(tsv_file):
    if op.isfile(tsv_file):
        os.remove(tsv_file)
        line_idx = op.splitext(tsv_file)[0] + '.lineidx'
        if op.isfile(line_idx):
            os.remove(line_idx)

def tsv_rm(tsv_file):
    rm_tsv(tsv_file)

def tsv_copy(src_tsv, dst_tsv):
    copy_file(src_tsv, dst_tsv)
    src_idx = op.splitext(src_tsv)[0] + '.lineidx'
    if op.isfile(src_idx):
        dst_idx = op.splitext(dst_tsv)[0] + '.lineidx'
        copy_file(src_idx, dst_idx)

def tsv_mv(src_file, dst_file):
    shutil.move(src_file, dst_file)
    src_idx = op.splitext(src_file)[0] + '.lineidx'
    if op.isfile(src_idx):
        dst_idx = op.splitext(dst_file)[0] + '.lineidx'
        shutil.move(src_idx, dst_idx)

def reorder_tsv_keys(in_tsv_file, ordered_keys, out_tsv_file):
    tsv = TSVFile(in_tsv_file)
    keys = [tsv.seek_first_column(i) for i in tqdm(range(len(tsv)))]
    key_to_idx = {key: i for i, key in enumerate(keys)}
    def gen_rows():
        for key in tqdm(ordered_keys):
            idx = key_to_idx[key]
            yield tsv.seek(idx)
    tsv_writer(gen_rows(), out_tsv_file)

def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)

class TSVFile(object):
    def __init__(self, tsv_file, cache_policy=None):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        self.cache_policy= cache_policy
        # the process always keeps the process which opens the
        # file. If the pid is not equal to the currrent pid, we will re-open
        # teh file.
        self.pid = None

        self._cache()

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            if not op.isfile(self.lineidx) and not op.islink(self.lineidx) \
                    or worth_create(self.tsv_file, self.lineidx, buf_second=60):
                generate_lineidx(self.tsv_file, self.lineidx)
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _cache(self):
        if self.cache_policy == 'memory':
            # make sure the tsv is opened here. don't put it in seek. If we put
            # it in the first call of seek(), it is loading all the content
            # there. With multi-workers in pytorch, each worker has to read all
            # the files and cache it to memory. If we load it here in the main
            # thread, it won't copy it to each worker
            logging.info('caching {} to memory'.format(self.tsv_file))
            try:
                import cStringIO as StringIO
            except:
                # python 3
                from io import StringIO
            result = StringIO.StringIO()
            total = op.getsize(self.tsv_file)
            import psutil
            avail = psutil.virtual_memory().available
            if avail < total:
                logging.info('not enough memory to cache {} < {}. fall back'.format(
                    avail, total))
            else:
                pbar = tqdm(total=total/1024./1024.)
                with open(self.tsv_file, 'r') as fp:
                    while True:
                        x = fp.read(1024*1024*100)
                        if len(x) == 0:
                            break
                        pbar.update(len(x) / 1024./1024.)
                        result.write(x)
                self._fp = result

        elif self.cache_policy == 'tmp':
            tmp_tsvfile = op.join('/tmp', self.tsv_file)
            tmp_lineidx = op.join('/tmp', self.lineidx)
            ensure_directory(op.dirname(tmp_tsvfile))

            from qd.qd_common import ensure_copy_file
            ensure_copy_file(self.tsv_file, tmp_tsvfile)
            ensure_copy_file(self.lineidx, tmp_lineidx)

            self.tsv_file = tmp_tsvfile
            self.lineidx = tmp_lineidx
            # do not run the following. Supposedly, this function is called in
            # init function. If we use multiprocess, the file handler will be
            # duplicated and thus the seek will have some race condition if we
            # have the following.
            #self._fp = open(self.tsv_file, 'r')
        elif self.cache_policy is not None:
            raise ValueError('unkwown cache policy {}'.format(self.cache_policy))

    def _ensure_tsv_opened(self):
        if self.cache_policy == 'memory':
            assert self._fp is not None
            return

        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

class TSVDataset(object):
    def __init__(self, name, data_root=None):
        self.name = name
        if data_root is None:
            proj_root = op.dirname(op.dirname(op.dirname(op.realpath(__file__))))
            data_root = op.join(proj_root, 'data', name)
        self._data_root = op.relpath(data_root)
        self._fname_to_tsv = {}

        self._split_to_key_to_idx = {}

    def seek_by_key(self, key, split, t=None, version=None):
        if split in self._split_to_key_to_idx:
            key_to_idx = self._split_to_key_to_idx[split]
        else:
            key_to_idx = {k: i for i, k in enumerate(self.load_keys(split))}
            self._split_to_key_to_idx[split] = key_to_idx
        idx = key_to_idx[key]
        return next(self.iter_data(split, t, version, filter_idx=[idx]))

    def load_labelmap(self):
        return load_list_file(self.get_labelmap_file())

    def load_pos_labelmap(self):
        return load_list_file(self.get_pos_labelmap_file())

    def get_tree_file(self):
        return op.join(self._data_root, 'tree.txt')

    def get_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.txt')

    def get_pos_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.pos.txt')

    def get_train_shuffle_file(self):
        return self.get_shuffle_file('train')

    def get_shuffle_file(self, split_name):
        return op.join(self._data_root, '{}.shuffle.txt'.format(split_name))

    def get_labelmap_of_noffset_file(self):
        return op.join(self._data_root, 'noffsets.label.txt')

    def load_key_to_idx(self, split):
        result = {}
        for i, row in enumerate(self.iter_data(split, 'label')):
            key = row[0]
            assert key not in result
            result[key] = i
        return result

    def load_keys(self, split):
        assert self.has(split, 'label')
        result = []
        for row in tqdm(self.iter_data(split, 'label')):
            result.append(row[0])
        return result

    def dynamic_update(self, dataset_ops):
        '''
        sometimes, we update the dataset, and here, we should update the file
        path
        '''
        if len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'sample':
            self._data_root = op.join('./output/data/',
                    '{}_{}_{}'.format(self.name,
                        dataset_ops[0]['sample_label'],
                        dataset_ops[0]['sample_image']))
        elif len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'mask_background':
            target_folder = op.join('./output/data',
                    '{}_{}_{}'.format(self.name,
                        '.'.join(map(str, dataset_ops[0]['old_label_idx'])),
                        dataset_ops[0]['new_label_idx']))
            self._data_root = target_folder

    def get_test_tsv_file(self, t=None):
        return self.get_data('test', t)

    def get_test_tsv_lineidx_file(self):
        return op.join(self._data_root, 'test.lineidx')

    def get_train_tsvs(self, t=None):
        if op.isfile(self.get_data('train', t)):
            return [self.get_data('train', t)]
        trainx_file = op.join(self._data_root, 'trainX.tsv')
        if not op.isfile(trainx_file):
            return []
        train_x = load_list_file(trainx_file)
        if t is None:
            return train_x
        elif t =='label':
            if op.isfile(self.get_data('trainX', 'label')):
                return load_list_file(self.get_data('trainX', 'label'))
            else:
                files = [op.splitext(f)[0] + '.label.tsv' for f in train_x]
                return files

    def get_train_tsv(self, t=None):
        return self.get_data('train', t)

    def get_lineidx(self, split_name):
        return op.join(self._data_root, '{}.lineidx'.format(split_name))

    def get_latest_version(self, split, t=None):
        assert t is not None, 'if it is none, it is always 0'
        v = 0
        if t is None:
            pattern = op.join(self._data_root, '{}.v*.tsv'.format(split))
            re_pattern = '{}\.v([0-9]*)\.tsv'.format(split)
        else:
            pattern = op.join(self._data_root, '{}.{}.v*.tsv'.format(
                split, t))
            re_pattern = '{}\.{}\.v([0-9]*)\.tsv'.format(split, t)
        all_file = glob.glob(pattern)
        import re
        re_results = [re.match(re_pattern, op.basename(f)) for f in all_file]
        candidates = ([int(re_result.groups()[0]) for re_result, f in
            zip(re_results, all_file) if re_result])
        if len(candidates) > 0:
            v = max(candidates)
        assert v >= 0
        return v

    def get_data(self, split_name, t=None, version=None):
        '''
        e.g. split_name = train, t = label
        if version = None or 0,  return train.label.tsv
        we don't have train.label.v0.tsv
        if version = 3 > 0, return train.label.v3.tsv
        if version = -1, return the highest version
        '''
        if version is None or version == 0:
            if t is None:
                return op.join(self._data_root, '{}.tsv'.format(split_name))
            else:
                return op.join(self._data_root, '{}.{}.tsv'.format(split_name,
                    t))
        elif version > 0:
            if t is None:
                return op.join(self._data_root, '{}.v{}.tsv'.format(split_name,
                    version))
            else:
                return op.join(self._data_root, '{}.{}.v{}.tsv'.format(split_name,
                    t, version))
        elif version == -1:
            if not op.isfile(self.get_data(split_name, t)):
                return self.get_data(split_name, t)
            v = self.get_latest_version(split_name, t)
            return self.get_data(split_name, t, v)


    def get_num_train_image(self):
        if op.isfile(self.get_data('trainX')):
            if op.isfile(self.get_shuffle_file('train')):
                return len(load_list_file(self.get_shuffle_file('train')))
            else:
                return 0
        else:
            return len(load_list_file(op.join(self._data_root, 'train.lineidx')))

    def get_trainval_tsv(self, t=None):
        return self.get_data('trainval', t)

    def get_noffsets_file(self):
        return op.join(self._data_root, 'noffsets.txt')

    def load_noffsets(self):
        logging.info('deprecated: pls generate it on the fly')
        return load_list_file(self.get_noffsets_file())

    def load_inverted_label(self, split, version=None, label=None):
        fname = self.get_data(split, 'inverted.label', version)
        if not op.isfile(fname):
            return {}
        elif label is None:
            tsv = TSVFile(fname)
            num_rows = len(tsv)
            result = {}
            for row in tqdm(tsv, total=num_rows):
                assert row[0] not in result
                assert len(row) == 2
                ss = row[1].split(' ')
                if len(ss) == 1 and ss[0] == '':
                    result[row[0]] = []
                else:
                    result[row[0]] = list(map(int, ss))
            return result
        else:
            all_label = load_list_file(self.get_data(split, 'labelmap', version))
            if label not in all_label:
                return {}
            result = {}
            idx = all_label.index(label)
            tsv = self._retrieve_tsv(fname)
            row = tsv.seek(idx)
            assert row[0] == label
            ss = row[1].split(' ')
            if len(ss) == 1 and ss[0] == '':
                result[row[0]] = []
            else:
                result[row[0]] = list(map(int, ss))
            return result

    def load_inverted_label_as_list(self, split, version=None, label=None):
        fname = self.get_data(split, 'inverted.label', version)
        if not op.isfile(fname):
            return []
        elif label is None:
            rows = tsv_reader(fname)
            result = []
            for row in rows:
                assert len(row) == 2
                ss = row[1].split(' ')
                if len(ss) == 1 and ss[0] == '':
                    result.append((row[0], []))
                else:
                    result.append((row[0], list(map(int, ss))))
            return result
        else:
            all_label = self.load_labelmap()
            result = []
            idx = all_label.index(label)
            tsv = self._retrieve_tsv(fname)
            row = tsv.seek(idx)
            assert row[0] == label
            ss = row[1].split(' ')
            if len(ss) == 1 and ss[0] == '':
                result.append((row[0], []))
            else:
                result.append((row[0], list(map(int, ss))))
            return result

    def has(self, split, t=None, version=None):
        return op.isfile(self.get_data(split, t, version)) or (
                op.isfile(self.get_data('{}X'.format(split), t, version)) and
                op.isfile(self.get_shuffle_file(split)))

    def last_update_time(self, split, t=None, version=None):
        tsv_file = self.get_data(split, t, version)
        if op.isfile(tsv_file):
            return os.path.getmtime(tsv_file)
        assert version is None or version == 0, 'composite dataset always v=0'
        tsv_file = self.get_data('{}X'.format(split), t, version)
        assert op.isfile(tsv_file)
        return os.path.getmtime(tsv_file)

    def iter_composite(self, split, t, version, filter_idx=None):
        splitX = split + 'X'
        file_list = load_list_file(self.get_data(splitX, t, version))
        tsvs = [self._retrieve_tsv(f) for f in file_list]
        shuffle_file = self.get_shuffle_file(split)
        if filter_idx is None:
            shuffle_tsv_rows = tsv_reader(shuffle_file)
            for idx_source, idx_row in shuffle_tsv_rows:
                idx_source, idx_row = int(idx_source), int(idx_row)
                row = tsvs[idx_source].seek(idx_row)
                if len(row) == 3:
                    row[1] == 'dont use'
                yield row
        else:
            shuffle_tsv = self._retrieve_tsv(shuffle_file)
            for i in filter_idx:
                idx_source, idx_row = shuffle_tsv.seek(i)
                idx_source, idx_row = int(idx_source), int(idx_row)
                row = tsvs[idx_source].seek(idx_row)
                if len(row) == 3:
                    row[1] == 'dont use'
                yield row

    def num_rows(self, split, t=None, version=None):
        f = self.get_data(split, t, version)
        if op.isfile(f) or op.islink(f):
            return TSVFile(f).num_rows()
        else:
            f = self.get_data(split + 'X', version=version)
            assert op.isfile(f), f
            return len(load_list_file(self.get_shuffle_file(split)))

    def iter_data(self, split, t=None, version=None,
            unique=False, filter_idx=None, progress=False):
        if progress:
            if filter_idx is None:
                num_rows = self.num_rows(split)
            else:
                num_rows = len(filter_idx)
            pbar = progressbar.ProgressBar(maxval=num_rows).start()
        splitX = split + 'X'
        if not op.isfile(self.get_data(split, t, version)) and \
                op.isfile(self.get_data(splitX, t, version)):
            if t is not None:
                if unique:
                    returned = set()
                for i, row in enumerate(self.iter_composite(split, t, version,
                        filter_idx=filter_idx)):
                    if unique and row[0] in returned:
                        continue
                    else:
                        yield row
                        if unique:
                            returned.add(row[0])
                    if progress:
                        pbar.update(i)
            else:
                rows_data = self.iter_composite(split, None, version=version,
                        filter_idx=filter_idx)
                rows_label = self.iter_data(split, 'label', version=version,
                        filter_idx=filter_idx)
                if unique:
                    returned = set()
                for i, (r_data, r_label) in enumerate(zip(rows_data, rows_label)):
                    r_data[0] = r_label[0]
                    r_data[1] = r_label[1]
                    if unique and r_data[0] in returned:
                        continue
                    else:
                        yield r_data
                        if unique:
                            returned.add(r_data[0])
                    if progress:
                        pbar.update(i)
        else:
            fname = self.get_data(split, t, version)
            if not op.isfile(fname):
                logging.info('no {}'.format(fname))
                return
            if filter_idx is None:
                for i, row in enumerate(tsv_reader(self.get_data(
                    split, t, version))):
                    yield row
                    if progress:
                        pbar.update(i)
            else:
                fname = self.get_data(split, t, version)
                tsv = self._retrieve_tsv(fname)
                if progress:
                    for i in tqdm(filter_idx):
                        yield tsv.seek(i)
                else:
                    for i in filter_idx:
                        yield tsv.seek(i)


    def _retrieve_tsv(self, fname):
        if fname in self._fname_to_tsv:
            tsv = self._fname_to_tsv[fname]
        else:
            tsv = TSVFile(fname)
            self._fname_to_tsv[fname] = tsv
        return tsv

    def write_data(self, rows, split, t=None, version=None):
        tsv_writer(rows, self.get_data(split, t, version))

    def update_data(self, rows, split, t, generate_info=None):
        '''
        if the data are the same, we will not do anything.
        '''
        assert t is not None
        v = self.get_latest_version(split, t)
        if self.has(split, t, v):
            is_equal = True
            # we first save it to a tmp tsv file
            self.write_data(rows, split, t + '.tmp', v + 1)
            for origin_row, new_row in zip(self.iter_data(split, t, v),
                    self.iter_data(split, t + '.tmp', v + 1)):
                if len(origin_row) != len(new_row):
                    is_equal = False
                    break
                for o, n in zip(origin_row, new_row):
                    if o != n:
                        is_equal = False
                        break
                if not is_equal:
                    break
            if not is_equal:
                logging.info('creating {} for {}'.format(v + 1, self.name))
                if generate_info:
                    self.write_data(generate_info, split, '{}.generate.info'.format(t), v + 1)
                tsv_mv(self.get_data(split, t + '.tmp', v + 1),
                        self.get_data(split, t, v + 1))
                return v + 1
            else:
                logging.info('ignore to create since the label matches the latest')
        else:
            assert v == 0
            v = -1
            logging.info('creating {} for {}'.format(v + 1, self.name))
            if generate_info:
                self.write_data(generate_info, split, '{}.generate.info'.format(t), v + 1)
            self.write_data(rows, split, t, version=v + 1)
            return v + 1

    def load_composite_source_data_split_versions(self, split):
        splitX = split + 'X'
        pattern = 'data/(.*)/(train|trainval|test)\.label\.v(.*)\.tsv'
        tsv_sources = [l for l, in tsv_reader(self.get_data(splitX,
            'origin.label'))]
        matched_result = [re.match(pattern, l).groups()
                for l in tsv_sources]

        return [(d, s, int(v)) for d, s, v in matched_result]

def tsv_writer(values, tsv_file_name, sep='\t'):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    import sys
    is_py2 = sys.version_info.major == 2
    with open(tsv_file_name_tmp, 'wb') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value
            if is_py2:
                v = sep.join(map(lambda v: str(v) if not isinstance(v, six.string_types) else v, value)) + '\n'
                if type(v) is unicode:
                    v = v.encode('utf-8')
            else:
                v = sep.join(map(lambda v: v.decode() if type(v) == bytes else str(v), value)) + '\n'
                v = v.encode()
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)

    if os.path.isfile(tsv_file_name):
        os.remove(tsv_file_name)
    if os.path.isfile(tsv_lineidx_file):
        os.remove(tsv_lineidx_file)
    os.rename(tsv_file_name_tmp, tsv_file_name)
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)

def tsv_reader(tsv_file_name, sep='\t'):
    with open(tsv_file_name, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def csv_reader(tsv_file_name):
    tsv_reader(tsv_file_name, ',')

def get_meta_file(tsv_file):
    return op.splitext(tsv_file)[0] + '.meta.yaml'

def extract_label(full_tsv, label_tsv):
    if op.isfile(label_tsv):
        logging.info('label file exists and will skip to generate: {}'.format(
            label_tsv))
        return
    if not op.isfile(full_tsv):
        logging.info('the file of {} does not exist'.format(full_tsv))
        return
    rows = tsv_reader(full_tsv)
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info('extract_label: {}-{}'.format(full_tsv, i))
            del row[2]
            assert len(row) == 2
            assert type(row[0]) == str
            assert type(row[1]) == str
            yield row
    tsv_writer(gen_rows(), label_tsv)

def create_inverted_tsv(rows, inverted_label_file, label_map):
    '''
    deprecated, use create_inverted_list
    save the results based on the label_map in label_map_file. The benefit is
    to seek the row given a label
    '''
    inverted = {}
    for i, row in enumerate(rows):
        labels = json.loads(row[1])
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
        else:
            assert type(labels) is int
            curr_unique_labels = [label_map[labels]]
        for l in curr_unique_labels:
            assert type(l) == str or type(l) == unicode
            if l not in inverted:
                inverted[l] = [i]
            else:
                inverted[l].append(i)
    def gen_rows():
        for label in inverted:
            assert label in label_map
        for label in label_map:
            i = inverted[label] if label in inverted else []
            yield label, ' '.join(map(str, i))
    tsv_writer(gen_rows(), inverted_label_file)

def create_inverted_list2(rows, th=None):
    inverted = {}
    keys = []
    for i, row in enumerate(rows):
        keys.append(row[0])
        labels = json.loads(row[1])
        if th is not None:
            labels = [r for r in labels if 'conf' in r and r['conf'] > th or
                            'conf' not in r]
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
        else:
            assert type(labels) is int
            curr_unique_labels = [str(labels)]
        for l in curr_unique_labels:
            assert type(l) == str or type(l) == unicode
            if l not in inverted:
                inverted[l] = [i]
            else:
                inverted[l].append(i)
    return inverted, keys

def is_verified_rect(rect):
    #allowed_keys = set(['class', 'rect', 'uhrs_confirm', 'uhrs_uncertain',
            #'conf', 'merge_from', 'class_from', 'change_from', 'from', 'diff',
            #'IsInside', 'IsGroupOf', 'IsDepiction', 'IsOccluded',
            #'IsTruncated', 'workerId', 'class_propagate_from', 'obj', 'uhrs'])
    #unknown_keys = [k for k in rect if k not in allowed_keys]
    #if len(unknown_keys) > 0:
        #logging.info('unknown keys = {}\n'.format(pformat(unknown_keys)))
        #pass

    if 'uhrs' in rect:
        judge_result = rect['uhrs']
        assert judge_result.get('1', 0) >= judge_result.get('2', 0)
        return True

    if 'class' not in rect or 'rect' not in rect:
        return False

    if 'uhrs_confirm' in rect:
        assert rect['uhrs_confirm'] > 0
        return True

    if 'conf' in rect:
        return False

    if 'merge_from' in rect:
        return all(is_verified_rect(r) for r in rect['merge_from'])

    return True

def create_inverted_list(rows):
    inverted = {}
    inverted_with_bb = {}
    inverted_no_bb = {}
    inverted_with_bb_verified = {}
    inverted_with_bb_noverified = {}
    logging.info('creating inverted')
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
            curr_unique_with_bb_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect'])])
            curr_unique_no_bb_labels = set([l['class'] for l in labels
                if 'rect' not in l or all(x == 0 for x in l['rect'])])
            curr_unique_with_bb_verified_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect']) and is_verified_rect(l)])
            curr_unique_with_bb_noverified_labels = set([l['class'] for l in labels
                if 'rect' in l and any(x != 0 for x in l['rect']) and not is_verified_rect(l)])
        else:
            assert type(labels) is int
            curr_unique_labels = [str(labels)]
            curr_unique_with_bb_labels = []
            curr_unique_no_bb_labels = curr_unique_labels
        def update(unique_labels, inv):
            for l in unique_labels:
                assert type(l) == str or type(l) == unicode
                if l not in inv:
                    inv[l] = [i]
                else:
                    inv[l].append(i)
        update(curr_unique_labels, inverted)
        update(curr_unique_with_bb_labels, inverted_with_bb)
        update(curr_unique_no_bb_labels, inverted_no_bb)
        update(curr_unique_with_bb_verified_labels, inverted_with_bb_verified)
        update(curr_unique_with_bb_noverified_labels, inverted_with_bb_noverified)
    return {'inverted.label': inverted,
            'inverted.label.with_bb': inverted_with_bb,
            'inverted.label.no_bb': inverted_no_bb,
            'inverted.label.with_bb.verified': inverted_with_bb_verified,
            'inverted.label.with_bb.noverified': inverted_with_bb_noverified}

def tsv_shuffle_reader(tsv_file):
    logging.warn('deprecated: using TSVFile to randomly seek')
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    lineidx = load_list_file(lineidx_file)
    random.shuffle(lineidx)
    with open(tsv_file, 'r') as fp:
        for l in lineidx:
            fp.seek(int(float(l)))
            yield [x.strip() for x in fp.readline().split('\t')]

def load_labelmap(data):
    dataset = TSVDataset(data)
    return dataset.load_labelmap()

def get_all_data_info2(name=None):
    if name is None:
        return sorted(os.listdir('./data'))
    else:
        dataset = TSVDataset(name)
        if not op.isfile(dataset.get_labelmap_file()):
            return []
        global_labelmap = None
        labels = dataset.load_labelmap()
        valid_split_versions = []
        splits = ['train', 'trainval', 'test']
        for split in splits:
            v = 0
            while True:
                if not dataset.has(split, 'label', v):
                    break
                labelmap = []
                label_count_rows = dataset.iter_data(split, 'inverted.label.count', v)
                label_count = [(r[0], int(r[1])) for r in label_count_rows]
                label_count = sorted(label_count, key=lambda x: x[1])
                valid_split_versions.append((split, v, [(i, l, c) for i, (l, c) in
                    enumerate(label_count)]))
                v = v + 1
        name_splits_labels = [(name, valid_split_versions)]
        return name_splits_labels

def get_all_data_info():
    names = os.listdir('./data')
    name_splits_labels = []
    names.sort(key=lambda n: n.lower())
    for name in names:
        dataset = TSVDataset(name)
        if not op.isfile(dataset.get_labelmap_file()):
            continue
        labels = dataset.load_labelmap()
        valid_splits = []
        if len(dataset.get_train_tsvs()) > 0:
            valid_splits.append('train')
        for split in ['trainval', 'test']:
            if not op.isfile(dataset.get_data(split)):
                continue
            valid_splits.append(split)
        name_splits_labels.append((name, valid_splits, labels))
    return name_splits_labels

def load_labels(file_name):
    rows = tsv_reader(file_name)
    key_to_rects = {}
    key_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        rects = json.loads(row[1])
        #assert key not in labels, '{}-{}'.format(file_name, key)
        key_to_rects[key] = rects
        key_to_idx[key] = i
    return key_to_rects, key_to_idx


