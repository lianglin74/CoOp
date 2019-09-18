import logging
from qd.process_tsv import build_taxonomy_impl
from qd.process_tsv import get_taxonomy_path
from qd.process_tsv import build_tax_dataset_from_db
from qd.process_tsv import ensure_inject_dataset
from qd.process_tsv import dump_to_taxonomy_dataset
from qd.tsv_io import TSVDataset
import os.path as op
from qd.process_tsv import build_taxonomy_from_single_source
from qd.process_tsv import get_data_sources
from qd.qd_common import load_list_file
from qd.qd_common import list_to_dict
from qd.qd_common import ensure_copy_folder
from qd.tsv_io import tsv_reader
import re
import random


def ensure_build_taxonomy_vehiclev1_1():
    # training
    datas = [
            {
                'name':       'TerniumVehicle',
                'split_infos': [{'split': 'train', 'version': 0},],
                'use_all': True
            }
        ]
    build_taxonomy_impl(
            get_taxonomy_path('TaxVehicleV1_1'),
            num_test=0,
            data='TaxVehicleV1_1',
            datas=datas,
            max_image_per_label=10000000000)

    # test on trainval
    datas = [
            {
                'name':       'TerniumVehicle',
                'split_infos': [{'split': 'trainval', 'version': 0},],
                'use_all': True
            }
        ]
    build_taxonomy_impl(
            get_taxonomy_path('TaxVehicleV1_1'),
            num_test=10000000000000000,
            data='TaxVehicleValV1_1',
            datas=datas,
            min_image_per_label=10000,
            max_image_per_label=0)

    datas = [
            {
                'name':       'TerniumVehicle',
                'split_infos': [{'split': 'test', 'version': 0},],
                'use_all': True
            }
        ]
    build_taxonomy_impl(
            get_taxonomy_path('TaxVehicleV1_1'),
            num_test=10000000000000000,
            data='TaxVehicleTestV1_1',
            datas=datas,
            min_image_per_label=10000,
            max_image_per_label=0)

def ensure_build_taxonomy_vehicle(data):
    if data in ['TaxVehicleV1_1', 'TaxVehicleV1_1_with_bb']:
        ensure_build_taxonomy_vehiclev1_1()
    else:
        raise NotImplementedError

def ensure_build_taxonomy_trafficsign(data):
    if data == 'TaxTrafficSignV1_1':
        datas = [{'name':       'TsinghuaTraffic_data',
                  'split_infos': [{'split': 'train', 'version': -1},
                                  {'split': 'trainval', 'version': -1}]}
                  ]
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    else:
        raise Exception('unknown data = {}'.format(data))

def add_all_rest_to_tax_dataset(from_data, data_sources, out_data):
    ensure_copy_folder(op.join('data', from_data),
            op.join('data', out_data))
    ensure_copy_folder(op.join('data', from_data + '_no_bb'),
            op.join('data', out_data + '_no_bb'))

    from_data_with_bb = from_data + '_with_bb'
    ref_dataset = TSVDataset(from_data_with_bb)

    all_idxsource_idxrow = [(int(i), int(j)) for i, j in
            tsv_reader(ref_dataset.get_shuffle_file('train'))]
    source_data_split_versions = ref_dataset.load_composite_source_data_split_versions('train')
    source_data_split_to_idx = {(d, s): i for i, (d, s, v) in
            enumerate(source_data_split_versions)}
    idxsource_to_all_idxrow = list_to_dict(all_idxsource_idxrow, 0)
    full_data_splits = []
    for d in data_sources:
        for si in d['split_infos']:
            full_data_splits.append((d['name'], si['split']))
    all_to_add_idxsource_idxrow = []
    for d, s in full_data_splits:
        idxsource = source_data_split_to_idx[(d, s)]
        all_existing_idxrow = idxsource_to_all_idxrow[idxsource]
        all_idxrow = list(range(TSVDataset(d).num_rows(s)))
        to_add_idx = set(all_idxrow).difference(set(all_existing_idxrow))
        all_to_add_idxsource_idxrow.extend([(idxsource, i)
            for i in to_add_idx])
    all_idxsource_idxrow.extend(all_to_add_idxsource_idxrow)
    random.shuffle(all_idxsource_idxrow)

    out_dataset = TSVDataset(out_data + '_with_bb')

    dump_to_taxonomy_dataset(ref_dataset,
            all_idxsource_idxrow,
            source_data_split_versions,
            lift_train=False,
            split='train',
            out_dataset=out_dataset)

def ensure_build_taxonomy_persononly(data):
    if data == 'TaxPersonOnlyV1_1' or data == 'TaxPersonOnlyV1_1_with_bb':
        data = 'TaxPersonOnlyV1_1'
        data_sources = [{'name': 'CrowdHuman',
                         'split_infos': [{'split': 'train',
                                          'version': 1}]}]
        for d in data_sources:
            ensure_inject_dataset(d['name'])
        build_tax_dataset_from_db(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=10000000)
    elif data in ['TaxPersonOnlyV1_2', 'TaxPersonOnlyV1_2_with_bb']:
        data = 'TaxPersonOnlyV1_2'
        data_sources = [{'name': 'CrowdHuman',
                         'split_infos': [{'split': 'train',
                                          'version': 1}]},
                         {'name': 'voc0712',
                          'split_infos': [{'split': 'train',
                                           'version': 0}]},
                         {'name': 'coco2017Full',
                          'split_infos': [{'split': 'train',
                                           'version': 0}]},
                          ]
        for d in data_sources:
            ensure_inject_dataset(d['name'])
        build_tax_dataset_from_db(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=1000000000)
    elif data in ['TaxPersonOnlyV1_3', 'TaxPersonOnlyV1_3_with_bb']:
        data = 'TaxPersonOnlyV1_3'
        data_sources = [{'name': 'CrowdHuman',
                         'split_infos': [{'split': 'train',
                                          'version': 1}],
                         'use_all': True},
                         {'name': 'voc0712',
                          'split_infos': [{'split': 'train',
                                           'version': 0}],
                          'use_all': True},
                         {'name': 'coco2017Full',
                          'split_infos': [{'split': 'train',
                                           'version': 0}],
                          'use_all': True},
                          ]
        for d in data_sources:
            ensure_inject_dataset(d['name'])
        build_tax_dataset_from_db(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=1000000000)

def ensure_build_taxonomy_oi5c(data):
    # the maximum is 807K for person
    if data in ['TaxOI5CV1_1_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 1
        min_image_per_label = 1000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    if data in ['TaxOI5CV1_2_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 0
        min_image_per_label = 1000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_3_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 0
        min_image_per_label = 10000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_4_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 0
        min_image_per_label = 100000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_5_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 0
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_6_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 0
        min_image_per_label = 2000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_7_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 0
        min_image_per_label = 50000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_1_2k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 1
        min_image_per_label = 2000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_1_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 1
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_1_10k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 1
        min_image_per_label = 10000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_2_3_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 3
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_2_4_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 4
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_2_5_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 5
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_2_7_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 7
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_2_8_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 8
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_2_9_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 9
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_2_10_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 10
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
        # V1_6 IsGroupOf study
    elif data in ['TaxOI5CV1_3_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 6
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_4_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 10
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_18_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 18
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_19_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 19
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_20_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 20
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_21_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 21
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_22_5k_with_bb']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 22
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_Tightness_5k']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 23
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_Clean24_5k']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 24
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV1_Tight25_5k']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 25
        min_image_per_label = 5000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV2_0_Man.Woman']:
        assert op.isfile(TSVDataset(data).get_labelmap_file())
    elif data in ['TaxOI5CV31_26_20k']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 26
        min_image_per_label = 20000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
    elif data in ['TaxOI5CV31_26_50k']:
        source_data = 'OpenImageV5C'
        source_split = 'train'
        source_version = 26
        min_image_per_label = 50000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)
            from qd.qd_common import copy_file
            copy_file('./data/OpenImageV5C31/labelmap.txt',
                    out_dataset.get_labelmap_file())
    else:
        raise ValueError('unknown {}'.format(data))

def ensure_build_taxonomy_oi4c(data):
    # the maximum is 807K for person
    if data in ['TaxOI4CV1_1_with_bb']:
        source_data = 'OpenImageV4C'
        source_split = 'train'
        source_version = 1
        min_image_per_label = 10000
        out_data = data
        out_dataset = TSVDataset(out_data)
        if op.isdir(out_dataset._data_root):
            logging.info('ignore to build since {} exists'.format(
                out_dataset._data_root))
        else:
            build_taxonomy_from_single_source(source_data,
                    source_split, source_version, min_image_per_label, out_data)

def ensure_build_taxonomy_cba3(data):
    if data in ['TaxCBA3V1_1', 'TaxCBA3V1_1_with_bb']:
        datas = get_data_sources('TaxCBA3V1')
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data='TaxCBA3V1_1',
                datas=datas,
                max_image_per_label=10000000000)

        data = 'TaxCBA3V1_TEST'
        for d in datas:
            for x in d['split_infos']:
                x['split'] = 'test'
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=1000000,
                data=data,
                datas=datas,
                max_image_per_label=0)
    else:
        raise Exception()

def add_extra_images(source_data, target_data_infos, top_n):
    dataset = TSVDataset(source_data)
    split = 'train'
    splitX = split + 'X'
    shuffle_file = dataset.get_shuffle_file(split)
    idx_source_idx_rows = [(int(idx_source), int(idx_row))
        for idx_source, idx_row in tsv_reader(shuffle_file)]
    file_list = load_list_file(dataset.get_data(splitX))
    datas = [op.basename(op.dirname(f)) for f in file_list]
    splits = [re.match('(train|trainval|test)\..*', op.basename(f)).groups()[0]
        for f in file_list]
    data_split_idxes = [(datas[idx_source], splits[idx_source], idx_row) for idx_source, idx_row
            in idx_source_idx_rows]

    data_to_split_to_split_info = {data_info['name']:
            {split_info['split']: split_info for split_info in data_info['split_infos']}
            for data_info in target_data_infos}
    assert all(d in data_to_split_to_split_info for d in datas)
    random.seed(888)
    random.shuffle(data_split_idxes)
    data_split_idxes = data_split_idxes[:top_n]
    data_to_split_idxes = list_to_dict(data_split_idxes, 0)
    for data in data_to_split_idxes:
        split_idxes = data_to_split_idxes[data]
        split_to_idxes = list_to_dict(split_idxes, 0)
        for split in split_to_idxes:
            idxes = list(set(split_to_idxes[split]))
            data_to_split_to_split_info[data][split]['must_have_indices'] = idxes

def ensure_build_taxonomy_inside(data):
    if data == 'TaxInsideV4_10':
        datas = get_data_sources(version='exclude_golden_use_voc_coco_all')
        datas.extend(get_data_sources('extra_furniture'))
        background_data = 'Tax1300V14.4_0.0_0.0_with_bb'
        add_extra_images(background_data, datas, 600000)
        for d in datas:
            d['select_by_verified'] = True
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_11':
        datas = get_data_sources(version='exclude_golden_use_voc_coco_all')
        datas.extend(get_data_sources('extra_furniture'))
        background_data = 'Tax1300V14.4_0.0_0.0_with_bb'
        add_extra_images(background_data, datas, 700000)
        for d in datas:
            d['select_by_verified'] = True
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_12':
        datas = get_data_sources(version='exclude_golden_use_voc_coco_all')
        datas.extend(get_data_sources('extra_furniture'))
        background_data = 'Tax1300V14.4_0.0_0.0_with_bb'
        add_extra_images(background_data, datas, 800000)
        for d in datas:
            d['select_by_verified'] = True
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_13':
        datas = get_data_sources(version='exclude_golden_use_voc_coco_all')
        datas.extend(get_data_sources('extra_furniture'))
        background_data = 'Tax1300V14.4_0.0_0.0_with_bb'
        add_extra_images(background_data, datas, 900000)
        for d in datas:
            d['select_by_verified'] = True
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_14':
        datas = get_data_sources(version='exclude_golden_use_voc_coco_all')
        datas.extend(get_data_sources('extra_furniture'))
        background_data = 'Tax1300V14.4_0.0_0.0_with_bb'
        add_extra_images(background_data, datas, 1000000)
        for d in datas:
            d['select_by_verified'] = True
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_15':
        datas = get_data_sources(version='v15.2')
        for d in datas:
            d['select_by_verified'] = True
            for split_info in d['split_infos']:
                split_info['version'] = -1
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_16':
        datas = get_data_sources(version='v15.2')
        for d in datas:
            d['select_by_verified'] = True
            for split_info in d['split_infos']:
                split_info['version'] = -1
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_17' or data == 'TaxInsideV4_17_with_bb':
        datas = get_data_sources(version='v15.2')
        for d in datas:
            d['select_by_verified'] = False
            for split_info in d['split_infos']:
                split_info['version'] = -1
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000000000)
    elif data == 'TaxInsideV4_18' or data == 'TaxInsideV4_18_with_bb':
        if not op.isdir(TSVDataset(data)._data_root):
            add_all_rest_to_tax_dataset('TaxInsideV4_16',
                    [{'name': 'voc0712',
                      'split_infos': [{'split': 'train'}]},
                     {'name': 'coco2017',
                      'split_infos': [{'split': 'train'}]}],
                    data)
    elif data == 'TaxInsideV4_19' or data == 'TaxInsideV4_19_with_bb':
        datas = get_data_sources(version='v_with_all_voc_coco')
        for d in datas:
            d['select_by_verified'] = True
            for split_info in d['split_infos']:
                split_info['version'] = -1
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                max_image_per_label=10000)
    else:
        raise Exception('unknown data = {}'.format(data))

def ensure_build_taxonomy(data):
    if not data.startswith('Tax'):
        logging.info('skip because {} does not start with Tax'.format(data))
        return
    if all(op.isdir('data/{}'.format(d)) for d in [data,
            data + '_with_bb', data + '_no_bb']):
        logging.info('skip because exists: {}'.format(data))
        return
    if data.startswith('TaxInside'):
        ensure_build_taxonomy_inside(data)
    elif data.startswith('TaxCBA3'):
        ensure_build_taxonomy_cba3(data)
    elif data.startswith('TaxTrafficSign'):
        ensure_build_taxonomy_trafficsign(data)
    elif data == 'Tax1300V15_1':
        data_sources = get_data_sources('v14.4')
        data_sources.extend(get_data_sources('extra_furniture'))
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=10000)
    elif data == 'Tax1300V15_2':
        data_sources = get_data_sources('v15.2')
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=10000)
    elif data == 'Tax1300V16_1':
        data_sources = get_data_sources('v15.2')
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=10000)
    elif data == 'Tax1300V16_2':
        data_sources = get_data_sources('v15.2')
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=100000)
    elif data == 'TaxProductV1_1':
        data_sources = get_data_sources('product_train')
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=data_sources,
                max_image_per_label=1000000)
    elif data.startswith('TaxPersonOnly'):
        ensure_build_taxonomy_persononly(data)
    elif data.startswith('TaxOI5C'):
        ensure_build_taxonomy_oi5c(data)
    elif data.startswith('TaxOI4C'):
        ensure_build_taxonomy_oi4c(data)
    elif data.startswith('TaxVehicle'):
        ensure_build_taxonomy_vehicle(data)
    else:
        raise ValueError('unknown {}'.format(data))

