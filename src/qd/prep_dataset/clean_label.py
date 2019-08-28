from qd.qd_common import dict_add
from qd.process_tsv import hash_sha1
from qd.qd_common import calculate_iou
from qd.process_tsv import find_same_rects
from qd.tsv_io import TSVDataset
from qd.qd_common import json_dump
from tqdm import tqdm
import simplejson as json


class RemoveSmall(object):
    def __init__(self, ratio=1e-3):
        self.ratio = ratio

    def __call__(self, param, info):
        rects, h, w = param['rects'], param['h'], param['w']

        th_h, th_w = h * self.ratio, w * self.ratio

        param['rects'] = [r for r in rects if r['rect'][2] - r['rect'][0] >
                th_w and r['rect'][3] - r['rect'][1] > th_h]

        info['remove_small_removed'] = info.get('remove_small_removed',
                0) + len(rects) - len(rects)
        info['remove_small_checked'] = info.get('remove_small_checked',
                0) + len(rects)

class TruncateInRange(object):
    def __call__(self, param, info):
        rects, h, w = param['rects'], param['h'], param['w']
        for r in rects:
            x0, y0, x1, y1 = r['rect']
            _x0 = min(max(x0, 0), w)
            _y0 = min(max(y0, 0), h)
            _x1 = min(max(x1, 0), w)
            _y1 = min(max(y1, 0), h)
            if _x0 != x0 or _y0 != y0 or _x1 != x1 or _y1 != y1:
                r['rect'] = [_x0, _y0, _x1, _y1]
                r['location_id'] = hash_sha1(r['rect'])
                info['truncate_num'] = info.get('truncate_num', 0) + 1
                truncates = info.get('truncates', [])
                truncates.append({'key': param['key'], 'rects': json_dump([r])})
            info['truncate_checked'] = info.get('truncate_checked', 0) + 1
        return param

class MergeColocate(object):
    def __init__(self, iou_th=0.85):
        self.iou_th = iou_th

    def __call__(self, param, info):
        origin_rects = param['rects']
        all_rects = []
        for r in origin_rects:
            all_iou = [calculate_iou(r['rect'], rects[0]['rect']) for rects in
                    all_rects]
            all_same_rects = [rects for iou, rects in zip(all_iou, all_rects)
                if iou > self.iou_th]
            if len(all_same_rects) == 0:
                all_rects.append([r])
            else:
                all_same_rects[0].append(r)
                for i in range(1, len(all_same_rects)):
                    all_same_rects[0].extend(all_same_rects[i])
                for i in range(1, len(all_same_rects)):
                    all_rects.remove(all_same_rects[i])
        for rects in all_rects:
            x0 = sum([r['rect'][0] for r in rects]) / len(rects)
            y0 = sum([r['rect'][1] for r in rects]) / len(rects)
            x1 = sum([r['rect'][2] for r in rects]) / len(rects)
            y1 = sum([r['rect'][3] for r in rects]) / len(rects)
            mean_location = [x0, y0, x1, y1]
            location_id = hash_sha1(mean_location)
            for r in rects:
                r['rect'] = mean_location
                r['location_id'] = location_id
            dict_add(info, 'merge_colocate_change_location', len(rects) - 1)
            existing = set()
            to_remove = []
            for r in rects:
                if r['class'] not in existing:
                    existing.add(r['class'])
                else:
                    to_remove.append(r)
            dict_add(info, 'merge_colocate_remove_same_loc_label', len(to_remove))
            for r in to_remove:
                rects.remove(r)

        param['rects'] = [r for rects in all_rects for r in rects]
        dict_add(info, 'merge_colocate_checked', len(origin_rects))
        dict_add(info, 'merge_colocate_merged', len(origin_rects) -
            len(param['rects']))

class RemoveDuplicate(object):
    def __init__(self, iou_th=0.9):
        self.iou_th = iou_th

    def __call__(self, param, info):
        rects = param['rects']
        rects2 = []
        for r in rects:
            same_rects = find_same_rects(r, rects2, iou=self.iou_th)
            if len(same_rects) == 0:
                rects2.append(r)
        param['rects'] = rects2

        dict_add(info, 'remove_duplicate_checked', len(rects))
        dict_add(info, 'remove_duplicate_removed', len(rects) - len(rects2))

class CleanerCompose(object):
    def __init__(self, components):
        self.components = components

    def __call__(self, param, info):
        for c in self.components:
            c(param, info)

def clean_label(data, split, version):
    cleaner = CleanerCompose([
        RemoveSmall(),
        TruncateInRange(),
        MergeColocate(),
        #RemoveDuplicate(),
        ])
    process_label(data, split, version, cleaner)

class ReplaceIsGroupOfByTightness(object):
    def __call__(self, param, info):
        rects = param['rects']
        for r in rects:
            is_group_of = r['IsGroupOf']
            assert is_group_of in [0, 1, -1]
            if is_group_of == 0:
                r['tightness'] = 1
            else:
                r['tightness'] = 0
        dict_add(info, 'replace_is_group_of_total', len(rects))
        dict_add(info, 'replace_is_group_of_num_yes', sum(r['tightness'] for r in rects))

def process_label(data, split, version, cleaner):
    assert isinstance(cleaner, CleanerCompose), 'we use its components to save debug info'
    dataset = TSVDataset(data)
    version = version
    if version == -1:
        version = dataset.get_latest_version(split, t='label')
    label_iter = dataset.iter_data(split, t='label', version=version)
    hw_iter = dataset.iter_data(split, t='hw')
    num_rows = dataset.num_rows(split)
    info = {}
    def gen_rows():
        for i, (label_row, hw_row) in tqdm(enumerate(zip(label_iter, hw_iter)),
                total=num_rows):
            label_key, str_label = label_row
            hw_key, str_hw = hw_row
            assert label_key == hw_key
            rects = json.loads(str_label)
            h, w = list(map(int, str_hw.split(' ')))
            param = {'key': label_key, 'rects': rects, 'h': h, 'w':
                w}
            cleaner(param, info)
            yield label_key, json_dump(param['rects'])

    def generate_info():
        yield 'based on version', version
        for c in cleaner.components:
            yield 'clean type', type(c)
            for k, v in c.__dict__.items():
                # add one tab only for visualization
                yield '', k, v
        for k, v in info.items():
            if type(v) is list:
                yield k,
                for sub_k, sub_v in v:
                    yield sub_k, sub_v
            else:
                yield k, v
    dataset.update_data(gen_rows(), split=split, t='label',
            generate_info=generate_info())

