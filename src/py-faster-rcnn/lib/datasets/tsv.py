# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import sys, os
#sys.path.insert(0,'D:\Src\IRISObjectDetection\src\py-faster-rcnn\lib');
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from tsv_eval import tsv_eval
from fast_rcnn.config import cfg
import base64
import json
import random

class tsv(imdb):
    def __init__(self, folder_name, image_set, devkit_path=None):
        """
        folder_name: the root folder of the tsv dataset
        image_set: train or val set
        """
        tsv_file_name = '%s.tsv'%image_set;
        imdb.__init__(self, 'tsv_' + folder_name + '_' + image_set)
        self._folder_name = folder_name
        self._image_set = image_set

        self._data_folder = os.path.join(cfg.DATA_DIR, folder_name)
        self._tsv_file = os.path.join(self._data_folder, tsv_file_name)
        self._tsv_f = open(self._tsv_file, 'r')

        assert os.path.exists(self._data_folder), \
                'Path does not exist: {}'.format(self._data_folder)
        assert os.path.exists(self._tsv_file), \
                'Path does not exist: {}'.format(self._tsv_file)

        result_folder = os.path.join(self._data_folder, 'results')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        # load lineidx file for random access
        lineidx_file = os.path.splitext(self._tsv_file)[0] + '.lineidx'
        with open(lineidx_file, 'r') as f:
            self._lineidx = [int(line.split('\t')[0]) for line in f]
        self._imgname = ["" for x in self._lineidx]
        # load label map
        labelmap_file = os.path.join(self._data_folder, 'labelmap.txt')
        with open(labelmap_file, 'r') as f:
            self._classes = [line.split('\t')[0].strip() for line in f]
        self._classes.insert(0, '__background__')    # always index 0

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = list(range(len(self._lineidx)))
        random.shuffle(self._image_index);
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}


    def image_path_at(self, i):
        """
        Return image i in the image sequence.
        i is the index in the train or test set
        """
        # find the line no first
        line_no = self.image_index[i]
        # seek to the beginning of that line
        self._tsv_f.seek(self._lineidx[line_no], 0)
        # read the line content
        line = self._tsv_f.readline().rstrip()
        cols = line.split('\t')
        imagestring = cols[2]
        self._imgname[line_no]=cols[0];
        # decode image string
        jpgbytestring = base64.b64decode(imagestring)

        return jpgbytestring
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        print 'load gt roidb from tsv file'
        label_file = self._tsv_file;
        gt_roidb = []
        self._imgname = [];
        with open(label_file, 'r') as f:
            for line in f:
                cols = [x.strip() for x in line.split("\t")];
                rects = json.loads(cols[1]);
                num_objs = len(rects);
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                gt_classes = np.zeros((num_objs), dtype=np.int32)
                overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
                seg_areas = np.zeros((num_objs), dtype=np.float32)
                label = ""
                for i in range(num_objs):
                    box = rects[i]['rect'];
                    boxes[i,:] = box;
                    label = rects[i]['class'].encode('ascii');
                    cls  = self._class_to_ind[label]
                    gt_classes[i] = cls;
                    overlaps[i, cls] = 1.0
                    seg_areas[i] = (box[2]-box[0] + 1) * (box[3]-box[1]+ 1)
                overlaps = scipy.sparse.csr_matrix(overlaps)
                self._imgname += [cols[0]];
                gt_roidb +=[{'imagename': cols[0],
                'classname': label,
                'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._data_folder,
            'results',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(self._imgname[index], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = tsv_eval(self.gt_roidb(), self.image_index,
                filename, cls, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.tsv2 import tsv
    d = tsv('voc2', 'train')
    d.set_proposal_method('gt')
    res = d.roidb
