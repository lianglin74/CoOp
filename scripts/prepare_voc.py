import os
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import shutil
import tarfile
import json
import base64

class pascal_voc():
    def __init__(self, data_path, image_set):
        self._data_path = data_path
        self._image_set = image_set
        self._image_index = self._load_image_set_index()

        self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : True,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_pascal_annotation(self, index, class_filter):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        gt = []

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            name = obj.find('name').text.lower().strip()
            if len(class_filter) > 0 and not name in class_filter:
                continue
            diff = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            bbox = [x1, y1, x2, y2]
            gt.append({'diff': diff, 'class': name, 'rect': bbox})

        return gt

    def dump_data_tsv(self, dst_file, classes):
        dst_folder = os.path.split(dst_file)[0]
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        print 'dump data for %s to %s' % (classes, dst_file)
        class_set = set(classes)

        count = 0
        with open(dst_file, 'w') as fout:
            for i, index in enumerate(self.image_index):
                gt = self._load_pascal_annotation(index, class_set)
                if len(gt) > 0:
                    count += 1
                    print '%d/%d => %d\r' % (i+1, len(self.image_index), count),
                    with open(self.image_path_from_index(index), 'rb') as f:
                        image_data = f.read()
                    fout.write('%s\t%s\t%s\n' % (index, json.dumps(gt), base64.b64encode(image_data)))
        
        print '\ndone.'

def save_labelmap(filename, classes):
    with open(filename, 'w') as fout:
        for name in classes:
            fout.write('%s\n'%name)

def download_voc_data(src_dir, dst_dir):
    file_list = ('VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar')
    for dep_filename in file_list:
        print("Copying %s" % dep_filename)
        shutil.copyfile(os.path.join(src_dir, dep_filename), os.path.join(dst_dir, dep_filename))
    
    for dep_filename in file_list:
        tar = tarfile.open(os.path.join(dst_dir, dep_filename), 'r')
        print("Extracting files from %s. Please wait..." % dep_filename)
        tar.extractall(dst_dir)
        print("Done.")
        tar.close()
        os.remove(os.path.join(dst_dir, dep_filename))

if __name__ == '__main__':
    src_dir = r'\\ivm-server2\IRIS\IRISObjectDetection\Data'
    dst_dir = 'data'
    
    download_voc_data(src_dir, dst_dir)
    
    train = pascal_voc(os.path.join(dst_dir, r'VOCdevkit\VOC2007'), 'trainval')
    test = pascal_voc(os.path.join(dst_dir, 'VOCdevkit\VOC2007'), 'test')
    
    classeslist = [['horse'],  
                   ['horse', 'bird'],
                   ['horse', 'dog', 'bird', 'boat', 'bottle'],
                   ['horse', 'dog', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow'],
                   ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
                   ];
    for classes in classeslist:
        datafolder = 'voc%d'%(len(classes))
        train.dump_data_tsv(os.path.join(dst_dir, datafolder, 'train.tsv'), classes)
        test.dump_data_tsv(os.path.join(dst_dir, datafolder, 'test.tsv'), classes)
        save_labelmap(os.path.join(dst_dir, datafolder, 'labelmap.txt'), classes)
    

