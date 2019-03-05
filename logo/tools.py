import os
import os.path as op
import numpy as np
import shutil
import base64
import _init_paths

import cv2

from google_images_download import google_images_download
from scripts.qd_common import init_logging, read_to_buffer, load_list_file, json_dump
from logo.utils import is_image_file


class StaticMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.minValue = np.Inf
        self.maxValue = - np.Inf
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = []

    def update(self, val, n=1):
        self.val.append(val)

        if self.minValue > val:
            self.minValue = val

        if self.maxValue < val:
            self.minValue = val

        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def download_prototype_logos_by_google(data, destFolder):
    logo_list = load_list_file("data/{}/labelmap.txt".format(data))
    response = google_images_download.googleimagesdownload()
    
    if not op.isdir("data/{}".format(destFolder)):
        os.mkdir("data/{}".format(destFolder))

    logo_id = 0
    
    filelist = []

    for logo in logo_list:
        arguments = {"keywords":"{}".format(logo),"suffix_keywords":"logo", "limit":10,"print_urls":True}
        absolute_image_paths = response.download(arguments)
        img_id = 0
        for k, abs_paths in absolute_image_paths.items():
            for abs_path in abs_paths:
                basename = os.path.basename(abs_path)
                ext = os.path.splitext(basename)[1]
                # import ipdb; ipdb.set_trace()
                # filelist.append((k, abs_path))
                shutil.move(abs_path, "data/{}/{}".format(destFolder, "{}_{:06}{}".format(logo, img_id, ext)))
                img_id += 1

        logo_id += 1

#
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def walk_through_files(path, file_extension=IMG_EXTENSIONS):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            isContain = False
            for ext in file_extension:
                isContain = isContain or filename.endswith(ext)

            if isContain:
                yield os.path.join(dirpath, filename)
    return

def write_db_to_tsv(dest_path, phase, images):
    with open(os.path.join(dest_path, phase + '.tsv'), "w") as tsv_file:
        i = 0
        total = len(images)
        imagesKeys = images.keys()
        imagesKeys.sort(key=lambda v: v.upper())

        # print imagesKeys
        for full_index in imagesKeys:
            img, boxes = images[full_index]

            if not boxes:
                print("No annotation for {}".format(full_index))
                continue

            tsv_file.write("{}\t{}\t{}\n".format(full_index, json.dumps(boxes), img))
            i = i + 1

            if i%1000 ==0:
                print float(i)/float(total)
    return

def build_dataset(srcFolder, destFilder):
    filelist = walk_through_files(srcFolder)
    fileset = []
    for filename in filelist:
        fileset.append(filename)
    fileset.sort()
    with open(op.join(destFilder,"train.tsv"), 'w') as fout:
        for filename in fileset:
            im = cv2.imread(filename)
            if im is None:
                continue

            # im = cv2.imread(filename, cv2.IMREAD_COLOR)
            imgkey = op.splitext(op.basename(filename))[0]
            label={}

            h, w, c = im.shape

            label['rect'] = [0,0, w - 1, h -1]
            label['class'] = imgkey.split('_')[0]
            labels = []
            labels.append(label)

            def read_img(filename):
                with open(filename, 'r') as fp:
                    return base64.b64encode(fp.read())

            b64string = read_img(filename)

            fout.write('\t'.join([imgkey, json_dump(labels), b64string]))
            fout.write('\n')

def main():
    # download_prototype_logos_by_google('FlickrLogos-32', "Flickr32PrototypeLogos")
    build_dataset("data/Flickr32PrototypeLogos", "data/Flickr32PrototypeLogos")
    return

if __name__ == "__main__":
    init_logging()
    main()
