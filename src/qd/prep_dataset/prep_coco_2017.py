import random
from random import shuffle
import logging
import os.path as op
import json
import base64
import cv2
from qd.process_tsv import tsv_writer
from qd.qd_common import init_logging
import copy


def prep_coco(coco_root, json_name, image_folder, out_tsv_file):
    annfile = op.join(coco_root, 'annotations', json_name);
    imgfolder = op.join(coco_root, 'images', image_folder);
    with open(annfile,'r') as jsin:
        print("Loading annotations...")
        truths = json.load(jsin)
        #map id to filename
        imgdict = {x['id']:x['file_name'] for x in truths['images']};
        catdict = {x['id']:x['name'] for x in truths['categories']};
        anndict = { x:[] for x in imgdict };
        for ann in truths['annotations']:
            crect = copy.deepcopy(ann)
            imgid = ann['image_id'];
            bbox = ann['bbox'];
            bbox[2] += bbox[0]-1;
            bbox[3] += bbox[1]-1;
            cid = ann['category_id'];
            del crect['image_id']
            del crect['bbox']
            del crect['category_id']
            crect['class'] = catdict[cid]
            crect['rect'] = bbox
            anndict[imgid]+=[crect];

    cnames=sorted(catdict.values());
    with open("labelmap.txt","w") as tsvout:
        for cname in cnames:
            tsvout.write(cname+"\n")
    print("Saving tsv...")

    def generate_tsv_row():
        random.seed(666)
        image_ids = anndict.keys()
        logging.info('shuffle the list ({})'.format(len(image_ids)))
        shuffle(image_ids)

        for i, image_id in enumerate(image_ids):
            imgf = op.join(imgfolder,imgdict[image_id]);
            im = cv2.imread(imgf)
            if im is None:
                logging.info('{} is not decodable'.format(imgf))
                continue
            with open(imgf, 'rb') as f:
                image_data = base64.b64encode(f.read());
            if (i % 100) == 0:
                logging.info(i)
            yield str(image_id), json.dumps(anndict[image_id]), image_data

    tsv_writer(generate_tsv_row(), out_tsv_file)


def test_prep_coco_full():
    coco_root = op.expanduser('~/data/raw_data/raw_coco/')
    # Full: with all meta data in the label fields, including iscrowded and
    # segmentation meta
    out_folder = './data/coco2017Full'
    input_output = []
    input_output += [('instances_train2017.json',
        'train2017',
        op.join(out_folder, 'train.tsv'))];
    input_output += [('instances_val2017.json',
        'val2017',
        op.join(out_folder, 'test.tsv'))];

    for json_name, image_folder, out_tsv_file in input_output:
        prep_coco(coco_root, json_name, image_folder, out_tsv_file)

def load_coco_annotation(annfile):
    with open(annfile,'r') as jsin:
        print("Loading annotations...")
        truths = json.load(jsin)
        #map id to filename
        imgdict = {x['id']:x['file_name'] for x in truths['images']};
        catdict = {x['id']:x['name'] for x in truths['categories']};
        anndict = { y:[] for x, y in imgdict.items() };
        for ann in truths['annotations']:
            imgid = imgdict[ann['image_id']];
            bbox = ann['bbox'];
            bbox[2] += bbox[0]-1;
            bbox[3] += bbox[1]-1;
            cid = ann['category_id'];
            crect = {'class':catdict[cid], 'rect':bbox}
            anndict[imgid] += [crect];
    return anndict

def test_prep_coco():
    '''
    deprecated. problem: it does not save the iscrowded field, used by coco
    evaluation; it does not keep the segmentation information. use
    test_prep_coco_full
    '''
    coco_root = op.expanduser('~/data/raw_data/raw_coco/')
    ann_folder = op.join(coco_root, "annotations");
    truthlocs = [('instances_train2017.json','train2017')];
    truthlocs = [('instances_val2017.json','val2017')];
    out_folder = './data/coco2017'

    for datasets in truthlocs:
        annfile = op.join(ann_folder, datasets[0]);
        imgfolder = op.join(coco_root, datasets[1]);
        with open(annfile,'r') as jsin:
            print("Loading annotations...")
            truths = json.load(jsin)
            #map id to filename
            imgdict = {x['id']:x['file_name'] for x in truths['images']};
            catdict = {x['id']:x['name'] for x in truths['categories']};
            anndict = { x:[] for x in imgdict };
            for ann in truths['annotations']:
                imgid = ann['image_id'];
                bbox = ann['bbox'];
                bbox[2] += bbox[0]-1;
                bbox[3] += bbox[1]-1;
                cid = ann['category_id'];
                crect = {'class':catdict[cid], 'rect':bbox}
                anndict[imgid]+=[crect];

        cnames=sorted(catdict.values());
        with open("labelmap.txt","w") as tsvout:
            for cname in cnames:
                tsvout.write(cname+"\n")
        print("Saving tsv...")

        def generate_tsv_row():
            image_ids = anndict.keys()
            logging.info('shuffle the list ({})'.format(len(image_ids)))
            shuffle(image_ids)

            for i, image_id in enumerate(image_ids):
                imgf = op.join(imgfolder,imgdict[image_id]);
                im = cv2.imread(imgf)
                if im is None:
                    logging.info('{} is not decodable'.format(imgf))
                    continue
                with open(imgf, 'rb') as f:
                    image_data = base64.b64encode(f.read());
                if (i % 100) == 0:
                    logging.info(i)
                yield str(image_id), json.dumps(anndict[image_id]), image_data

        tsv_writer(generate_tsv_row(), op.join(out_folder, datasets[1] +
            '.tsv'))

def test_generate_label_to_catid():
    coco_root = op.expanduser('~/data/raw_data/raw_coco/')
    ann_folder = op.join(coco_root, "annotations");
    from qd.qd_common import read_to_buffer
    x = json.loads(read_to_buffer(op.join(ann_folder,
        'instances_train2017.json')))
    cat = x['categories']
    from qd.qd_common import write_to_yaml_file
    write_to_yaml_file(cat, './data/coco2017Full/coco_categories.yaml')

if __name__ == '__main__':
    init_logging()
    #test_prep_coco()
    test_prep_coco_full()
    test_generate_label_to_catid()

