from tqdm import tqdm
import torch
import json
import torchvision.transforms as transforms
import logging

from qd.mask.structures.bounding_box import BoxList

from qd.qd_pytorch import TSVSplitImage
from qd.tsv_io import TSVDataset


class MaskTSVDataset(TSVSplitImage):

    """Docstring for MaskTSVDataset. """

    def __init__(self, data, split, version=0,
                 transforms=None,
                 multi_hot_label=False,
                 label_type=None, # gradually not to use multi_hot_label
                 cache_policy=None, labelmap=None,
                 remove_images_without_annotations=True,
                 bgr2rgb=False,
                 max_box=300,
                 dict_trainer=False,
                 filter_box=True,
                 ):
        # filter_box: when doing testing, and we want to extract the feature
        # based on gt box, we should not do anything to filter the box.
        # we will not use the super class's transform, but uses transforms
        # instead
        super().__init__(data, split, version=version,
                cache_policy=cache_policy, transform=None, labelmap=labelmap)
        assert label_type in [None, 'multi_hot', 'multi_domain']
        if multi_hot_label:
            assert label_type == 'multi_hot'
        self.label_type = label_type
        self.transforms = transforms
        self.use_seg = False
        dataset = TSVDataset(data)
        assert dataset.has(split, 'hw')
        from qd.qd_pytorch import TSVSplitProperty
        self.hw_tsv = TSVSplitProperty(data, split, t='hw')
        self.all_key_hw = None
        self.dataset = dataset
        if remove_images_without_annotations:
            from qd.process_tsv import load_key_rects
            self.ensure_load_key_hw()
            key_rects = load_key_rects(dataset.iter_data(split, t='label',
                version=version))
            self.shuffle = [i for i, ((key, rects), (_, (h, w))) in enumerate(zip(key_rects,
                self.all_key_hw)) if self.will_non_empty(rects, w, h) > 0]
        else:
            self.shuffle = None
        self.multi_hot_label = multi_hot_label
        self._id_to_img_map = None
        self.bgr2rgb = bgr2rgb
        self.max_box = max_box

        if self.label_type == 'multi_domain':
            from qd.data_layer.transform import ConvertToDomainClassIndex
            self.multi_domain_idx_gen = ConvertToDomainClassIndex(data)

        import os.path as op
        if op.isfile(dataset.get_txt('attributemap')):
            self.attribute_map = dataset.load_txt('attributemap')
        else:
            self.attribute_map = None
        self.dict_trainer = dict_trainer
        self.filter_box = filter_box

    def ensure_load_key_hw(self):
        if self.all_key_hw is None:
            self.all_key_hw = []
            logging.info('loading hw')
            for i in tqdm(range(len(self.hw_tsv))):
                key, h, w = self.read_key_hw(i)
                self.all_key_hw.append((key, [h, w]))

    @property
    def id_to_img_map(self):
        self.ensure_load_key_hw()
        if self._id_to_img_map is None:
            self._id_to_img_map = {i: key for i, (key, _) in
                    enumerate(self.all_key_hw)}
        return self._id_to_img_map

    def get_keys(self):
        return [self.read_key_hw(i)[0] for i in range(len(self.hw_tsv))]

    def _tsvcol_to_label(self, col):
        anno = json.loads(col)
        return anno

    def will_non_empty(self, anno, w, h):
        # coco data has this kind of property
        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]
        anno = [a for a in anno if a['class'] in self.label_to_idx]
        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, (w, h), mode="xyxy")
        target = target.clip_to_image(remove_empty=True)
        return len(target) > 0

    def __getitem__(self, idx):
        if self.label_type == 'multi_hot':
            assert self.multi_hot_label
            img, target, idx = self.get_item_multi_hot_label(idx)
        elif self.label_type == 'multi_domain':
            img, target, idx = self.get_item_for_multi_domain_softmax(idx)
        else:
            img, target, idx = self.get_item_for_softmax(idx)
        if self.dict_trainer:
            return {'image': img, 'target': target, 'idx': idx}
        else:
            return img, target, idx

    def add_tightness(self, target, anno):
        tightness = [rect.get('tightness', 1) for rect in anno]
        tightness = torch.ones(len(anno))
        for i in range(len(tightness)):
            tightness[i] = anno[i].get('tightness', 1)
        target.add_field('tightness', tightness)

    def add_attributes(self, target, anno):
        if self.attribute_map is not None:
            attributes = [ [0] * 16 for _ in range(len(anno))]
            for i, obj in enumerate(anno):
                attributes[i][:len(obj["attributes_encode"])] = obj["attributes_encode"]
            attributes = torch.tensor(attributes)
            target.add_field("attributes", attributes)

    def get_image_ann(self, idx):
        cv_im, anno, key = super(MaskTSVDataset, self).__getitem__(idx)

        if self.bgr2rgb:
            import cv2
            cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)

        img = transforms.ToPILImage()(cv_im)
        if isinstance(anno, int):
            # this is image-level labels
            w, h = img.size
            anno = [{'class': str(anno),
                'rect': [0, 0, w, h]}]

        return img, anno

    def get_composite_source_idx(self):
        assert self.shuffle is None, 'not supported'
        return self.tsv.get_composite_source_idx()

    def get_item_multi_hot_label(self, idx):
        iteration, idx, max_iter = idx['iteration'], idx['idx'], idx['max_iter']
        if self.shuffle:
            idx = self.shuffle[idx]

        img, anno = self.get_image_ann(idx)

        # coco data has this kind of property
        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]

        anno = [a for a in anno if a['class'] in self.label_to_idx]

        from qd.qd_common import merge_class_names_by_location_id
        anno = merge_class_names_by_location_id(anno)

        # we randomly select the max_box results. in the future, we should put it
        # in the Transform
        max_box = self.max_box
        # it will occupy 10G if it is 300 for one image.
        if len(anno) > max_box:
            logging.info('maximum box exceeds {} and we will truncate'.format(max_box))
            import random
            random.shuffle(anno)
            anno = anno[:max_box]
        anno = [o for o in anno if 'rect' in o]
        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")
        # the first one is for background
        classes = torch.zeros((len(anno), len(self.label_to_idx)))
        for i, obj in enumerate(anno):
            for c, conf in zip(obj['class'], obj['conf']):
                classes[i, self.label_to_idx[c]] = conf
        target.add_field("labels", classes)

        self.add_tightness(target, anno)
        self.add_attributes(target, anno)

        assert not self.use_seg, 'not tested'
        #masks = [obj["segmentation"] for obj in anno]
        #from qd.mask.structures.segmentation_mask import SegmentationMask
        #masks = SegmentationMask(masks, img.size)
        #target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            trans_input = {'image': img,
                           'rects': target,
                           'iteration': iteration,
                           'max_iter': max_iter,
                           'dataset': self,
                           }
            trans_out = self.transforms(trans_input)
            img, target = trans_out['image'], trans_out['rects']

        return img, target, idx

    def get_item_for_multi_domain_softmax(self, idx):
        iteration, idx, max_iter = idx['iteration'], idx['idx'], idx['max_iter']
        if self.shuffle:
            idx = self.shuffle[idx]
        img, anno = self.get_image_ann(idx)
        # we randomly select the max_box results. in the future, we should put it
        # in the Transform
        max_box = self.max_box
        # it will occupy 10G if it is 300 for one image.
        if len(anno) > max_box:
            logging.info('maximum box exceeds {} and we will truncate'.format(max_box))
            import random
            random.shuffle(anno)
            anno = anno[:max_box]
        anno = [o for o in anno if 'rect' in o]
        if any('location_id' in a for a in anno):
            # in this case, all locations should be unique for now.
            assert len(set(a['location_id'] for a in anno)) == len(anno)

        # coco data has this kind of property
        anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]

        anno = [a for a in anno if a['class'] in self.label_to_idx]

        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = self.multi_domain_idx_gen({'label': anno})['label_idx']
        # 0 is the background
        target.add_field("labels", classes)

        self.add_tightness(target, anno)
        self.add_attributes(target, anno)

        if self.use_seg:
            masks = [obj["segmentation"] for obj in anno]
            from qd.mask.structures.segmentation_mask import SegmentationMask
            masks = SegmentationMask(masks, img.size)
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            trans_input = {'image': img,
                           'rects': target,
                           'iteration': iteration,
                           'max_iter': max_iter,
                           'dataset': self,
                           }
            trans_out = self.transforms(trans_input)
            img, target = trans_out['image'], trans_out['rects']

        return img, target, idx

    def get_item_for_softmax(self, idx):
        iteration, idx, max_iter = idx['iteration'], idx['idx'], idx['max_iter']
        if self.shuffle:
            idx = self.shuffle[idx]
        img, anno = self.get_image_ann(idx)
        # we randomly select the max_box results. in the future, we should put it
        # in the Transform
        max_box = self.max_box
        # it will occupy 10G if it is 300 for one image.
        if len(anno) > max_box:
            logging.info('maximum box exceeds {} and we will truncate'.format(max_box))
            import random
            random.shuffle(anno)
            anno = anno[:max_box]
        anno = [o for o in anno if 'rect' in o]
        #for o in anno:
            #o['rect'] = [-2, -2, -1, -1]
        #logging.info('debugging')
        if any('location_id' in a for a in anno):
            # in this case, all locations should be unique for now.
            assert len(set(a['location_id'] for a in anno)) == len(anno)

        if self.filter_box:
            # coco data has this kind of property
            anno = [obj for obj in anno if obj.get("iscrowd", 0) == 0]

            anno = [a for a in anno if a['class'] in self.label_to_idx]

        boxes = [obj["rect"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        # 0 is the background
        classes = [self.label_to_idx.get(obj["class"], -1) + 1 for obj in anno]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        self.add_tightness(target, anno)
        self.add_attributes(target, anno)

        if self.use_seg:
            masks = [obj["segmentation"] for obj in anno]
            from qd.mask.structures.segmentation_mask import SegmentationMask
            masks = SegmentationMask(masks, img.size)
            target.add_field("masks", masks)

        if self.filter_box:
            target = target.clip_to_image(remove_empty=True)
        #else:
            #target = target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            trans_input = {'image': img,
                           'rects': target,
                           'iteration': iteration,
                           'max_iter': max_iter,
                           'dataset': self,
                           }
            trans_out = self.transforms(trans_input)
            img, target = trans_out['image'], trans_out['rects']

        return img, target, idx

    def __len__(self):
        if self.shuffle:
            return len(self.shuffle)
        else:
            return super(MaskTSVDataset, self).__len__()

    def get_img_info(self, index):
        if self.shuffle:
            index = self.shuffle[index]
        key, h, w = self.read_key_hw(index)
        result = {'height': h, 'width': w}
        return result

    def read_key_hw(self, idx_after_shuffle):
        key, str_hw = self.hw_tsv[idx_after_shuffle]
        try:
            h, w = [int(x) for x in str_hw.split(' ')]
        except:
            x = json.loads(str_hw)
            if isinstance(x, list):
                assert len(x) == 1
                x = x[0]
            h, w = x['height'], x['width']
        return key, h, w


