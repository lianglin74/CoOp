import cv2
from deprecated import deprecated
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.dataloader import default_collate

from ..lib.dataset import TSVDataset, TSVDatasetPlusYaml, TSVDatasetWithoutLabel, CropClassTSVDataset, CropClassTSVDatasetYaml

def get_train_data_loader(args):
    train_dataset = _get_dataset("train", args)

    if args.balance_sampler:
        assert not args.balance_class and not args.distributed
        train_sampler = make_class_balanced_sampler(train_dataset)
    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True,
            sampler=train_sampler)

    return train_dataset, train_loader, train_sampler

@deprecated("use get_train_data_loader or get_test_data_loader")
def get_data_loader(args):
    train_transform = get_pt_transform("train", args)
    test_transform = get_pt_transform("test", args)

    if args.data.endswith('.yaml'):
        train_dataset = CropClassTSVDatasetYaml(args.data, session_name='train', transform=train_transform, enlarge_bbox=args.enlarge_bbox)
    else:
        raise NotImplementedError()

    if args.balance_sampler:
        assert not args.balance_class and not args.distributed
        train_sampler = make_class_balanced_sampler(train_dataset)
    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

    shuffle = (train_sampler is None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.data.endswith('.yaml'):
        val_dataset = CropClassTSVDatasetYaml(args.data, session_name='val', transform=test_transform, enlarge_bbox=args.enlarge_bbox)
    else:
        raise NotImplementedError()
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.debug:
        # imgdir = os.path.join(args.output_dir, "imgs")
        # if not os.path.exists(imgdir):
        #     os.mkdir(imgdir)
        # target_labels = {0:0, 1:0, 2:0, 3:0}
        # for idx in range(len(train_dataset)):
        #     img, label = train_dataset[idx]
        #     if label.item() in target_labels and target_labels[label.item()]<10:
        #         save_image(img, os.path.join(imgdir, "train_{}_{}.jpg".format(label.item(), idx)))
        #         target_labels[label.item()] += 1

        # target_labels = {0:0, 1:0, 2:0, 3:0}
        # for idx in range(len(val_dataset)):
        #     img, label = val_dataset[idx]
        #     if label.item() in target_labels and target_labels[label.item()]<10:
        #         save_image(img, os.path.join(imgdir, "val_{}_{}.jpg".format(label.item(), idx)))
        #         target_labels[label.item()] += 1

        sample_counts = [0] * train_dataset.label_dim()
        for i, (input, target) in enumerate(train_loader):
            if i%100==0:
                print(i)
            for t in range(len(target)):
                sample_counts[target[t].item()] += 1
        import ipdb; ipdb.set_trace()
    return train_loader, val_loader, train_sampler, train_dataset

def my_collate(batch):
    imgs = default_collate([item[0] for item in batch])
    cols = [item[1] for item in batch]
    return [imgs, cols]

def get_testdata_loader(args):
    if not args.data.lower().endswith('.yaml'):
        raise NotImplementedError()
    if args.opencv:
        test_transform = cv_transform
    else:
        test_transform = get_pt_transform("test", args)

    test_dataset = CropClassTSVDatasetYaml(args.data, session_name='test',
            transform=test_transform, enlarge_bbox=args.enlarge_bbox)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=my_collate,
        num_workers=args.workers, pin_memory=True)

    return test_loader

def _get_dataset(phase, args):
    if phase == "train":
        transform = get_pt_transform("train", args)
    else:
        assert phase == "test"
        if args.opencv:
            transform = cv_transform
        else:
            transform = get_pt_transform("test", args)

    assert args.data.endswith("yaml"), "not supported data type: {}".format(args.data)
    return CropClassTSVDatasetYaml(args.data, session_name=phase, transform=transform, enlarge_bbox=args.enlarge_bbox)

def get_pt_transform(phase, args):
    rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.input_size == 224:
        target_size = 256
        crop_size = 224
    elif args.input_size == 112:
        target_size = 128
        crop_size = 112
    else:
        raise ValueError("input size {} not supported".format(args.input_size))

    if phase.lower() == "test":
        if hasattr(args, "data_aug") and args.data_aug == 1:
            test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(target_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    # rgb_normalize,
                ])
        else:
            test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(target_size),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    rgb_normalize,
                ])
        return test_transform

    if phase.lower() == "train":
        if args.data_aug == 0:
            train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    # transforms.RandomAffine(degrees=10),
                    transforms.RandomResizedCrop(crop_size, scale=(0.25,1), ratio=(2./3., 3./2.)),
                    transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0.5, hue=0.1),
                    transforms.ToTensor(),
                    rgb_normalize,
                ])
        elif args.data_aug == 1:
            train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(crop_size, scale=(0.25,1)),
                    # transforms.ColorJitter(brightness=(0.66667, 1.5), contrast=0, saturation=(0.66667, 1.5), hue=(-0.1, 0.1)),
                    # transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    # rgb_normalize,
                ])
        elif args.data_aug == 2:
            train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    # transforms.RandomAffine(0, shear=15),
                    transforms.RandomAffine(degrees=10),
                    # transforms.RandomAffine(0, shear=15),
                    transforms.RandomResizedCrop(crop_size, scale=(0.25,1), ratio=(2./3., 3./2.)),
                    transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0.5, hue=0.1),
                    transforms.ToTensor(),
                    rgb_normalize,
                ])
        elif args.data_aug == 3:
            train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(0, shear=15),
                    transforms.RandomAffine(degrees=10),
                    transforms.RandomAffine(0, shear=15),
                    transforms.RandomResizedCrop(crop_size, scale=(0.25,1), ratio=(2./3., 3./2.)),
                    # transforms.ColorJitter(brightness=(0.66667, 1.5), contrast=0, saturation=(0.66667, 1.5), hue=(-0.1, 0.1)),
                    transforms.ToTensor(),
                    rgb_normalize,
                ])
        elif args.data_aug == 4:
            train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(crop_size, scale=(0.25,1), ratio=(2./3., 3./2.)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    rgb_normalize,
                ])
        else:
            raise ValueError()
        return train_transform

    raise ValueError("unknow phase: {}".format(phase))

def cv_transform(imageMat):
    # TODO: put constant in one place
    targetSize = 256
    centerCropSize = 224
    pytorch_mean_bgr = [0.406, 0.456, 0.485]
    pytorch_std_bgr = [0.225, 0.224, 0.229]
    stdValue = 255*np.array(pytorch_std_bgr)
    meanValue = np.array(pytorch_mean_bgr) / np.array(pytorch_std_bgr)

    height, width, _ = imageMat.shape

    if (width <= height and width != targetSize):
        height = targetSize * height / width
        width = targetSize
    elif (width > height and height != targetSize):
        width = targetSize * width / height
        height = targetSize
    imageMat1 = cv2.resize(imageMat, (width, height))

    top = int(round((height - centerCropSize)/2.0))
    left = int(round((width - centerCropSize)/2.0))
    imageMat2 = imageMat1[top: top+centerCropSize, left: left+centerCropSize, :]
    imageMat3 = imageMat2 / stdValue
    imageMat3 = imageMat3 - meanValue
    imageMat4 = torch.from_numpy(np.transpose(imageMat3, [2,0,1])).type(torch.FloatTensor)
    return imageMat4

def make_class_balanced_sampler(train_dataset):
    weight_per_class = []
    for count in train_dataset.label_counts:
        if count == 0:
            weight_per_class.append(0.0)
        else:
            weight_per_class.append(max(len(train_dataset)/float(count), 9e-4))
    assert len(weight_per_class) == train_dataset.label_dim()
    weights = torch.tensor([0.] * len(train_dataset))
    for i in range(len(train_dataset)):
        weights[i] = weight_per_class[train_dataset.get_target(i)]
    return torch.utils.data.WeightedRandomSampler(weights, len(train_dataset))
