import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.dataloader import default_collate

from ..lib.dataset import TSVDataset, TSVDatasetPlusYaml, TSVDatasetWithoutLabel, CropClassTSVDataset, CropClassTSVDatasetYaml

def get_data_loader(args, logger):
    if not args.data.lower().endswith('.yaml'):
        raise NotImplementedError()
    else:
        train_dataset = CropClassTSVDatasetYaml(args.data, session_name='train', transform=get_pt_transform("train"))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if not args.data.lower().endswith('.yaml'):
        raise NotImplementedError()
    else:
        val_dataset = CropClassTSVDatasetYaml(args.data, session_name='val', transform=get_pt_transform("test"))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # TODO: move to unit test or debug mode
    if False:
        imgdir = os.path.join(args.output_dir, "imgs")
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)
        for idx in range(100):
            img, label = train_dataset[idx]
            save_image(img, os.path.join(imgdir, "train_{}_{}.jpg".format(label.item(), idx)))
            img, label = val_dataset[idx]
            save_image(img, os.path.join(imgdir, "val_{}_{}.jpg".format(label.item(), idx)))

    return train_loader, val_loader, train_sampler, train_dataset

def my_collate(batch):
    imgs = default_collate([item[0] for item in batch])
    cols = [item[1] for item in batch]
    return [imgs, cols]

def get_testdata_loader(args):
    if not args.data.lower().endswith('.yaml'):
        raise NotImplementedError()
    else:
        # NOTE: OpenCV transform is used in testing
        val_dataset = CropClassTSVDatasetYaml(args.data, session_name='test', transform=cv_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=my_collate,
        num_workers=args.workers, pin_memory=True)

    return val_loader


def get_pt_transform(phase):
    # NOTE: OpenCV loads image in BGR
    bgr_normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                     std=[0.225, 0.224, 0.229])
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            bgr_normalize,
        ])
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2,1)),
            transforms.RandomAffine(degrees=180),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5, hue=0.2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            bgr_normalize,
        ])
    if phase.lwoer() == "test":
        return test_transform
    elif phase.lower() == "train":
        return train_transform
    else:
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
