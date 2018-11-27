import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.dataloader import default_collate

from ..lib.dataset import TSVDataset, TSVDatasetPlusYaml, TSVDatasetWithoutLabel, CropClassTSVDataset, CropClassTSVDatasetYaml

def get_data_loader(args, logger):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2,1)),
            transforms.RandomAffine(degrees=180),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5, hue=0.2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if not args.data.lower().endswith('.yaml'):
        raise NotImplementedError()
    else:
        train_dataset = CropClassTSVDatasetYaml(args.data, session_name='train', transform=train_transform)

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
        val_dataset = CropClassTSVDatasetYaml(args.data, session_name='val', transform=test_transform)

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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])

    if not args.data.lower().endswith('.yaml'):
        raise NotImplementedError()
    else:
        val_dataset = CropClassTSVDatasetYaml(args.data, session_name='test', transform=test_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=my_collate,
        num_workers=args.workers, pin_memory=True)

    return val_loader
