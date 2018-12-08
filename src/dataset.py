import os
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from xml.etree import ElementTree
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data.dataloader import default_collate
opj = os.path.join

import config


class CocoDataset(CocoDetection):
    def __getitem__(self, index):
        """
        Returns
        - path: (str) image file name
        - img: (Tensor) with size [C, H, W]
        - TODO: memory improvements ?
        - target_tensor: (list of Tensor) each list item with size [xc, yc, w, h, label]
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        w, h = img.size
        target = coco.loadAnns(ann_ids)
        annos = torch.zeros(len(target), 5)
        for i in range(len(target)):
            bbox = torch.Tensor(target[i]['bbox'])  # [x1, y1, w, h]
            label = config.datasets['coco']['category_id_mapping'][int(target[i]['category_id'])]
            annos[i, 0] = (bbox[0] + bbox[2] / 2) / w  # xc
            annos[i, 1] = (bbox[1] + bbox[3] / 2) / h  # yc
            annos[i, 2] = bbox[2] / w  # w
            annos[i, 3] = bbox[3] / h  # h
            annos[i, 4] = label  # 0-80
        if self.transform is not None:
            img = self.transform(img)
        return path, img, annos

    @staticmethod
    def collate_fn(batch):
        """Collate function for Coco DataLoader

        Returns
        - names: (tuple) each is a str of image filename
        - images: (Tensor) with size [bs, C, H, W]
        - annos: (tuple) each is a Tensor of annotations
        """
        names, images, annos = zip(*batch)
        images = default_collate(images)
        return names, images, annos


class VocDataset(torch.utils.data.dataset.Dataset):
    """Image datasets for PASCAL VOC

    Args
    - train_list: (str) full path to train list file
    """

    def __init__(self, train_list, transform):
        with open(train_list) as f:
            paths = f.readlines()
        self.img_paths = [x.strip() for x in paths]
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_tensor = self.transform(Image.open(img_path))
        img_label_path = img_path.replace('JPEGImages', 'labels').replace('.jpg', '.txt')
        img_anno = self.parse_label(img_label_path)
        return (img_path, img_tensor, img_anno)

    def __len__(self):
        return len(self.img_paths)

    def parse_label(self, label_path):
        """Parsing label

        Args
          label_path: (str) path to label file

        Returns
          img_anno: (Tensor) with size [#bbox, 5]
            offsets are scaled to (0,1) and in format [xc, yc, w, h, label]
        """
        bs = torch.Tensor(np.loadtxt(label_path))
        if len(bs.size()) == 1:  # only one object
            bs = bs.unsqueeze(0)

        img_anno = torch.zeros(bs.size())
        img_anno[:, :4] = bs[:, 1:]
        img_anno[:, 4] = bs[:, 0]

        return img_anno

    @staticmethod
    def collate_fn(batch):
        """Collate function for Voc DataLoader

        Returns
          paths: (tuple) each is a str of filepath to image
          images: (Tensor) with size [bs, C, H, W]
          annos: (tuple) each is a Tensor of annotations
        """
        names, images, annos = zip(*batch)
        images = default_collate(images)
        return names, images, annos


def prepare_train_dataset(name, reso, batch_size=32):
    """Prepare dataset for training

    Args  
    - name: (str) dataset name
    - reso: (int) training image resolution
    - batch_size: (int) default 32

    Returns
    - img_datasets: (CocoDataset) image datasets
    - trainloader: (Dataloader) dataloader for training
    """
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=reso, interpolation=3),
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.2),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    path = config.datasets[name]

    if name == 'coco':
        img_datasets = CocoDataset(root=path['train_imgs'], annFile=path['train_anno'], transform=transform)
        dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=CocoDataset.collate_fn)
    elif name == 'voc':
        img_datasets = VocDataset(train_list=path['train_imgs'], transform=transform)
        dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=4, shuffle=True, collate_fn=VocDataset.collate_fn)

    return img_datasets, dataloder


def prepare_val_dataset(name, reso, batch_size=32):
    """Prepare dataset for validation

    Args  
      name: (str) dataset name [tejani, hinter]
      reso: (int) validation image resolution
      batch_size: (int) default 32

    Returns
      img_datasets: (CocoDataset)
      dataloader: (Dataloader)
    """
    transform = transforms.Compose([
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ToTensor()
    ])

    path = config.datasets[name]

    if name == 'coco':
        img_datasets = CocoDataset(root=path['val_imgs'], annFile=path['val_anno'], transform=transform)
        dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=4, collate_fn=CocoDataset.collate_fn, shuffle=True)
    elif name == 'voc':
        img_datasets = VocDataset(train_list=path['val_imgs'], transform=transform)
        dataloder = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=4, collate_fn=VocDataset.collate_fn, shuffle=True)


    return img_datasets, dataloder


class DemoDataset(torch.utils.data.dataset.Dataset):
    """Dataset for evaluataion"""

    def __init__(self, imgs_dir, transform):
        """
        Args
          imgs_dir: (str) test images directory
          transform: (torchvision.transforms)
        """
        self.imgs_dir = imgs_dir
        self.imgs_list = os.listdir(imgs_dir)
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        img_path = os.path.join(self.imgs_dir, img_name)
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        return img_path, img_tensor

    def __len__(self):
        return len(self.imgs_list)


def prepare_demo_dataset(path, reso, batch_size=1):
    """Prepare dataset for demo

    Args
      path: (str) path to images
      reso: (int) evaluation image resolution
      batch_size: (int) default 1

    Returns
      img_datasets: (torchvision.datasets) demo image datasets
      dataloader: (DataLoader)
    """
    transform = transforms.Compose([
        transforms.Resize(size=(reso, reso), interpolation=3),
        transforms.ToTensor()
    ])

    img_datasets = DemoDataset(path, transform)
    dataloader = torch.utils.data.DataLoader(img_datasets, batch_size=batch_size, num_workers=8)

    return img_datasets, dataloader
