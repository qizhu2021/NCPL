import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import os.path
import cv2
import torchvision


def make_dataset(image_list, labels, args):
    if labels is not None:
        len_ = len(image_list)
        # images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
        images = [(image_list[i].split()[0], labels[i]) for i in range(len_)]
    else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def make_confident_dataset(image_list, pseudo_labels, real_labels, args):
    len_ = len(image_list)
    # images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    images = [(image_list[i].split()[0], pseudo_labels[i], real_labels[i]) for i in range(len_)]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, args, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_confident_dataset(image_list, labels, args)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList(Dataset):
    def __init__(self, image_list, args, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels, args)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" +
                                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageListWithPath(Dataset):
    def __init__(self, image_list, args, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels, args)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" +
                                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, args, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels, args)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" +
                                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_confident(Dataset):
    def __init__(self, image_list, args, pseudo_labels=None, real_labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_confident_dataset(image_list, pseudo_labels, real_labels, args)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" +
                                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target, real = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, real, index

    def __len__(self):
        return len(self.imgs)
