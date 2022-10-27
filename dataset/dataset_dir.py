import glob
import os
import pickle as pkl

import albumentations as A
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from .randaugment_loc import RandAugmentMC
from PIL import Image
import pandas as pd

size = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB
# def get_train_transforms(epoch, dim = config.im_size):
#     return A.Compose(
#         [             
#             # resize like Resize in fastai
#             A.SmallestMaxSize(max_size=dim, p=1.0),
#             A.RandomCrop(height=dim, width=dim, p=1.0),
#             A.VerticalFlip(p = 0.5),
#             A.HorizontalFlip(p = 0.5)
#             #A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#         ]
#   )
transform_param = A.ReplayCompose(
    [
        # A.RandomResizedCrop(size, size, scale=(0.33, 1.0), ratio=(0.7, 1.35)),
        A.GaussNoise(var_limit=(10.0, 20.0), p=0.15),
        A.MotionBlur(blur_limit=[5, 20], p=0.15),
        A.Downscale(scale_min=0.25, scale_max=0.5,p=0.2),
        A.ISONoise(color_shift=(0.03, 0.1), intensity=(0.3, 0.8),p=0.15), # ToDO not skip but except from all
        A.JpegCompression(quality_lower=10, quality_upper=50, p=0.15),
        #MedianBlur
        #MultiplicativeNoise
        # Superpixels
        # # A.GaussianBlur(p=0.15),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.3),
        # # A.none
        # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
transform_weak = A.Compose(
    [
        A.OneOf([
            A.RandomResizedCrop(size, size, scale=(0.3, 0.8), ratio=(0.9, 1.1),interpolation=cv2.INTER_CUBIC),
            A.RandomSizedCrop([size,2*size],size,size,interpolation=cv2.INTER_CUBIC),
            # A.RandomCrop()
            ],p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3
        ),

        A.CoarseDropout(
            max_height=24, max_width=24, p=0.1
        ),

    ]
)

transform_test = A.Compose(
    [
        # # A.SmallestMaxSize(max_size=int(1.15*size), p=1.0),
        # # A.CenterCrop(size, size),
        # A.Resize(size, size),
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
    ]
)


def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], value=255.0):
  
    image = image.astype(np.float32)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    image = image / value
    image = (image - mean) / std

    return image



class DatasetDir(Dataset):
    """Face Landmarks dataset."""

    def __init__(
            self, data_dir, pkl_path=None, idx=None, state="train", tensor_norm=False, is_add=False
    ):  # , loader=default_loader):
        self.state = state
        # self.transform = transform
        self.data_dir = data_dir
        self.tensor_norm=tensor_norm
        self.idx = idx
        self.names=os.listdir(data_dir)
        # self.list_dataset=isinstance(self.data,list)
        # self.is_add=is_add
        # print(len(transform_param)) 
        self.n_attrs=len(transform_param)+1
        if pkl_path:
            print('DICT dataset')
            self.data = pkl.load(open(pkl_path, 'rb'))
            self.attrs = self.data
            self.names = list(self.attrs.keys())
            # print(self.names[0])
            self.n_attrs = len(self.attrs[self.names[0]]) #.shape[0]

           
        # print(self.names[0])
        if idx is None:
            self.idx = np.arange(len(self.names))
       

        # self.paths += glob.glob(f"{data_dir}/**/*.jpg")

    def __getitem__(self, index0):

        try:
            index = self.idx[index0] #ToDO replace idx
            path = f'{self.data_dir}/{self.names[index]}'
            image = cv2.imread(path)
        
            if image is None:
                print(path)
       

            image=transform_weak(image=image)['image']
            if np.random.rand()>0.2:
                aug = transform_param(image=image)
                image=aug['image']
                aug_trans=aug['replay']['transforms']
                augs = [int(a['applied']) for a in aug_trans]

                augs=np.array(augs)      
                attrs=np.hstack((augs,[int((augs.sum()-augs[3])>0)]))
            else:
                attrs=np.zeros(self.n_attrs,np.int32)
           
            return self.to_tensor(image), attrs, path
        except:
            print(index, index0)
            return torch.zeros((3, size, size)).float(), np.zeros(self.n_attrs, np.int32), ''
            
    def to_tensor(self, x):
        if self.tensor_norm:
            x = normalize(x)
        elif x.dtype == np.uint8:
            x = x / 255
        x = x.transpose(2, 0, 1)
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.idx)


class DatasetImage(Dataset):
    """Face Landmarks dataset."""

    def __init__(
            self, data_dir, is_test=False, is_multi=False, tensor_norm=False, n_crops=16
    ):  # , loader=default_loader):

        self.data_dir = data_dir
        self.paths = glob.glob(f"{data_dir}/**/*.jpg")
        self.paths += glob.glob(f"{data_dir}/**/*.jpeg")
        self.paths += glob.glob(f"{data_dir}/*.jpg")
        # self.is_test=is_test
        if is_test:
            self.transform = transform_test
        else:
            self.transform = transform_train
        self.is_multi=is_multi
        self.is_test = is_test
        self.tensor_norm=tensor_norm
        self.n_crops=n_crops

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path)
        key = path.split('/')[-1].split('.')[0]
        images=[]
        try:
            for i in range(self.n_crops):
                image_cur = self.transform(image=image)["image"]
                images.append(self.to_tensor(image_cur))
            #return torch.stack(images), path,1
            return images, path,1
        except:
            #return torch.zeros((self.n_crops,3,size,size)),path,-1
            return [torch.zeros((3,size,size))]*self.n_crops,path,-1
        
       


    def to_tensor(self, x):
        if self.tensor_norm:
            x = normalize(x)
        elif x.dtype == np.uint8:
            x = x / 255
        x = x.transpose(2, 0, 1)
        
        return torch.from_numpy(x).float()

    def __len__(self):
        return len(self.paths)


