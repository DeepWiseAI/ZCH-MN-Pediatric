import os
import PIL
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset
import csv
import glob
import pandas as pd


class inter_dataset(Dataset):
    def __init__(self, dataset_folder='ZCH-Data-test',suffix=None, train_type='train', transform=None):
        self.transform = transform
        self.train_type = train_type
        self.image_path = dataset_folder 
        self.imagelist = []
        for id in os.listdir(self.image_path):
            if suffix is None:
                self.imagelist.extend(glob.glob(os.path.join(self.image_path, id, '*')))
            else:
                self.imagelist.extend(glob.glob(os.path.join(self.image_path, id, '*'+suffix)))
                

    def __getitem__(self, item: int):
        image = Image.open(self.imagelist[item]).convert('RGB')
        label = Image.open(self.imagelist[item]).convert('RGB')

        sample = {'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets
            sample = self.transform(sample, self.train_type)
            sample['path'] = self.imagelist[item]
        return sample['image'], sample['label'], sample['path']

    def __len__(self):
        return len(self.imagelist)
