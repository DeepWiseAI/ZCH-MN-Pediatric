import os
import PIL
import torch
import numpy as np

from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset

import json



class SkinDataset(Dataset):
    def __init__(self, json_file, root_path, seg_path, transform=None):
        self.data = json.load(open(json_file))
        self.transform = transform
        # here, 借鉴了 camel_data.py 中的代码
        self.pat_id = [] 
        self.imgpath, self.lesion_labels = [], []
        self.root_path = root_path
        self.seg_path = seg_path
        for pid, nodes in self.data.items():
            for node in nodes:
                node['imgpath'] = [
                        os.path.join(root_path, imgpath) #
                        for imgpath in node['imgpath']
                    ]
                
                self.imgpath.append(node['imgpath'])
                self.lesion_labels.append(node['label'])
                self.pat_id.append(pid)

    def __len__(self):
        return len(self.imgpath)
    
    def __getitem__(self, idx):
        lesion_label = self.lesion_labels[idx] 
        full_path = self.imgpath[idx]
        pat_id = self.pat_id[idx]
        images = []
        for imgp in full_path:
            image = Image.open(imgp).convert('RGB')
            maskpath = imgp.replace(self.root_path, self.seg_path)
            mask = Image.open(maskpath).convert('L')

            image_np = np.array(image)
            mask_np = np.array(mask)
            nonzero_indices = np.nonzero(mask_np)
            if len(nonzero_indices[0]) > 0:
                top_left = (min(nonzero_indices[0]),min(nonzero_indices[1]))
                bottom_right = (max(nonzero_indices[0]),max(nonzero_indices[1]))
            else:
                top_left = (0,0)
                bottom_right = (image.size[0],image.size[1]) 
            cropped_image = image_np[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
            cropped_image_pil = Image.fromarray(cropped_image)

            if self.transform:
                image = self.transform(cropped_image_pil)
            images.append(image)
        images = torch.stack(images)
        images = images.mean(dim=0)
        
        if images.shape[0] != 3:  
            raise ValueError(f"Expected 3 channels but got {images.shape[0]}")
        
        return images, '', lesion_label, pat_id, full_path