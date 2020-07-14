#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import albumentations as albu
import numpy.ma as ma # mask 만들기

from albumentations import pytorch as AT # 나중에 사용
from torch.utils.data import Dataset
from model.data_loader import train_PatchLoader
from model.data_loader import valid_PatchLoader


# In[2]:


class MSI_train_Dataset(Dataset):
    # 커스텀 dataset
    def __init__(self, transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]), preprocessing=None, k_folds=1):
        
        patch_loader = train_PatchLoader(n_kfold=k_folds, seed=42)
        df = patch_loader.get_all_patches()
        
        self.len = df.shape[0]
        self.slide_path = df['slide_path']
        self.mask_path = df['mask_path']
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        slide_path = self.slide_path[idx]
        mask_path = self.mask_path[idx]
        
        slide = cv2.imread(slide_path)
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)    # Gray
#        mask = np.transpose(mask, axes=(2, 0, 1))
        mask = mask/255.0
        augmented = self.transforms(image=slide, mask=mask)
        slide = augmented['image']
        mask = augmented['mask']
        mask = ma.make_mask(mask, shrink=False)
        mask = mask.squeeze()
        mask = np.expand_dims(mask, axis=0)

        
        if self.preprocessing:
            preprocessed = self.preprocessing(image=slide, mask=mask)
            slide = preprocessed['image']
            mask = preprocessed['mask']
        
        return slide, mask

    def __len__(self):
        return self.len


# In[3]:


class MSI_valid_Dataset(Dataset):
    # 커스텀 dataset
    def __init__(self, transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]), preprocessing=None, k_folds=1):
        
        patch_loader = train_PatchLoader(n_kfold=k_folds, seed=42)
        df = patch_loader.get_all_patches()
        
        self.len = df.shape[0]
        self.slide_path = df['slide_path']
        self.mask_path = df['mask_path']
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        slide_path = self.slide_path[idx]
        mask_path = self.mask_path[idx]
        
        slide = cv2.imread(slide_path)
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#        mask = np.transpose(mask, axes=(2, 0, 1))
        mask = mask/255.0
        augmented = self.transforms(image=slide, mask=mask)
        slide = augmented['image']
        mask = augmented['mask']
        mask = ma.make_mask(mask, shrink=False)
        mask = mask.squeeze()
        mask = np.expand_dims(mask, axis=0)


        if self.preprocessing:
            preprocessed = self.preprocessing(image=slide, mask=mask)
            slide = preprocessed['image']
            mask = preprocessed['mask']
        
        return slide, mask

    def __len__(self):
        return self.len


# In[ ]:




