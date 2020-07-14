#%%
import argparse
import gc
import os
import random
import time
import warnings
from datetime import datetime

import albumentations as albu
import cv2
import easydict
import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations import pytorch as AT  # 나중에 사용
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split  # 나중에 사용
from torch import sigmoid
# from catalyst.utils import set_global_seed, prepare_cudnn
# from catalyst import utils
# from catalyst.dl.runner import SupervisedRunner
# from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, tqdm_notebook, trange

from configs import getConfig
from model.data_loader import kfold_data_generator
from model.dataset import MSI_train_Dataset, MSI_valid_Dataset
from model.net import fpn, unet
from utils import save_setting_info

# arg = getConfig()
arg = easydict.EasyDict({
    "model_name": 'unet',
    "k_folds": 1,
    "BACKBONE": 'resnet34',
    "PRETRAINED": None,
    "epochs": 150,
    "BATCH_SIZE": 32,
    'lr' : 0.001,
    'WORKERS' : 0,
    'Threshold' : 0.6
    })

# 구글스프레드시트 설정
scope = [
'https://spreadsheets.google.com/feeds',
'https://www.googleapis.com/auth/drive',
]

json_file_name = 'money-check-260910-c217b2d4ba6a.json'

credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope) 
gs = gspread.authorize(credentials)


spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1tpboitN1otWUB9sG7MQdoYG70pcxNZqBPBGZhp_ExX0/edit#gid=0'

# 스프레스시트 문서 가져오기 
doc = gs.open_by_url(spreadsheet_url)

# 시트 선택하기
worksheet = doc.worksheet('시트1')


warnings.filterwarnings("ignore")

# %%
def get_training_augmentation():
    train_transform = [albu.HorizontalFlip(p=0.5)]
                       #albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                       #albu.GridDistortion(p=0.5),
                       #albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)]
    
    return albu.Compose(train_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
    ]
    return albu.Compose(_transform)

def to_tensor(x, **kwargs):
    """
    Convert image or mask.

    Args:
        x:
        **kwargs:

    Returns:

    """

    return x.transpose(2, 0, 1).astype('float32')

def tensor_imshow(image):
    np_image = np.transpose(image.cpu().numpy(), axes=(1, 2, 0))
    cv2.imshow('image', np_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dice_no_threshold(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    #activation: str = "Sigmoid"
):

    #activation_fn = get_activation_fn(activation)
    #outputs = activation_fn(outputs)
    outputs = sigmoid(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice


# %%
# backbone
model_name = arg.model_name
k_folds = arg.k_folds
BACKBONE = arg.BACKBONE
PRETRAINED = arg.PRETRAINED
BATCH_SIZE = arg.BATCH_SIZE
WORKERS = arg.WORKERS
epochs = arg.epochs
Threshold = arg.Threshold
lr = arg.lr
worker = 'Doohyung'

time_path = datetime.today().strftime('%Y-%m-%d_%H%M%S')

# set model
if model_name == 'fpn':
    model = fpn(backbone = BACKBONE, pretrained_weights = PRETRAINED, activation = 'sigmoid')
else:
    model = unet(backbone = BACKBONE, pretrained_weights = PRETRAINED, activation = 'sigmoid')

#preprocessing_fn = smp.encoders.get_preprocessing_fn(BACKBONE, PRETRAINED)

train_dataset = MSI_train_Dataset()
valid_dataset = MSI_valid_Dataset()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

writer = SummaryWriter(os.path.join('./results', time_path))

save_setting_info(arg, os.path.join('./results', time_path), time_path)

# 구글스프레드 시트 저장
index = 1
while 1:
    value_list = worksheet.row_values(index)
    if value_list == []:
        index -= 1
        break
    index += 1

valid_dice_score_max = 0
valid_dice_epoch_max = 0

arg_list = list(arg.values())
arg_list.insert(0,time_path)
arg_list.insert(0,index)
arg_list.append(valid_dice_score_max)
arg_list.append(valid_dice_epoch_max)
arg_list.append(worker)
arg_list[5] = 'None'
worksheet.append_row(arg_list)

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, criterion, optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr= lr)
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
dice_criterion = smp.utils.losses.DiceLoss()
bce_criterion = nn.BCELoss()

model = nn.DataParallel(model.to(device))


train_start_time = time.time()

valid_dice_score_max = 0
for epoch in range(1, epochs+1):
    train_epoch_loss = 0
    train_dice_score = 0

    model.train()
    # Training
    for i, (images, mask) in enumerate(tqdm(train_loader)):
        images = torch.tensor(images, device=device, dtype=torch.float32)
        mask  = torch.tensor(mask, device=device, dtype=torch.float32)

        output = model(images)
        dice_loss = dice_criterion(output, mask)     # Output & Mask Shape: (Batch, 1, 512, 512) 
        # bce_loss = bce_criterion(output, mask)
        loss = dice_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()
        dice_cof = dice_no_threshold(output.cpu(), mask.cpu(), threshold=Threshold).item()
        train_dice_score += dice_cof * images.size(0)
        
    # calculate average losses
    train_epoch_loss = train_epoch_loss/len(train_loader)
    train_dice_score = train_dice_score/len(train_loader.dataset)

    print(f'Epoch:{epoch}')
    print(f'Train: loss:{train_epoch_loss:.3f} | dice score:{train_dice_score:.3f}')

    valid_epoch_loss = 0
    valid_dice_score = 0
    model.eval()

    with torch.no_grad():
        for (images, mask) in enumerate(tqdm(valid_loader)):
            images = torch.tensor(images, device=device, dtype=torch.float32)
            mask  = torch.tensor(mask, device=device, dtype=torch.float32)

            output = model(images)
            
            dice_loss = dice_criterion(output, mask)     # Output & Mask Shape: (Batch, 1, 512, 512) 
            # bce_loss = bce_criterion(output, mask)
            loss = dice_loss
            valid_epoch_loss += loss.item()

            dice_cof = dice_no_threshold(output.cpu(), mask.cpu(), threshold=Threshold).item()
            valid_dice_score += dice_cof * images.size(0)

        # calculate average losses
        valid_epoch_loss = valid_epoch_loss/len(valid_loader)
        valid_dice_score = valid_dice_score/len(valid_loader.dataset)

        print(f'valid_dice_score : {valid_dice_score}')
        print(f'valid_dice_score_max : {valid_dice_score_max}')

        if valid_dice_score > valid_dice_score_max:
            torch.save(model.module.state_dict(), './results/{}/best_model.pt'.format(time_path))
            print('model_saved')
            valid_dice_score_max = valid_dice_score
            valid_dice_epoch_max = epoch

            index = 2
            while 1:
                cell = 'B' + str(index)
                cell_value = worksheet.acell(cell).value
                if cell_value == time_path:
                    break
                index += 1
            cell1 = 'L'+ str(index)
            cell2 = 'M'+ str(index)
            worksheet.update_acell(cell1, valid_dice_score_max)
            worksheet.update_acell(cell2, valid_dice_epoch_max)

        if epoch % 5 == 0:
            torch.save(model.module.state_dict(), './results/{}/epoch_{}.pt'.format(time_path, epoch))
            print('model_saved')


        print(f'Valid: loss:{valid_epoch_loss:.3f} | dice score:{valid_dice_score:.3f}')

    scheduler.step(valid_dice_score)

    writer.add_scalar('train loss', train_epoch_loss, epoch)
    writer.add_scalar('train score', train_dice_score, epoch)
    writer.add_scalar('valid loss', valid_epoch_loss, epoch)
    writer.add_scalar('valid score', valid_dice_score, epoch)

    del images, mask
    gc.collect()
    torch.cuda.empty_cache()



'''
fold 필요할때 사용
for f_idx, (train_idx, valid_idx) in enumerate(folds):
    train_df = all_patches_sample.iloc[train_idx]
    valid_df = all_patches_sample.iloc[valid_idx]
'''   

'''
결과 표로 볼때 사용
utils.plot_metrics(
logdir=logdir, 
# specify which metrics we want to plot
metrics=["loss", "dice", 'lr', '_base/lr'])
'''


# %%
