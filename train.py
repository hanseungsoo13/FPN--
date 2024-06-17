from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torchvision
import zipfile
import cv2
import torch
from PIL import Image
from torchvision import transforms
from Dataset import Customdataset
from FPN import FPN
from monai.metrics import DiceMetric
from monai.losses.dice import DiceLoss,one_hot
from tqdm import tqdm

device = torch.device('cuda')
print(torch.cuda.is_available())

model = FPN(final_channels=59).to(device)

batch_size = 5
num_epochs = 10
lr = 1e-3
optimizer=torch.optim.Adam(model.parameters())

os.chdir ('./Desktop/한승수/군바리/new_start/논문리뷰/FPN')

img_path = './dataset/jpeg_images/IMAGES/'
seg_path = './dataset/jpeg_masks/MASKS/'

dataset = Customdataset(img_path,seg_path)

train_data, val_data = torch.utils.data.random_split(dataset,[0.8,0.2])

train_loader = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data,batch_size,shuffle=False)

i,t=next(iter(train_loader))
a= type(i)
print(a)


def loss_dice(pred,target,C):
    #target_ = one_hot(target[:,None,...], num_classes=C)
    dice = DiceLoss(reduction='mean')
    loss = dice(pred,target)
    return loss


def train(train_loader=train_loader, valid_loader = val_loader, model=model):

    for epoch in range(num_epochs):
        loop = tqdm(train_loader, leave=True)
        mean_loss=[]
        t_loss=0

        model.train()

        for b_id, (x,y) in enumerate(loop):
            optimizer.zero_grad()
            pred = model(x)
            t_loss = loss_dice(pred,y,59)
            t_loss.backward()
            optimizer.step()
            
            l = t_loss.item()
            mean_loss.append(l)
            loop.set_postfix(loss = l)
            
        print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

        model.eval()
        with torch.no_grad():
            v_loss=0
            mean_v_loss=[]
            for b_id, (x,y) in enumerate(valid_loader):
                pred = model(x)
                v_loss = loss_dice(pred,y,59)
                v_l = v_loss.item()
                mean_v_loss.append(v_l)
            print(f"Mean Valid Loss was {sum(mean_v_loss)/len(mean_v_loss)}")

train(train_loader)