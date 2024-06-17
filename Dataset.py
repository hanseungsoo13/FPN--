from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torchvision
import zipfile
import cv2
import torch
from PIL import Image
from torchvision import transforms

def data_unzip(path = './dataset'):
    os.makedirs(path,exist_ok=True)
    with zipfile.ZipFile('archive.zip','r') as zip_ref:
        zip_ref.extractall(path)

class Customdataset(Dataset):
    def __init__(self,
                 img_path,
                 mask_path,
                 transform = True,
                 device='cuda'
                 ):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.device = device

    def transformation(self,img):
        my_transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()])
        return my_transform(img)
        
    def __getitem__(self,index):
        img_open_path = str(self.img_path) + 'img_' + '0'*(4-len(str(index+1))) + str(index+1) + '.jpeg'
        mask_open_path = str(self.mask_path) + 'seg_' + '0'*(4-len(str(index+1))) + str(index+1) + '.jpeg'
        img = Image.open(img_open_path).convert("RGB")
        mask = Image.open(mask_open_path)
        
        if self.transform:
            img = self.transformation(img)
            mask = self.transformation(mask)

        img = img.to(self.device)
        mask = mask.to(self.device)
        
        return img,mask
    
    def __len__(self):
        return len(os.listdir(self.img_path)) #listdir: list file in folder

        