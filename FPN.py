#FPN 구현

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models

dir(torchvision.models)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def initialize(self): #He initialize
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant(m.weight,1)
                nn.init.constant(m.bias,0)
        

class Conv3x3block(nn.Module):
    def __init__(self,in_channels,out_channels,upsample=False):
        super().__init__()
        self.upsample=upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),stride=1,padding=1,bias=False ),
            nn.GroupNorm(32,out_channels),
            nn.ReLU(inplace=True)
            )
    def forward(self,x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True) ###
        return x

class FPNBlock(nn.Module):
    def __init__(self,pyramid_channels,skip_channels):
        super().__init__()
        #encoding output feature connection with conv1*1
        self.skip_conv = nn.Conv2d(skip_channels,pyramid_channels,kernel_size=1)
    def forward(self,x):
        x, skip = x #pyramid output과 encoding output으로 나눔

        x = F.interpolate(x,scale_factor=2,mode="nearest")
        skip = self.skip_conv(skip)

        x=x+skip
        return x
    
class SegmentationBlock(nn.Module):
    def __init__(self, in_channels,out_channels,n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3block(in_channels,out_channels,upsample=bool(n_upsamples))
        ]

        if n_upsamples >1:
            for _ in range (1,n_upsamples):
                blocks.append(Conv3x3block(out_channels,out_channels,upsample=True))
        
        self.block = nn.Sequential(*blocks)
    
    def forward(self,x):
        return self.block(x)
        
class FPNEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self,x):
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        
        x1 = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x1)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        x = x4,x3,x2,x1,x0
        return x

class FPNDecoder(Model):
    def __init__(self,
                 encoder_channels,
                 pyramid_channels=64,
                 segmentation_channels=32,
                 final_upsampling=4,
                 final_channels=1,
                 dropout=0.2,
                 merge_policy ='add'
                ):
        super().__init__()

        if merge_policy not in ['add','cat']:
            raise ValueError("merge_policy must be one of: ['add','cat']")
        self.merge_policy = merge_policy

        self.final_upsampling = final_upsampling
        self.conv1 = nn.Conv2d(encoder_channels[1],pyramid_channels,kernel_size=(1,1))
        
        self.p4 = FPNBlock(pyramid_channels,encoder_channels[2])
        self.p3 = FPNBlock(pyramid_channels,encoder_channels[3])
        self.p2 = FPNBlock(pyramid_channels,encoder_channels[4])

        self.s5 = SegmentationBlock(pyramid_channels,segmentation_channels,n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels,segmentation_channels,n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels,segmentation_channels,n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels,segmentation_channels,n_upsamples=0)

        self.dropout = nn.Dropout2d(p=dropout,inplace=True)

        if self.merge_policy == "cat":
            segmentation_channels *= 4

        self.final_conv = nn.Conv2d(segmentation_channels,final_channels,kernel_size=1,padding=0)

        self.initialize()
    
    def forward(self, x):
        c5,c4,c3,c2,_ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5,c4])
        p3 = self.p3([p4,c3])
        p2 = self.p2([p3,c2])        

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        if self.merge_policy == "add":
            x = s5+s4+s3+s2
        elif self.merge_policy == "cat":
            x = torch.cat([s5,s4,s3,s2],dim=1)

        x = self.dropout(x)
        x = self.final_conv(x)

        if self.final_upsampling is not None and self.final_upsampling>1:
            x = F.interpolate(x,scale_factor=self.final_upsampling,mode='bilinear',align_corners = True)
        return x



class FPN(nn.Module):
    def __init__(self,
                 encoder=torchvision.models.resnet50,
                 pretrained=True,
                 final_channels=1):
        super().__init__()
        self.pretrained=pretrained
        filters_dict = [2048, 2048, 1024, 512, 256]
        self.final_channels=final_channels

        self.encoder = FPNEncoder(encoder(pretrained=pretrained).to('cuda'))
        self.decoder = FPNDecoder(encoder_channels=filters_dict,final_channels=1)
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x