import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),            
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.double_conv(x)
        return x

class down(nn.Module):    
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.maxpool_conv(x)
        return x
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(OutConv, self).__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 1)
        
        def forward(self, x):
            x = self.conv(x)
            return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def create_dataset():
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        os.makedirs('dataset/images')
        os.makedirs('dataset/label')

        x_train = []
        x_label = []
        for file in glob('.\\dataset\\images\\*'):
            for file_name in glob(file+'\\*'):
                img = np.array(Image.open(file_name), dtype='float32') / 255.0
                x_train.append(img)

        for file in glob('.\\dataset\\label\\*'):
            for file_name in glob(file+'\\*'):
                img = np.array(Image.open(file_name), dtype='float32') / 255.0
                x_label.append(img)

        np.random.seed(116)
        np.random.shuffle(x_train)
        np.random.seed(116)
        np.random.shuffle(x_label)

        np.save('dataset\\x_train.npy', x_train)
        np.save('dataset\\x_label.npy', x_label)
        return





