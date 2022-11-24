# -*- coding: utf-8 -*-

import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,  nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
           
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
       
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super().__init__()

  
        self.conv1 = nn.Conv2d(nc, ngf,
            4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ngf,ngf*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)


        self.conv3 = nn.Conv2d(ngf*2, ngf*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*4)

        
        

        self.conv4 = nn.Conv2d(ngf*4, 1, 4, 1, 0, bias=False)
        self.dr1 = nn.Dropout(0.3)
       
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
       
        x = F.leaky_relu(self.dr1(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        #x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = self.conv4(x)
        

        return x
