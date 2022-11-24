# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings

def create_loader(opt,kwargs):
    print('--load mnist dataset--')    
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,),std=(0.5,))
    
    ])
    anomaly_list = list(np.arange(0,10))
    random_list = np.arange(0,10)
    np.random.shuffle(random_list)
    
    dataset1 = datasets.MNIST(root ='data-mnist', train=True, download=True,transform=data_transform)
    
    data1 = dataset1.data
    target1 = dataset1.targets
    
    
    if(opt.k1==1):
        data1_p = data1[target1==anomaly_list[random_list[0]]]
        target1_p = target1[target1==anomaly_list[random_list[0]]]
    elif(opt.k1==2):
        data1_p = data1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])]
        target1_p = target1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])]
    elif(opt.k1 == 3):
        data1_p = data1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])]
        target1_p = target1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])]
    else:
        data1_p = data1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])|(target1==anomaly_list[random_list[3]])|(target1==anomaly_list[random_list[4]])]
        target1_p = target1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])|(target1==anomaly_list[random_list[3]])|(target1==anomaly_list[random_list[4]])]
    randIdx_normal = np.arange(data1_p.shape[0])
    np.random.shuffle(randIdx_normal)
    
    if(opt.k2==1):
        data1_n = data1[target1==anomaly_list[random_list[5]]]
        target1_n = target1[target1==anomaly_list[random_list[5]]]
    elif(opt.k2==2):
        data1_n = data1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])]
        target1_n = target1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])]
    elif(opt.k2 == 3):
        data1_n = data1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])]
        target1_n = target1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])]
    else:
        data1_n = data1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])|(target1==anomaly_list[random_list[8]])|(target1==anomaly_list[random_list[9]])]
        target1_n = target1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])|(target1==anomaly_list[random_list[8]])|(target1==anomaly_list[random_list[9]])]
    
    
    randIdx = np.arange(data1_n.shape[0])
    np.random.shuffle(randIdx)
    num1 = 6000
    
    dataset1.data = torch.cat((data1_p[randIdx_normal[:num1]],data1_n[randIdx[:int(opt.gamma_p*num1//(1-opt.gamma_p))]]),dim=0)
    dataset1.targets = torch.cat((target1_p[randIdx_normal[:num1]],target1_n[randIdx[:int(opt.gamma_p*num1//(1-opt.gamma_p))]]),dim=0)
    
    
    num2 = data1_n.shape[0]
    dataset2 = datasets.MNIST(root ='data-mnist', train=True, download=True,transform=data_transform)
    data2 = data1_n[randIdx[num2-int(num1//(1/opt.gamma_c)):num2]]
    target2 = target1_n[randIdx[num2-int(num1//(1/opt.gamma_c)):num2]]
    
    
    randIdx = np.arange(data2.shape[0])
    np.random.shuffle(randIdx)
    
    dataset2.data = data2
    dataset2.targets = target2

    
    dataset4 = datasets.MNIST(root ='data-mnist', train=False, download=True,transform=data_transform)
    data4 = dataset4.data
    target4 = dataset4.targets
    if(opt.k1==1):
        data4_p = data4[target4==anomaly_list[random_list[0]]]
        target4_p = target4[target4==anomaly_list[random_list[0]]]
    elif(opt.k1==2):
        data4_p = data4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])]
        target4_p = target4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])]
    elif(opt.k1 == 3):
        data4_p = data4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])]
        target4_p = target4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])]
    else:
        data4_p = data4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])|(target4==anomaly_list[random_list[3]])|(target4==anomaly_list[random_list[4]])]
        target4_p = target4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])|(target4==anomaly_list[random_list[3]])|(target4==anomaly_list[random_list[4]])]
    randIdx_test = np.arange(data4_p.shape[0])
    dataset4.data = data4_p[randIdx_test[:5000]]
    dataset4.targets = target4_p[randIdx_test[:5000]]
    
    

    
    
    train_pos = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size//2, shuffle=True, drop_last = False,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset4,batch_size=100, shuffle=True, drop_last = True, **kwargs)
    return train_pos,train_neg,test_loader
