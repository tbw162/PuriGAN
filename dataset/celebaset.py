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
    print('--load celeba dataset--')
    data_transform=transforms.Compose(
            [transforms.ToPILImage(),transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]
        )

    from torch.utils.data import Dataset
    class TrainDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
            self.transform = data_transform
        def __getitem__(self, index):
           
            text = self.transform(self.data[index])
            labels = self.labels[index]
            
            return text, labels
        
        def __len__(self):
            return len(self.data)
    X = np.load('Celeba_X.npy')
    y = np.load('Celeba_y.npy')
    X = torch.from_numpy(X)
    y = torch.from_numpy(y).float()
    X = X.permute(0,3,1,2)
    X_pos = X[y==1]
    y_pos = y[y==1]
    X_neg = X[y==0]
    y_neg = y[y==0]
    randIdx = np.arange(X_neg.shape[0])
    np.random.shuffle(randIdx)
    
    num1 = int(opt.gamma_p*X_pos.shape[0]//(1-opt.gamma_p))
    train_pos_data = torch.cat((X_pos,X_neg[randIdx[:num1]]),dim=0)
    train_pos_label = torch.cat((y_pos,y_neg[randIdx[:num1]]),dim=0)
    dataset1 = TrainDataset(train_pos_data,train_pos_label)
    
    train_neg_data = X_neg[randIdx[num1:num1+X_pos.shape[0]//int(num1//(1/opt.gamma_c))]]
    train_neg_label = y_neg[randIdx[num1:num1+y_pos.shape[0]//int(num1//(1/opt.gamma_c))]]
    dataset2 = TrainDataset(train_neg_data, train_neg_label)
    
    X_gt = np.load('Celebagt_X.npy')
    y_gt = np.load('Celebagt_y.npy')
    X_gt = torch.from_numpy(X_gt)
    y_gt = torch.from_numpy(y_gt).float()
    X_gt = X_gt.permute(0,3,1,2)
    dataset4 = TrainDataset(X_gt,y_gt)
    
    
    
    
    
    train_pos = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    train_neg = torch.utils.data.DataLoader(dataset2, batch_size=opt.batch_size, shuffle=True, drop_last = False,**kwargs)
    
    test_loader = torch.utils.data.DataLoader(dataset4,batch_size=100, shuffle=True, drop_last = True, **kwargs)
    return train_pos,train_neg,test_loader
