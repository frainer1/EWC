# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 00:03:24 2022

@author: HP
"""

import torch
import torchvision
from torchvision import datasets, transforms

def get_dataloaders(dataset, batch_size=100, subset=False, num_workers=0):
    """"
    returns three dataloaders. the first two load the training set, 
    the third loads the testset.
        - 'dataset' should be one of mnist, fashion
        - 'batch_size' determines batch size for all three dataloaders
        - 'subset' determines whether a subset of 1000 trainining images is used.
            If 'subset'is true, the batch_size is adjusted to 1000 automatically.
    """

    if dataset == 'mnist':
        trainset = datasets.MNIST('../pytorch_data', train=True, download=True)
        testset = datasets.MNIST('../pytorch_data', train=False)
    if dataset == 'fashion':
        trainset = datasets.FashionMNIST('../pytorch_data', train=True, download=True)

        testset = datasets.FashionMNIST('../pytorch_data', train=False)
    if subset:
        h = 1000
        trainset, _ = torch.utils.data.random_split(trainset, [h,59000], generator=torch.Generator().manual_seed(42))
        batch_size = h
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
            pin_memory=False, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
            pin_memory=False, drop_last=False)

    return trainloader, testloader
