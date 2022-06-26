# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 00:00:57 2022

@author: HP
"""

import model
import torch
import image_loader
from utils import test, permute_mnist

import matplotlib.pyplot as plt
import numpy as np

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

trainloader, testloader = image_loader.get_dataloaders('mnist',
                                          batch_size=100, 
                                          subset=False, 
                                          num_workers=0)
def train_model(model, train_loader, test_loader, epochs=5, method='SGD', device='cpu', measure_freq=100, permute=False, seed=0):
    """
    Parameters
    ----------
    model : model.FCNet
        model to be trained
    train_loader : torch.utils.data.DataLoader   
    test_loader : torch.utils.data.DataLoader   
    epochs : int, optional
        number of epochs the model is trained. The default is 5.
    method : string, optional
        one of ["SGD", "mySGD", "EWC"]. Determines method used for parameter update. The default is 'SGD'.
    device : string, optional
        either 'cpu' or 'cuda'. The default is 'cpu'.
    measure_freq : int, optional
        determines how often current training loss and test accuracy are displayed, i.e. after every measure_freq number of mini-batches. The default is 100.
    permute : boolean, optional
        if 'True' the model is trained on permuted-mnist dataset. The default is False.
    seed : int, optional
        Only relevant if 'permute' is 'True'. Seed used for permutation of the pixels. The default is 0.

    Returns
    -------
    None.

    """
    
    model.set_method(method)
    
    for epoch in range(epochs):
        print('\n\nEpoch', epoch+1)
        for t, (X, y) in enumerate(trainloader):
            if permute:
                X = permute_mnist(X, seed)
            X = X.to(device)
            y = y.to(device)
            
            if t%measure_freq == 0:
                model.set_hooks(False)
                loss = criterion(model(X), y)
                model.set_hooks(True)
                
                do_print = 1
                if do_print:
                    print('train loss', loss.item(), '   (mini-batch estimate)')
            
            # my implementation
            model.parameter_update(X, y)
                            
        test(model, device, testloader, criterion, permute, seed)


in_channel = 1
out_channels = [10, 20]
kernel_sizes = [5, 5]
poolings = [True, True]
layer_widths = [320, 50, 50, 10]
criterion = torch.nn.CrossEntropyLoss()

m = model.CNN(in_channel, out_channels, kernel_sizes, layer_widths, poolings=poolings)
m.to(device)

num_tasks = 1
for task in range(num_tasks):
    train_model(m, trainloader, testloader, device=device)
    

