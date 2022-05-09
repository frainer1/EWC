# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:18:44 2022

@author: Flo
"""

import model
import torch
import numpy as np
import dataloaders
from utils import test

# for comparison
import torch.optim as optim

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
torch.manual_seed(0)
np.random.seed(0)

trainloader, testloader = dataloaders.get_dataloaders('mnist',
                                          batch_size=100, 
                                          subset=False, 
                                          num_workers=0)

layer_widths = [28*28, 400, 300, 10]
m = model.FCNet(layer_widths)
m.to(device)

measure_freq = 100

criterion = torch.nn.CrossEntropyLoss()


n = model.FCNet(layer_widths)
n.to(device)
optimizer = optim.SGD(n.parameters(), lr=0.01, momentum=0)

for epoch in range(5):
    print('\n\nEpoch', epoch)
    for t, (X, y) in enumerate(trainloader):
        X = X.to(device)
        y = y.to(device)
        
        if t%measure_freq == 0:
            m.set_hooks(False)
            loss_m = criterion(m(X), y)
            m.set_hooks(True)
            
            loss_n = criterion(n(X), y)
            do_print = 1
            if do_print:
                print('train loss m', loss_m.item(), '   (mini-batch estimate)')
                print('train loss n', loss_n.item(), '   (mini-batch estimate)')  
        
        # my implementation        
        m.parameter_update(X,y)

        # torch implementation
        optimizer.zero_grad()
        loss = criterion(n(X), y)
        loss.backward()
        optimizer.step()
    test(m, device, testloader, criterion)
    test(n, device, testloader, criterion)