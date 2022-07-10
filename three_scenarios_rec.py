# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:46:30 2022

@author: HP
"""

import model
import torch
import image_loader
from utils import test, permute_mnist, train_model

import matplotlib.pyplot as plt
import numpy as np

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
trainloader, testloader = image_loader.get_dataloaders('mnist',
                                          batch_size=128, 
                                          subset=False, 
                                          num_workers=0)


layer_widths = [32*32, 1000, 1000, 10]
criterion = torch.nn.CrossEntropyLoss()

m = model.FCNet(layer_widths, bias=True)
m.optimizer = torch.optim.Adam(m.parameters(), betas=[0.9, 0.999], lr=1e-4)

m.to(device)

num_tasks = 10

m.set_task_weight(1e4)
m.set_online_lambda(1)

for task in range(num_tasks):
    seed = np.random.randint(0, 1e7)
    train_model(m, trainloader, testloader, epochs=10, permute=True, seed=task, device=device, method='EWC')
    
    # train EWC model
    print("EWC")
    print("Task: ", task+1)
    train_model(m, trainloader, testloader, epochs=5, permute=True, seed=task, device=device, method='EWC')
    for _, (X, _) in enumerate(trainloader):
        X = permute_mnist(X, task)
        X = X.to(device)
        m.full_fisher_estimate(X)
        # m.mc_fisher_estimate(X)
    m.update_fisher()
    
    # testing models on all previous tasks
    accs_EWC = []
    
    for t in range(task+1):
        print("Testerror task {}, method: {}".format(t+1, m.method))
        accs_EWC.append(test(m, device, testloader, criterion, permute=True, seed=t))

print("Average test accuracy after training 10 tasks: ", np.mean(accs_EWC))