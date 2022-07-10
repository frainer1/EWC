# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:18:44 2022

@author: Flo
"""

import model
import torch
import dataloaders
from utils import test, permute_mnist

import matplotlib.pyplot as plt
import numpy as np

import csv
from pathlib import Path

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

trainloader, testloader = dataloaders.get_dataloaders('mnist',
                                          batch_size=200, 
                                          subset=False, 
                                          num_workers=0)

# h = "/Users/HP/Documents/GitHub/EWC/plots/FCN/"
h = "/Users/Flo/Desktop/Studium/Info/FS22/BA/EWC/plots/FCN/"

layer_widths = [28*28, 200, 200, 10]
criterion = torch.nn.CrossEntropyLoss()

m = model.FCNet(layer_widths, bias=True)

"""
initialize networks with fixed values

init_params(m) taken from https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch 
"""
def init_params(m):
    torch.manual_seed(0)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
m.apply(init_params)

m.to(device)

def train_model(model, train_loader, test_loader, epochs=5, method='EWC', device='cpu', measure_freq=100, permute=False, seed=0):
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
        for t, (X, y) in enumerate(train_loader):
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
                            
        test(model, device, test_loader, criterion, permute, seed)

num_tasks = 10

"""
accs_naive = []
m.set_task_weight(0)
for task in range(num_tasks):
    # train comparison model
    print("EWC with tw 0")
    print("Task: ", task+1)
    train_model(m, trainloader, testloader, epochs=5, permute=True, seed=task, device=device, method='EWC')
    
    new_accs = []
    for t in range(task+1):   
        print("Testerror task {}, method: {}".format(t+1, m.method))
        new_accs.append(test(m, device, testloader, criterion, permute=True, seed=t))
    accs_naive.append(new_accs)
    
with open(h + "naive_accs.csv", 'a') as csvfile:
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)   
    csvwriter.writerow(accs_naive)
    csvfile.close()
"""

# task_weights = [1e-3, 1e-2]
task_weights = [1e-3, 1e-2, 0.1, 0.25, 0.4, 0.75, 1, 10]
on_lambdas = task_weights[:-1]

for tw in task_weights:
    for lamda in on_lambdas:
        m.set_task_weight(tw)
        m.set_online_lambda(lamda)
        m.set_batchsize(trainloader.batch_size)
        # reinitialize model and optimizer
        m.apply(init_params)
        m.optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        
        p = "FF/Bias/tw{}_lambda{}".format(tw, lamda)
        if not Path(h+p).exists():
            Path(h+p).mkdir(parents=True)
        for task in range(num_tasks):
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
            accs_naive = []
            with open(h+"naive_accs.csv", "r") as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    l = []
                    for r in row:
                        l.append(float(r))
                    accs_naive.append(l)
                
                csvfile.close()
            
            for t in range(task+1):
                print("Testerror task {}, method: {}".format(t+1, m.method))
                accs_EWC.append(test(m, device, testloader, criterion, permute=True, seed=t))
                
                print("Testerror task {}, method: naive".format(t+1))
                print("accuracy: ", accs_naive[task][t])
        
            # plot accuracies
            plt.figure()
            plt.title("Test accuracy after training {} tasks with tw {} and lambda {}".format(task+1, tw, lamda))
            plt.plot(np.arange(task+1)+1, accs_EWC, 'o', label="EWC")
            plt.plot(np.arange(task+1)+1, accs_naive[task], 'x', label="naive")
            plt.ylabel("Test accuracy in %")
            plt.xlabel("Task")
            plt.legend()
            plt.savefig(h+p+"/tasks{}.png".format(task+1))