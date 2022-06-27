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

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

trainloader, testloader = dataloaders.get_dataloaders('mnist',
                                          batch_size=100, 
                                          subset=False, 
                                          num_workers=0)

layer_widths = [28*28, 50, 50, 10]
criterion = torch.nn.CrossEntropyLoss()

# EWC model
m = model.FCNet(layer_widths, bias=False)


# SGD model for comparison
n = model.FCNet(layer_widths, bias=False)

"""
initialize both networks with the same values

init_params(m) taken from https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch 
"""
def init_params(m):
    torch.manual_seed(0)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(0.01)
        
m.apply(init_params)
n.apply(init_params)

m.to(device)
n.to(device)

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

num_tasks = 10

#task_weights = [0.1, 0.25, 0.5, 0.75, 1, 10]
task_weights = [1]
on_lamdas = [0.4]
#on_lamdas = task_weights

"""
for task in range(num_tasks):
    # train comparison model
    print("SGD")
    print("Task: ", task+1)
    train_model(n, trainloader, epochs=5, permute=True, seed=task, device=device, method='SGD')
    
    accs_SGD = []
    for t in range(task+1):   
        print("Testerror task {}, method: {}".format(t+1, n.method))
        accs_SGD.append(test(n, device, testloader, criterion, permute=True, seed=t))
    
    with open("SGD_accs.csv", 'a') as csvfile:
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)
        
        csvwriter.writerow(accs_SGD)
        csvfile.close()
"""

for tw in task_weights:
    for lamda in on_lamdas:
        m.set_task_weight(tw)
        m.set_online_lambda(lamda)
        for task in range(num_tasks):
            # train EWC model
            print("EWC")
            print("Task: ", task+1)
            train_model(m, trainloader, testloader, epochs=1, permute=True, seed=task, device=device, method='EWC')
            for _, (X, _) in enumerate(trainloader):
                X = permute_mnist(X, task)
                X = X.to(device)
                #m.full_fisher_estimate(X)
                m.mc_fisher_estimate(X)
            m.update_fisher()
            
            # testing models on all previous tasks
            accs_EWC = []
            accs_SGD = []
            with open("SGD_accs.csv", "r") as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    l = []
                    for r in row:
                        l.append(float(r))
                    accs_SGD.append(l)
                
                csvfile.close()
            
            for t in range(task+1):
                print("Testerror task {}, method: {}".format(t+1, m.method))
                accs_EWC.append(test(m, device, testloader, criterion, permute=True, seed=t))
                
                print("Testerror task {}, method: {}".format(t+1, n.method))
                print("accuracy: ", accs_SGD[task][t])
        
            # plot accuracies
            plt.figure()
            plt.title("Test accuracy after training {} tasks with tw {} and lambda {}".format(task+1, tw, lamda))
            plt.plot(np.arange(task+1)+1, accs_EWC, 'o', label="EWC")
            plt.plot(np.arange(task+1)+1, accs_SGD[task], 'x', label="SGD")
            plt.ylabel("Test accuracy in %")
            plt.xlabel("Task")
            plt.legend()
            plt.savefig("mc_tw{}_l{}_tasks{}.png".format(tw, lamda, task+1))


"""
import matplotlib.pyplot as plt    
f, axarr = plt.subplots(1,2)
axarr[0].imshow(X_perm[0].view(28,28), cmap="gray")
axarr[1].imshow(X[0].view(28,28), cmap="gray")
np.vectorize(lambda ax:ax.axis('off'))(axarr);
"""