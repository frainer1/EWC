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

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

trainloader, testloader = dataloaders.get_dataloaders('mnist',
                                          batch_size=1000, 
                                          subset=False, 
                                          num_workers=0)

layer_widths = [28*28, 200, 200, 10]
criterion = torch.nn.CrossEntropyLoss()

# EWC model
m = model.FCNet(layer_widths)
m.to(device)

# SGD model for comparison
n = model.FCNet(layer_widths)
n.to(device)

def train_model(model, dataloader, epochs=5, method='SGD', device='cpu', measure_freq=100, permute=False, seed=0):
    """
    Parameters
    ----------
    model : model.FCNet
        model to be trained
    dataloader : torch.utils.data.DataLoader   
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
        print('\n\nEpoch', epoch)
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
    
"""
# not used

def create_task(dataloader):
    X_perm = None
    for _, (X, _) in enumerate(dataloader):
        if X_perm is None:
            X_perm = permute_mnist(X, 0)
        else:    
            X_perm = torch.cat((X_perm, permute_mnist(X, 0)))
    return X_perm
"""

"""

"""

num_tasks = 5
for task in range(num_tasks):
    # train EWC model
    print("EWC")
    print("Task: ", task+1)
    train_model(m, trainloader, epochs=5, permute=True, seed=task, device=device, method='EWC')
    for _, (X, _) in enumerate(trainloader):
        X = permute_mnist(X, task)
        X = X.to(device)
        m.on_task_update(X)
    m.update_fisher()
    
    # train comparison model
    print("SGD")
    print("Task: ", task+1)
    train_model(n, trainloader, epochs=5, permute=True, seed=task, device=device, method='SGD')
    
    # testing models on all previous tasks
    accs_EWC = []
    accs_SGD = []
    for t in range(task+1):
        print("Testerror task {}, method: {}".format(t+1, m.method))
        accs_EWC.append(test(m, device, testloader, criterion, permute=True, seed=t))
        
        print("Testerror task {}, method: {}".format(t+1, n.method))
        accs_SGD.append(test(n, device, testloader, criterion, permute=True, seed=t))

    # plot accuracies
    plt.title("Test accuracy after training {} tasks".format(task+1))
    plt.plot(np.arange(task+1)+1, accs_EWC, 'o', label="EWC")
    plt.plot(np.arange(task+1)+1, accs_SGD, 'x', label="SGD")
    plt.ylabel("Test accuracy in %")
    plt.xlabel("Task")
    plt.legend()
    plt.show()


"""
import matplotlib.pyplot as plt    
f, axarr = plt.subplots(1,2)
axarr[0].imshow(X_perm[0].view(28,28), cmap="gray")
axarr[1].imshow(X[0].view(28,28), cmap="gray")
np.vectorize(lambda ax:ax.axis('off'))(axarr);
"""