# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 00:00:57 2022

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
                                          batch_size=200, 
                                          subset=False, 
                                          num_workers=0)


"""
split_mnist from https://github.com/ContinualAI/colab/blob/master/notebooks/permuted_and_split_mnist.ipynb
"""
def split_mnist(train_x, train_y, test_x, test_y, n_splits=5):
    """ Given the training set, split the tensors by the class label. """
    n_classes = 10
    if n_classes % n_splits != 0:
        print("n_classes should be a multiple of the number of splits!")
        raise NotImplementedError
    class_for_split = n_classes // n_splits
    mnist_train_test = [[],[]]  # train and test
    for id, data_set in enumerate([(train_x, train_y), (test_x, test_y)]):
        for i in range(n_splits):
            start = i * class_for_split
            end = (i + 1) * class_for_split
            split_idxs = np.where(np.logical_and(data_set[1] >= start, data_set[1] < end))[0]
            mnist_train_test[id].append((data_set[0][split_idxs], data_set[1][split_idxs]))
    return mnist_train_test



in_channel = 1
out_channels = [10, 20]
kernel_sizes = [3, 3]
paddings = [1, 0]
poolings = [True, True]
layer_widths = [720, 100, 100, 10]
criterion = torch.nn.CrossEntropyLoss()

num_tasks = 5
task_weights = [0, 0.4, 10]
on_lamdas = [0.7]

for tw in task_weights:
    for lamda in on_lamdas:
        # reinitialize model
        m = model.CNN(in_channel, out_channels, kernel_sizes, layer_widths, paddings = paddings, poolings=poolings, bias=True)
        m.set_task_weight(tw)
        m.set_online_lambda(lamda)
        m.set_batchsize(trainloader.batch_size)

        for task in range(num_tasks):
            print("EWC")
            print("Task ", task + 1)
            train_model(m, trainloader, testloader, epochs=3, device=device, method="EWC")
            for _, (X, _) in enumerate(trainloader):
                X = permute_mnist(X, task).reshape(-1, 1, 28, 28)
                X = X.to(device)
                #m.full_fisher_estimate(X)
                m.mc_fisher_estimate(X)
            m.update_fisher()
            
            # testing models on all previous tasks
            accs_EWC = []
            
            for t in range(task+1):
                print("Testerror task {}, method: {}".format(t+1, m.method))
                accs_EWC.append(test(m, device, testloader, criterion, permute=True, seed=t))
        
            # plot accuracies
            plt.figure()
            plt.title("Test accuracy after training {} tasks with tw {} and lambda {}".format(task+1, tw, lamda))
            plt.plot(np.arange(task+1)+1, accs_EWC, 'o', label="EWC")
            plt.ylabel("Test accuracy in %")
            plt.xlabel("Task")
            plt.legend()
            plt.savefig("cnn_mc_tw{}_l{}_tasks{}.png".format(tw, lamda, task+1))
