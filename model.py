# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:14:50 2022

@author: Flo
"""

import torch
import torch.nn as nn
from utils import hlinear
import numpy as np
import torch.optim as optim

class hsequential(nn.Sequential):
    def __init__(self, module_list, method="SGD", lr=1e-2, output_dim = 10, online_lamda = 0.9, task_weight = 0.95):
        """
        online_lamda: weight used for computing running average of Fisher, i.e. F_i = online_lamda * F_i-1 + F_i 
        """
        
        super(hsequential, self).__init__(*module_list)
        self.set_method(method)
        self.set_lr(lr)
        self.set_output_dim(output_dim)
        self.set_online_lamda(online_lamda)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0)
        self.task_weight = task_weight
    
    def get_device(self):
        return next(self.parameters()).device
    
    def set_method(self, method):
        self.method = method
        for layer in self.hmodules():
            layer.method = self.method
    
    def set_online_lamda(self, lamda):
        self.online_lamda = lamda
        for layer in self.hmodules():
            layer.online_lamda = lamda
            
    def set_lr(self, lr):
        self.lr = lr
        
        """
        this part is only relevant for mySGD, i.e. when no standard optimizer is used
        """
        for layer in self.hmodules():
            layer.lr = self.lr
            
    def set_output_dim(self, output_dim):
        self.output_dim = output_dim
        for layer in self.hmodules():
            layer.labels = output_dim
        
    def parameter_update(self, X, y):
        if self.method == "mySGD":
            self.parameter_update_SGD(X, y)
        if self.method == "EWC":
            self.parameter_update_EWC(self.optimizer, X, y)
        if self.method == "SGD":
            self.optim_step(self.optimizer, X, y)
        
        # reset update information:
        for layer in self.hmodules():
            layer.input_act = None
            layer.output_grad = None
    
    def optim_step(self, optimizer, X, y):
        """
        optimizer step for torch.optim.SGD
        """
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(self(X), y)
        loss.backward()
        optimizer.step()
    
    def parameter_update_SGD(self, X, y):
        #X.requires_grad_(True)
        output = self(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        loss.backward()
        #X.requires_grad_(False)
        self.zero_grad()
        
        for layer in self.hmodules():
            layer.sgd_update()
    
    def parameter_update_EWC(self, optimizer, X, y):
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(self(X), y) + self.task_weight*self.EWC_loss(X, y)
        loss.backward()
        optimizer.step()
    
    def EWC_loss(self, X, y):
        loss = 0
        for layer in self.hmodules():
            loss += layer.ewc_loss()
        return loss
    
    def set_hooks(self, forward_hook=None):
        if forward_hook is not None:
            self.forward_hook = forward_hook
        for layer in self.hmodules():
            layer.forward_hook = self.forward_hook
    
    def hmodules(self):
        """
        returns a list of all hooked linear layers in the model
        """
        
        out = []
        for mod in self.modules():
            if isinstance(mod, hlinear):
                out.append(mod)
        return out
    
    def on_task_update(self, X):
        # reset update information and save optimal parameters
        for layer in self.hmodules():
            layer.input_act = None
            layer.output_grad = None
            layer.save_opt_params()
        
        # prepare fisher update
        log_soft = torch.nn.LogSoftmax(dim=1)
        for label in range(self.output_dim):
            output = self(X)            
            probs = torch.nn.functional.softmax(output, dim=1).detach()
            log_probs = log_soft(output)
            weights = torch.sqrt(probs[:, label])
            loss = -torch.dot(weights, log_probs[:, label]) / np.sqrt(X.shape[0])
            loss.backward()
            self.zero_grad()
            
        # compute current fisher estimate
        for layer in self.hmodules():
            layer.compute_fisher()
        
    def update_fisher(self):
        device = self.get_device()
        for layer in self.hmodules():
            layer.update_fisher(device)
                    
        
        
def FCNet(layer_widths, bias = False, **kwargs):
    mods = []
    num_layers = len(layer_widths)
    for i in range(num_layers-2):
        mods.append(hlinear(layer_widths[i], layer_widths[i+1], bias = bias))
        mods.append(nn.ReLU())
    mods.append(hlinear(layer_widths[num_layers-2], layer_widths[-1], bias = bias))
    return hsequential(mods, **kwargs)
