# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:14:50 2022

@author: Flo
"""

import torch
import torch.nn as nn
from utils import hlinear
from utils import hConv2d
import numpy as np
import torch.optim as optim
from torch.distributions.categorical import Categorical

class hsequential(nn.Sequential):
    def __init__(self, module_list, method="SGD", lr=1e-2, output_dim = 10, online_lambda = 0.9, task_weight = 0.95):
        """
        online_lamda: weight used for computing running average of Fisher, i.e. F_i = online_lamda * F_i-1 + F_i 
        """
        
        super(hsequential, self).__init__(*module_list)
        self.set_method(method)
        self.set_lr(lr)
        self.set_output_dim(output_dim)
        self.set_online_lambda(online_lambda)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.task_weight = task_weight
    
    def set_task_weight(self, task_weight):
        self.task_weight = task_weight
        
    def get_device(self):
        return next(self.parameters()).device
    
    def set_method(self, method):
        self.method = method
        for layer in self.hmodules():
            layer.method = self.method
    
    def set_online_lambda(self, lamda):
        self.online_lambda = lamda
        for layer in self.hmodules():
            layer.online_lambda = lamda
            
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
        X.requires_grad_(True)
        loss = criterion(self(X), y) + self.task_weight*self.EWC_loss()
        loss.backward()
        X.requires_grad_(False)
        optimizer.step()
    
    def EWC_loss(self):
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
        returns a list of all hooked layers in the model
        """
        
        out = []
        for mod in self.modules():
            if isinstance(mod, hlinear) or isinstance(mod, hConv2d):
                out.append(mod)
        return out
    
    def full_fisher_estimate(self, X):
        """
        computes "full" fisher as opposed to MC version
        """
        # reset update information
        for layer in self.hmodules():
            layer.input_act = None
            layer.output_grad = None
        
        # prepare fisher update
        log_soft = torch.nn.LogSoftmax(dim=1)
        for label in range(self.output_dim):
            X.requires_grad_(True)
            output = self(X)            
            probs = torch.nn.functional.softmax(output, dim=1).detach()
            log_probs = log_soft(output)
            weights = torch.sqrt(probs[:, label])
            loss = -torch.dot(weights, log_probs[:, label]) / np.sqrt(X.shape[0])
            if loss > 1e5:
                print("ffe loss: ", loss)
            loss.backward()
            X.requires_grad_(False)
            self.zero_grad()
            
        # compute current fisher estimate
        for layer in self.hmodules():
            layer.compute_fisher()
            
    def mc_fisher_estimate(self, X):
        """
        computes fisher update by drawing one label per datapoint from model distribution 
        """
        # reset update information
        for layer in self.hmodules():
            layer.input_act = None
            layer.output_grad = None
        
        criterion = nn.CrossEntropyLoss(reduction="sum")
        X.requires_grad_(True)
        output = self(X)
        labels = Categorical(logits=output).sample()
        loss = criterion(output, labels) / torch.sqrt(torch.tensor(1.0*X.shape[0]))
        loss.backward()
        X.requires_grad_(False)
        self.zero_grad()
        
        # compute current fisher estimate
        for layer in self.hmodules():
            layer.compute_fisher()
        
    def update_fisher(self):
        device = self.get_device()
        # update fisher estimate and save optimal parameters
        for layer in self.hmodules():
            layer.update_fisher(device)
            layer.save_opt_params()
                    
                
def FCNet(layer_widths, bias = False, **kwargs):
    mods = []
    num_layers = len(layer_widths)
    for i in range(num_layers-2):
        mods.append(hlinear(layer_widths[i], layer_widths[i+1], bias = bias))
        mods.append(nn.ReLU())
    mods.append(hlinear(layer_widths[num_layers-2], layer_widths[-1], bias = bias))
    return hsequential(mods, **kwargs)


def CNN(in_channel, out_channels, kernel_sizes, layer_widths, paddings = None, dilations=None, strides=None, poolings=None, bias = False, **kwargs):
    mods = []
    
    """
    specify convolutional layers
    """
    inc = None
    for i in range(len(out_channels)):
        dil = 1, 
        s = 1, 
        pad = 0
        if not dilations is None:
            dil = dilations[i]
        if not strides is None:
            s = strides[i]
        if not paddings is None:
            pad = paddings[i]
        if inc is None:
            inc = in_channel
        else:
            inc = out_channels[i-1]
        out = out_channels[i]
        kern = kernel_sizes[i]
        mods.append(hConv2d(inc, out, kern, stride=s, padding=pad, dilation=dil, bias=bias))
        mods.append(nn.ReLU())
        if not poolings is None and poolings[i]:
            mods.append(nn.MaxPool2d(2, 2))
            
    """
    specify fc-layers
    """
    mods.append(nn.Flatten())
    num_layers = len(layer_widths)
    for i in range(num_layers-2):
        mods.append(hlinear(layer_widths[i], layer_widths[i+1], bias = bias))
        mods.append(nn.ReLU())
    mods.append(hlinear(layer_widths[num_layers-2], layer_widths[-1], bias = bias))
    return hsequential(mods, **kwargs)