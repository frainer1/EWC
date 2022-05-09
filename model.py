# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:14:50 2022

@author: Flo
"""

import torch
import torch.nn as nn
from utils import hlinear

class hsequential(nn.Sequential):
    def __init__(self, module_list, optimizer="SGD", lr=1e-3, output_dim = 10):
        super(hsequential, self).__init__(*module_list)
        self.set_optimizer(optimizer)
        self.set_lr(lr)
        self.set_output_dim = output_dim
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        for layer in self.hmodules():
            layer.optimizer = self.optimizer
    
    def set_lr(self, lr):
        self.lr = lr
        for layer in self.hmodules():
            layer.lr = self.lr
        
    def parameter_update(self, X, y):
        if self.optimizer == "SGD":
            self.parameter_update_SGD(X, y)
        if self.optimizer == "EWC":
            self.parameter_update_EWC(X, y)
    
    
    def parameter_update_SGD(self, X, y):
        X.requires_grad_(True)
        output = self(X)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, y)
        loss.backward()
        X.requires_grad_(False)
        self.zero_grad()
        
        for layer in self.hmodules():
            layer.sgd_update()
    
    def parameter_update_EWC(self, X, y):
        pass
    
    def set_hooks(self, forward_hook=None):
        if forward_hook is not None:
            self.forward_hook = forward_hook
        for layer in self.hmodules():
            layer.forward_hook = self.forward_hook
    
    def hmodules(self):
        out = []
        for mod in self.modules():
            if isinstance(mod, hlinear):
                out.append(mod)
        return out
        
def FCNet(layer_widths, bias = False, **kwargs):
    mods = []
    num_layers = len(layer_widths)
    for i in range(num_layers-2):
        mods.append(hlinear(layer_widths[i], layer_widths[i+1], bias = bias))
        mods.append(nn.ReLU())
    mods.append(hlinear(layer_widths[num_layers-2], layer_widths[-1], bias = bias))
    return hsequential(mods, **kwargs)
