# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:20:45 2022

@author: Flo
"""

import torch
import torch.nn as nn

class hlinear(nn.Linear):
    """
    linear layer with registered hooks for activation and gradient manipulation
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_hooks()
        
        self.input_act = None
        self.output_grad = None
        
    def register_hooks(self):
        self.register_full_backward_hook(self._backward_hook)
        self.register_forward_hook(self._forward_hook)
        
        self.forward_hook = True
    
    def _forward_hook(self, module, input_act, output_act):
        if(self.forward_hook):
            if self.input_act is None:
                self.input_act = input_act[0].detach()
            else:
                self.input_act = torch.cat((self.input_act, 
                                                        input_act[0].detach()))
        
    def _backward_hook(self, module, input_grad, output_grad):
        if self.output_grad is None:
            self.output_grad = output_grad[0].detach()
        else:
            self.output_grad = torch.cat((self.output_grad, 
                                   output_grad[0].detach()))
    """        
    def update(self):
        self.weight.requires_grad_(False)
        self.weight -= self.lr * self.update_direction
        self.weight.requires_grad_(True)
    """
    
    def sgd_update(self):
        self.weight.requires_grad_(False)
        #self.update = torch.outer(self.output_grad[-1].view(-1), self.input_act[-1].view(-1))
        self.update = self.output_grad.t() @ self.input_act
        self.weight -= self.lr*self.update
        self.weight.requires_grad_(True)
        
        
    
        

def test(model, device, testloader, criterion):
    test_loss = 0
    correct = 0
    num_datapoints = 0
    for t, (X, y) in enumerate(testloader):
        num_datapoints += y.size()[0]
        X, y = X.to(device), y.to(device)
        model.set_hooks(False)
        output = model(X)
        model.set_hooks(True)
        test_loss += criterion(output, y).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= num_datapoints
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, num_datapoints,
        100. * correct / num_datapoints))
    return 100. * correct / num_datapoints
            
    
