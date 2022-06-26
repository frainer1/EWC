# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:20:45 2022

@author: Flo
"""

import torch
import torch.nn as nn
    
class hLayer(nn.Module):
    def __init__(self):
        

class hlinear(nn.Linear):
    """
    linear layer with registered hooks for activation and gradient manipulation
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.register_hooks()
        
        self.in_features = in_features
        self.out_features = out_features   
        self.bias_used = bias
        
        self.input_act = None
        self.output_grad = None
            
        self.register_buffer('opt_weights', torch.zeros(out_features, in_features))
        self.register_buffer('opt_bias', torch.zeros(out_features, 1))
        
        if bias:
            self.in_features += 1 # one extra column to store fisher regarding biases
            
        self.register_buffer('fisher', torch.zeros(self.in_features, out_features))
        self.register_buffer('new_fisher', torch.zeros(self.in_features, out_features))
        
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
            
    def save_opt_params(self):
        self.opt_weights = self.weight.clone().detach()
        if self.bias_used:
            self.opt_bias = self.bias.clone().detach()
        
    def compute_fisher(self):
        """
        computes current estimate of the Fisher information
        
        function is called in model.hsquential.full_fisher_estimate, grads therefore correspond to d/dw -1/batchsize * sum_x(prob(y|x) * log_prob(y|x))
        """
        batchsize = int(self.input_act.shape[0] / self.labels)
        for label in range(self.labels):
            acts = self.input_act[(batchsize*label):(batchsize*(label+1))]
            grads = self.output_grad[(batchsize*label):(batchsize*(label+1))]
            acts2 = torch.mul(acts, acts)
            grads2 = torch.mul(grads, grads)
            if self.bias_used:
                #print("before: ", self.new_fisher.shape)
                self.new_fisher[:self.in_features-1] += acts2.t() @ grads2
                #print("between: ", self.new_fisher.shape)
                self.new_fisher[self.in_features-1] += torch.sum(grads2, dim=0)/batchsize
                #print("after: ", self.new_fisher.shape)
                
                #summed_grads = torch.sum(grads2, dim=0)
                #print("grads:", grads2.shape)
                #print("proc: ", summed_grads.shape)
            else:
                self.new_fisher += acts2.t() @ grads2
                
    
    def update_fisher(self, device):
        """
        update running average of the fisher information
        called on task update, after all batches have been processed
        """
        self.fisher = self.online_lambda*self.fisher + self.new_fisher
        #reset buffer
        self.new_fisher = torch.zeros(self.in_features, self.out_features).to(device)
        
        #print("opt bias: ", self.opt_bias.shape)
        #print("bias: ", self.bias.shape)
        
    def ewc_loss(self):
        """
        computes and returns sum_i (F_i * (theta_i - theta_i_opt)**2)
        """
        param_diff = self.weight.clone().detach() - self.opt_weights.clone()
        if self.bias_used:
            bias_diff = self.bias.clone().detach().reshape(-1, 1) - self.opt_bias.clone()
            #print("bias:", bias_diff.shape)
            #print("before: ", param_diff.shape)
            param_diff = torch.cat((param_diff, bias_diff), dim=1)
            #print("after: ", param_diff.shape)
            
        param_diff_2 = torch.mul(param_diff, param_diff)
        loss_M = torch.mul(self.fisher.t(), param_diff_2)
        return torch.sum(loss_M, dim = [0, 1])
    
    def sgd_update(self):
        """
        vanilla SGD update: theta_new = theta_old - lambda*dL/dtheta
        """
        
        self.weight.requires_grad_(False)
        #self.update = torch.outer(self.output_grad[-1].view(-1), self.input_act[-1].view(-1))
        self.update = self.output_grad.t() @ self.input_act
        self.weight -= self.lr*self.update
        self.weight.requires_grad_(True)
        
class hConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
            dilation=1, groups=1, bias=True, padding_mode='zeros', nonlinearity='relu'):
        self.nonlinearity = nonlinearity
        super(hConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        
    

    

def test(model, device, testloader, criterion, permute=False, seed=0):
    test_loss = 0
    correct = 0
    num_datapoints = 0
    for t, (X, y) in enumerate(testloader):
        num_datapoints += y.shape[0]
        if permute:
            X = permute_mnist(X, seed)
        X, y = X.to(device), y.to(device)
        model.set_hooks(False)
        output = model(X)
        model.set_hooks(True)
        test_loss += criterion(output, y).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= t+1
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, num_datapoints,
        100. * correct / num_datapoints))
    return 100. * correct / num_datapoints
            
def permute_mnist(data, seed):
    torch.manual_seed(seed)
    h = w = 28
    indx = torch.randperm(h*w)
    return torch.index_select(data, 1, indx)
    