# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 04:23:05 2022

@author: r00526841
"""




import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import *



def mse_loss(input, target):
    return torch.mean((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def FairLoss(y_true, y_pred):
    x = y_pred - y_true
    c = 100000.0
    den = np.abs(x) + c
    grad = c * x / den
    hess = c * c / den ** 2
    return grad, hess


def PseudHuberLoss(y_true, y_pred):
    d = y_pred - y_true

    delta = 100#50#20.0
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d/scale_sqrt
    hess = 1/scale_sqrt
    return grad, hess

def FairLossLoss_torch(y_true, y_pred):
    c = 1.0
    tmp = (y_true-y_pred).abs()/c

    loss = c*c*(tmp - torch.log(1.0+tmp))
    #pdb.set_trace()

    return  loss

def PseudHuberLoss_torch(y_true, y_pred):

    delta = 1.0
    delta2 = delta * delta

    tmp = ((y_true - y_pred)/delta) * ((y_true - y_pred)/delta)

    loss = delta2 * (torch.sqrt(1.0 + tmp) - 1.0)

    return loss

       
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_true, y_pred, weight=None):
        # print(f"y_true : {y_true}")
        # print(f"y_pred : {y_pred}")
        # print(f"weight : {weight}")

        if weight is not None:
            loss = torch.sqrt(torch.mean(weight * (y_pred - y_true)**2) + self.eps)

        else:

            loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        #print(f"loss : {loss}")
        #sys.exit()
        return loss


def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction="batchmean")
    #outputs = torch.log(inputs)
    loss = criterion(inputs, labels)
    #loss = loss.sum()/loss.shape[0]
    #pdb.set_trace()
    return loss

# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
class FocalLossWithOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target_one_hot):
        y = target_one_hot#one_hot(target, input.size(-1))

        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()

def setTorchEvalFunc(y_pred, y_true):

    #pdb.set_trace()

    return nn.L1Loss()(y_pred,  y_true).detach()