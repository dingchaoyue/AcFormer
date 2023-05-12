#! python3
# -*- encoding: utf-8 -*-
"""
@file    :   cb_loss.py
@time    :   2023/02/27 14:05:30
@author  :   mnt
@version :   1.0
@contact :   ecnuzdm@gmail.com
@subject :   
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, num_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, num_of_classes].
      samples_per_cls: A python list of size [num_of_classes].
      num_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_of_classes

    labels_one_hot = F.one_hot(labels, num_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,num_of_classes)

    if loss_type == "cb_focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "cb_sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "cb_softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    
    return cb_loss

class ClassBalancedLoss(nn.Module):
      """
        A pytorch implementation of Class-Balanced Loss Based on Effective Number of Samples
        where CB denoted Class-Balanced Loss, L is an ordinary loss function, which can be 
        replaced arbitrarily
      Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, num_of_classes].
          samples_per_cls: A python list of size [num_of_classes].
          num_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
      """
      def __init__(self, loss_type, num_of_classes, samples_per_cls, beta=0.9999, gamma=2.0):
          super(ClassBalancedLoss, self).__init__()
          self.loss_type = loss_type
          self.num_of_cls = num_of_classes
          self.beta = beta
          self.gamma = gamma
          self.samples_per_cls =samples_per_cls

      def forward(self, logits, labels):          
          assert logits.shape[0] == labels.shape[0]
          effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
          weights = (1.0 - self.beta) / np.array(effective_num)
          weights = weights / np.sum(weights) * self.num_of_cls
          labels_one_hot = F.one_hot(labels, self.num_of_cls).float()

          # new tenors need put onto gpus
          weights = torch.tensor(weights).float().cuda()
          weights = weights.unsqueeze(0)
          weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
          weights = weights.sum(1)
          weights = weights.unsqueeze(1)
          weights = weights.repeat(1,self.num_of_cls)

          if self.loss_type == "cb_focal":
              cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
          elif self.loss_type == "cb_sigmoid":
              cb_loss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot, weight = weights)
          elif self.loss_type == "cb_softmax":
              pred = logits.softmax(dim = 1)
              cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)

          return cb_loss


class BalancedCrossEntropyLoss(nn.Module):
      def __init__(self, num_classes, samples_per_cls, beta=0.99):
          super(BalancedCrossEntropyLoss,self).__init__()
          self.num_classes = num_classes
          self.samples_per_cls = samples_per_cls # freqs per cls
          self.beta = beta
          # self.weights = torch.ones(num_classes)
          self.eps = 1e-6
          self.gamma = 2.0


      def forward(self, input, target):
            """
            output logits shape ::
            target shape ::
            """
            import pdb; pdb.set_trace()

            assert input.size() == target.size(), "check your input and target size"
            logits = input
            one_hot_target = torch.zeros(target.size(0), self.num_classes).cuda().scatter_(1, target.view(-1,1).long(), 1)

            # mini-batch frequency
            # freq = torch.mean(one_hot_target, dim=0)
            weights = (1 - self.beta) / (1 - torch.pow(self.beta, self.samples_per_cls + self.eps))
            self.weights = torch.tensor(weights, device = input.device)

            # calculate loss
            # softmax_cross_entropy_loss
            # loss = torch.mean(-torch.sum(self.weights*one_hot_target*torch.log_softmax(input,dim=1),dim=1))

            if self.loss_type == "s_cb_focal":
                cb_loss = focal_loss(one_hot_target, logits, self.weights, self.gamma)
            elif self.loss_type == "s_cb_sigmoid":
                # function that measures binary cross entropy between target and output logits.                  
                cb_loss = F.binary_cross_entropy_with_logits(input = logits, target = one_hot_target, weight = self.weights)
            elif self.loss_type == "s_cb_softmax":                
                pred = logits.softmax(dim = 1)
                cb_loss = F.binary_cross_entropy(input = pred, target = one_hot_target, weight = self.weights)            
            return cb_loss
      


          














if __name__ == '__main__':
    num_of_classes = 5
    logits = torch.rand(10,num_of_classes).float()
    labels = torch.randint(0,num_of_classes, size = (10,))
    beta = 0.9999
    gamma = 2.0
    samples_per_cls = [2,3,1,2,2]
    loss_type = "focal"
    cb_loss = CB_loss(labels, logits, samples_per_cls, num_of_classes, loss_type, beta, gamma)
    print(cb_loss)