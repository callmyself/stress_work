import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.sigmoid = nn.Sigmoid()
        self.softmax= nn.Softmax()
    def forward(self, inputs, targets):
        newtarget =torch.Tensor(np.zeros((targets.shape[0],2))).to('cuda:0')
        for i in range(targets.shape[0]):
            if targets[i]==0:
                newtarget[i,:]=torch.Tensor([1,0])
            else:
                newtarget[i,:]=torch.Tensor([0,1])
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, newtarget, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(self.sigmoid(inputs), newtarget, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class FocalLosss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, predict, target):
        pt = self.softmax(predict)
        loss=0
        for i in range(len(target)):
            data = -self.alpha *(pt[i,1-target[i]])**self.gamma*torch.log(pt[i,target[i]])
            loss =loss +data
        loss = loss/len(target)
        return loss