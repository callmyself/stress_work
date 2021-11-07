import torch
import torch.nn as nn
class tryFC(nn.Module):
    def __init__(self):
        super(tryFC,self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(768,64)
        self.fc2 = nn.Linear(64,3)
        self.fc3 = nn.Linear(128,3)
        self.dropout1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(69)
        self.avg = nn.AvgPool1d(69)
    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = x.permute([0,2,1])
        x = self.avg(x)
        x = x.view(-1,3)
        return x


'''
用来做测试:
conv1d = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,stride=2)
conv2d = nn.Conv1d(in_channels=64,out_channels =64,kernel_size=1,dilation=2)
conv3d = nn.Conv1d(in_channels=64,out_channels = 64,kernel_size=1)
bn1 = nn.BatchNorm1d(64)
ga_pool = nn.AdaptiveAvgPool1d(1)
avgpool1 = nn.AvgPool1d(kernel_size = 3)
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
conv4d = nn.Conv1d(in_channels = 64, out_channels = 64,kernel_size = 1)
conv5d = nn.Conv1d(in_channels = 64, out_channels = 128,kernel_size = 5)
conv6d = nn.Conv1d(in_channels =128,out_channels = 256,kernel_size=5)
x1 = conv1d(x)
x = conv2d(x1)
x = bn1(x)
x = relu(x)
x = conv3d(x)
x = x1 + x
x1 = ga_pool(x)
x1 = conv4d(x1)
x1 = sigmoid(x1)
x = x1 * x
x = conv5d(x)
x = avgpool1(x)
x = conv6d(x)
x = avgpool1(x)#(1,256,8887)
'''