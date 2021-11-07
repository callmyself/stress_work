import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
class SENet(nn.Module):
    def __init__(self,channel_size,times):
        super(SENet,self).__init__()
        self.hid_size = hid_size
        self.times = times
        self.fc1 = nn.Linear(hid_size,hid_size//times)
        self.fc2 = nn.Linear(hid_size//times,hid_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.SpatialDropout = nn.Dropout2d(p=0.5,inplace = False)
    def forward(self,x):
        a,b,c = x.shape
        x1 = self.pool(x)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.softmax(x1)
        x1 = x1.view(a,-1,1)
        x = torch.mul(x,torch.tile(x1,(1,1,c)))
        x = x.view(a,b,c,1)
        x = self.SpatialDropout(x)
        x = x.view(a,b,c)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,self.conv2,self.chomp2,self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res),out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # for i in range(num_levels):
        #     dilation_size = 2 ** i
        #     in_channels = num_inputs if i == 0 else num_channels[i-1]
        #     out_channels = num_channels[i]
        #     layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
        #                              padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # self.network = nn.Sequential(*layers)
        self.layer1 = TemporalBlock(num_inputs,num_channels[0],kernel_size,stride=1, dilation=2,padding=(kernel_size-1) * 2, dropout=dropout)
        self.layer2 = TemporalBlock(num_inputs,num_channels[1],kernel_size,stride=1, dilation=2**2,padding=(kernel_size-1) * (2**2), dropout=dropout)
        self.layer3 = TemporalBlock(num_inputs,num_channels[2],kernel_size,stride=1, dilation=2**3,padding=(kernel_size-1) * (2**3), dropout=dropout)
        self.layer4 = TemporalBlock(num_inputs,num_channels[3],kernel_size,stride=1, dilation=2**4,padding=(kernel_size-1) * (2**4), dropout=dropout)
        self.layer5 = TemporalBlock(num_inputs,num_channels[4],kernel_size,stride=1, dilation=2**5,padding=(kernel_size-1) * (2**5), dropout=dropout)
        self.layer6 = TemporalBlock(num_inputs,num_channels[5],kernel_size,stride=1, dilation=2**6,padding=(kernel_size-1) * (2**6), dropout=dropout)
        self.layer7 = TemporalBlock(num_inputs,num_channels[6],kernel_size,stride=1, dilation=2**7,padding=(kernel_size-1) * (2**7), dropout=dropout)
    
    def forward(self, x):
        x,x1 = self.layer1(x)
        x,x2 = self.layer2(x)
        x,x3 = self.layer3(x)
        x,x4 = self.layer4(x)
        x,x5 = self.layer5(x)
        x,x6 = self.layer6(x)
        _,x7 = self.layer7(x)
        return x1 + x2 + x3 + x4 + x5 + x6 + x7
