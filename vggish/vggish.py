import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 5:
        (n_out, n_in, length, height, width) = layer.weight.size()
        n = n_in * length * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_hidden(layer):
    # Before we've done anything, we dont have any hidden state.
    # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
    return (torch.zeros(1, 1, layer.hidden_size), torch.zeros(layer.batch_size(), 1, layer.hidden_size))


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return x

class VggishConvBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        return x


class Vggish(nn.Module):
    def __init__(self, classes_num):
        super(Vggish, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock3(in_channels=128, out_channels=256)
        self.conv_block4 = VggishConvBlock3(in_channels=256, out_channels=512)
        self.conv_block5 = VggishConvBlock3(in_channels=512, out_channels=512)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc_final = nn.Linear(256, 32, bias=True)
        self.fc_final2 = nn.Linear(32,classes_num,bias=True)
        self.init_weights()
        self.dropout=nn.Dropout(p=0.3)
    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        #x = self.conv_block4(x)
        #x = self.conv_block5(x)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])#output = (batchsize,512,1,1)
        x = x.view(x.shape[0:2])#x = (batchsize,512)
        x = self.fc_final(x)
        # x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc_final2(x)
        #x = F.log_softmax(self.fc_final2(x), dim=-1)

        return x
