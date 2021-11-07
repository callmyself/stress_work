import torch.nn.functional as F
from torch import nn
from tcn1 import TemporalConvNet
import math
import torch
class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / math.sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.fc = nn.Linear(32,output_size)
        self.avgpool1d = nn.AvgPool1d(996)
        # self.atten = Self_Attention(76,32,76)
    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        batchsize = inputs.shape[0]
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        # y1 = self.avgpool1d(y1)
        # o = self.linear(y1.view(batchsize,-1))
        o = self.linear(y1[:,:,-1])
        # o = self.fc(o)
        return o
