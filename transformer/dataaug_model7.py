import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from transformer_encoder import TransformerEncoderLayer
import transformer_encoder
import torch
import torch.nn as nn
import numpy as np
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

class modules(nn.Module):
    def __init__(self,fea_type,head,inputlen,hidden_size,object,hid):
        super(modules,self).__init__()
        self.hidden_size = hidden_size        
        self.transformer = transformer_encoder.build_transformer(d_model = fea_type,dropout=0.6,nhead=head,dim_feedforward=int(fea_type/2),num_encoder_layers=1,normalize_before=True,input_len = inputlen)
        self.gru1 = nn.GRU(input_size = fea_type, hidden_size = hidden_size, bidirectional = True, batch_first = True, num_layers=1)
        self.fc = nn.Linear(fea_type,1)
        self.fc2 = nn.Linear(hidden_size,object)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.6)
        self.atten = Self_Attention(fea_type,hid,fea_type)
        self.maxpool = nn.MaxPool1d(inputlen)
        self.tanh = nn.Tanh()
    def forward(self,x):
        batchsize,seq_len,fea_len = x.shape
        x = self.transformer(x,mask=None)
        # x = self.atten(x)
        # x = self.maxpool(x.permute([0,2,1]))
        # x = x.view(batchsize,-1)
        # x = self.fc2(x)
        # x,_ = self.gru1(x)
        x = self.dropout(x)
        x1 = self.fc(x)
        x1 = self.tanh(x1)
        x1 = x1.view(batchsize,-1)
        x1 = self.softmax(x1)
        x1 = x1.view(batchsize,-1,1)
        x1 = torch.tile(x1,(1,1,fea_len))
        x = x * x1
        x = torch.sum(x,dim=1)
        # x1_newdata = torch.zeros((batchsize,self.hidden_size)).to(device)
        # for i in range(batchsize):
            # maxdata,maxplace = torch.max(x1[i,:],-1)
            # x1_newdata[i,:] = x[i,maxplace,:]
        # x = self.fc2(x1_newdata)
        x = self.fc2(x)
        return x        

class FC(nn.Module):
    def __init__(self):
        super(FC,self).__init__()
        self.fc1 = nn.Linear(256*3,64)
        self.fc2 = nn.Linear(64,3)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(self.dropout(x))
        x = self.fc2(x)
        return x
class Transformers(nn.Module):
    def __init__(self):
        super(Transformers,self).__init__()
        self.hidden_size = 300
        self.class_audio1_1 = modules(76*5,5,49,76*5,256,64)
        self.class_audio2_1 = modules(512,8,498,512,256,64)
        self.class_text = modules(768,8,69,768,256,64)
        self.FC = FC()

    def forward(self,x_audio1,x_audio2,x_text):
        batchsize = x_audio1.shape[0]
        x1 = self.class_audio1_1(x_audio1[:,:,:])
        x2 = self.class_audio2_1(x_audio2[:,:,:])
        x3 = self.class_text(x_text)
        # x = x1 + x2 
        # x = torch.cat([x,x3],1)
        x = torch.cat([x1,x2,x3],1)
        xx = self.FC(x)
        return xx