import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from transformer_encoder import TransformerEncoderLayer
import transformer_encoder
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def max_avg_pool(x, r=0.5):#wd's MAP
 '''
 input:
 x: (b, t, c)
 r -> 1: avg pool
 r -> 0: max pool
 return: (b, c)
 '''
 _, t, _ = x.shape
 k = int(t*r)
 s = torch.sum(x, dim=-1)
 values, indices = s.topk(k=k, dim=-1, largest=True, sorted=True)
 threshold = values[:,k-1].unsqueeze(dim=-1)
 mask = torch.where(s<threshold, torch.tensor([0.]).to(device), torch.tensor([1.]).to(device)).unsqueeze(dim=-1)
 x = torch.sum(x * mask, dim=1)/torch.sum(mask, dim=1)
 return x
class Transformers(nn.Module):
    def __init__(self):
        super(Transformers,self).__init__()
        self.transformer_audio1 = transformer_encoder.build_transformer(d_model=76*5, dropout=0.6, nhead=5, dim_feedforward=38*5, num_encoder_layers=1, normalize_before=True, input_len=99)
        self.transformer_audio2 = transformer_encoder.build_transformer(d_model=512, dropout=0.6, nhead=8, dim_feedforward=256, num_encoder_layers=1, normalize_before=True, input_len=499)
        self.transformer_text = transformer_encoder.build_transformer(d_model =768, dropout = 0.6, nhead =8,dim_feedforward =384 , num_encoder_layers =1,normalize_before = True,input_len =69)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(76*5,1)
        self.fc2 = nn.Linear(512,1)
        self.fc3 = nn.Linear(768,1)
        self.fc4 = nn.Linear(76*5,256)
        self.fc5 = nn.Linear(512,256)
        self.fc6 = nn.Linear(768,256)
        self.fc7 = nn.Linear(768,3)
        self.fc8 = nn.Linear(256,64)
        self.fc13 = nn.Linear(64,3)
        self.fc9 = nn.Linear(1,1)
        self.fc10 = nn.Linear(1,1)
        self.fc11 = nn.Linear(1,1)
        self.fc12 = nn.Linear(3,3)        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(99)
        self.bn2 = nn.BatchNorm1d(499)
        self.bn3 = nn.BatchNorm1d(69)
        self.maxpool1 = nn.MaxPool1d(99)
        self.maxpool2 = nn.MaxPool1d(499)
        self.maxpool3 = nn.MaxPool1d(69)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Softmax(dim=1)
        self.tanh  = nn.Tanh()
    def forward(self,x_audio1,x_audio2,x_text):

        '''
        将注意力机制修改为:softmax(A*A.T)*A
        输入维度(batchsize,a,b) 输出维度(batchsize,a,b)
        '''
        batchsize = x_audio1.shape[0]
        '''
        过transformer
        '''
        x1 = self.transformer_audio1(x_audio1,mask = None)
        x2 = self.transformer_audio2(x_audio2,mask = None)
        x3 = self.transformer_text(x_text,mask = None)
        #注意力机制
        x1 = torch.matmul(self.sigmoid(torch.matmul(x1,x1.permute([0,2,1]))),x1)
        x2 = torch.matmul(self.sigmoid(torch.matmul(x2,x2.permute([0,2,1]))),x2)
        x3 = torch.matmul(self.sigmoid(torch.matmul(x3,x3.permute([0,2,1]))),x3)
        x1 = x1.permute([0,2,1])
        x1 = self.maxpool1(x1)
        x1 = x1.view(batchsize,-1)
        x2 = x2.permute([0,2,1])
        x2 = self.maxpool2(x2)
        x2 = x2.view(batchsize,-1)
        x3 = x3.permute([0,2,1])
        x3 = self.maxpool3(x3)
        x3 = x3.view(batchsize,-1)
        # x1 = max_avg_pool(x1)
        # x2 = max_avg_pool(x2)
        # x3 = max_avg_pool(x3)
        x1 = self.fc4(x1)
        x2 = self.fc5(x2)
        x3 = self.fc6(x3)  
        x = x1 + x2 + x3  
        x = self.fc8(x)
        x = self.relu3(self.dropout(x))
        x = self.fc13(x)

        return x