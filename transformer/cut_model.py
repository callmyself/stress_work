import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from transformer_encoder import TransformerEncoderLayer
import transformer_encoder
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Transformers(nn.Module):
    def __init__(self):
        super(Transformers,self).__init__()
        self.transformer_audio1 = transformer_encoder.build_transformer(d_model=76*5, dropout=0.6, nhead=5, dim_feedforward=38*5, num_encoder_layers=1, normalize_before=True, input_len=49)
        self.transformer_audio2 = transformer_encoder.build_transformer(d_model=512, dropout=0.6, nhead=8, dim_feedforward=256, num_encoder_layers=1, normalize_before=True, input_len=498)
        self.transformer_text = transformer_encoder.build_transformer(d_model =768, dropout = 0.6, nhead =8,dim_feedforward =384 , num_encoder_layers =1,normalize_before = True,input_len =40)
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
        self.bn1 = nn.BatchNorm1d(49)
        self.bn2 = nn.BatchNorm1d(498)
        self.bn3 = nn.BatchNorm1d(40)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Softmax(dim=1)
        self.tanh  = nn.Tanh()
    def forward(self,x_audio1,x_audio2,x_text):
        batchsize = x_audio1.shape[0]
        x1 = self.transformer_audio1(x_audio1,mask = None)
        x11 = self.fc1(x1)
        x11 = x11.view(batchsize,-1)
        x11 = self.sigmoid(x11)
        x1_newdata = torch.Tensor(np.zeros((batchsize,380))).to(device)
        for i in range(batchsize):
            maxdata,maxplace = torch.max(x11[i,:],-1)
            x1_newdata[i,:] = x1[i,maxplace,:]

        x2 = self.transformer_audio2(x_audio2,mask = None)

        x22 = self.fc2(x2)
        x22 = x22.view(batchsize,-1)
        x22 = self.sigmoid(x22)
        x2_newdata = torch.Tensor(np.zeros((batchsize,512))).to(device)
        for i in range(batchsize):
            maxdata,maxplace = torch.max(x22[i,:],-1)
            x2_newdata[i,:] = x2[i,maxplace,:]

        x3 = self.transformer_text(x_text,mask = None)

        x33 = self.fc3(x3)
        x33 = x33.view(batchsize,-1)
        x33 = self.sigmoid(x33)
        x3_newdata = torch.Tensor(np.zeros((batchsize,768))).to(device)
        for i in range(batchsize):
            maxdata,maxplace = torch.max(x33[i,:],-1)
            x3_newdata[i,:] = x3[i,maxplace,:]
        x1 = self.fc4(x1_newdata)
        x2 = self.fc5(x2_newdata)
        x3 = self.fc6(x3_newdata)  
        x = x1 + x2 + x3  
        x = self.fc8(x)
        x = self.relu3(self.dropout(x))
        x = self.fc13(x)
        return x