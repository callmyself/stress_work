import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from transformer_encoder import TransformerEncoderLayer
import transformer_encoder
import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class modules(nn.Module):
    def __init__(self,fea_type,head,inputlen,hidden_size,object):
        super(modules,self).__init__()
        self.hidden_size = hidden_size        
        self.transformer = transformer_encoder.build_transformer(d_model = fea_type,dropout=0.6,nhead=head,dim_feedforward=int(fea_type/2),num_encoder_layers=1,normalize_before=True,input_len = inputlen)
        # self.gru1 = nn.GRU(input_size = fea_type, hidden_size = hidden_size, bidirectional = True, batch_first = True, num_layers=1)
        self.fc = nn.Linear(hidden_size,1)
        self.fc2 = nn.Linear(hidden_size,object)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.6)
    def forward(self,x):
        batchsize = x.shape[0]
        x = self.transformer(x,mask=None)
        # x,_ = self.gru1(x)
        # x = self.dropout(x)
        x1 = self.fc(x)
        x1 = x1.view(batchsize,-1)
        x1 = self.softmax(x1)
        x1_newdata = torch.zeros((batchsize,self.hidden_size)).to(device)
        for i in range(batchsize):
            maxdata,maxplace = torch.max(x1[i,:],-1)
            x1_newdata[i,:] = x[i,maxplace,:]
        x = self.fc2(x1_newdata)
        return x        

class FC(nn.Module):
    def __init__(self):
        super(FC,self).__init__()
        self.fc1 = nn.Linear(512,64)
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
        self.class_audio1_1 = modules(76*5,5,49,76*5,256)
        self.class_audio2_1 = modules(512,8,498,512,256)
        self.class_text = modules(768,8,69,768,256)
        self.FC = FC()

    def forward(self,x_audio1,x_audio2,x_text):
        batchsize = x_audio1.shape[0]
        x1 = self.class_audio1_1(x_audio1[:,:,:])
        x2 = self.class_audio2_1(x_audio2[:,:,:])
        x3 = self.class_text(x_text)
        x = x1 + x2 
        x = torch.cat([x,x3],1)
        xx = self.FC(x)
        return xx