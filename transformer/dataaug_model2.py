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
        self.gru1 = nn.GRU(input_size = fea_type, hidden_size = hidden_size, bidirectional = True, batch_first = True, num_layers=1)
        self.fc = nn.Linear(hidden_size*2,1)
        self.fc2 = nn.Linear(hidden_size*2,object)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.6)
    def forward(self,x):
        batchsize = x.shape[0]
        x = self.transformer(x,mask=None)
        x,_ = self.gru1(x)
        x = self.dropout(x)
        x1 = self.fc(x)
        x1 = x1.view(batchsize,-1)
        x1 = self.softmax(x1)
        x1_newdata = torch.zeros((batchsize,self.hidden_size*2)).to(device)
        for i in range(batchsize):
            maxdata,maxplace = torch.max(x1[i,:],-1)
            x1_newdata[i,:] = x[i,maxplace,:]
        x = self.fc2(x1_newdata)
        return x        

class FC(nn.Module):
    def __init__(self):
        super(FC,self).__init__()
        self.fc1 = nn.Linear(256,64)
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
        self.class_audio1_1 = modules(76*5,5,49,self.hidden_size,256)
        self.class_audio1_2 = modules(76*5,5,49,self.hidden_size,256)
        self.class_audio1_3 = modules(76*5,5,49,self.hidden_size,256)
        self.class_audio2_1 = modules(512,8,498,self.hidden_size,256)
        self.class_audio2_2 = modules(512,8,498,self.hidden_size,256)
        self.class_audio2_3 = modules(512,8,498,self.hidden_size,256)
        self.class_text = modules(768,8,69,self.hidden_size,256)
        self.FC1 = FC()
        self.FC2 = FC()
        self.FC3 = FC()
        self.FC4 = FC()

    def forward(self,x_audio1,x_audio2,x_text):
        batchsize = x_audio1.shape[0]
        x1_1 = self.class_audio1_1(x_audio1[:,0,:,:])
        x1_2 = self.class_audio1_2(x_audio1[:,1,:,:])
        x1_3 = self.class_audio1_3(x_audio1[:,2,:,:])
        x2_1 = self.class_audio2_1(x_audio2[:,0,:,:])
        x2_2 = self.class_audio2_2(x_audio2[:,1,:,:])
        x2_3 = self.class_audio2_3(x_audio2[:,2,:,:]) 
        x3 = self.class_text(x_text)
        xx = torch.zeros((batchsize,4,3)).to(device)
        x0 = x1_1 + x2_1
        x1 = x1_2 + x2_2
        x2 = x1_3 + x2_3
        xx[:,0,:] = self.FC1(x0)
        xx[:,1,:] = self.FC2(x1)
        xx[:,2,:] = self.FC3(x2)
        xx[:,3,:] = self.FC4(x3)
        xx = torch.mean(xx,axis = 1)
        xx = xx.view(batchsize,-1)
        return xx