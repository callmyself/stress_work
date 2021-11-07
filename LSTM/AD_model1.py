import torch
import torch.nn as nn
import numpy as np
import os
class BiLSTM2(torch.nn.Module):
    def __init__(self,input_size,hidden_size):#,batch_size):
        super(BiLSTM2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.batch_size =batch_size
        self.lstm1 =nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,bidirectional = True,batch_first=True,num_layers=2,dropout=0.5)
        #self.lstm2 =nn.LSTM(input_size=self.hidden_size*2,hidden_size=self.hidden_size,bidirectional = True,batch_first=True,dropout=0.3)
        self.relu = nn.ReLU()
        self.glu = nn.GLU()
    #def init_hidden(self,batch_size):
        #return (torch.zeros(4,batch_size, self.hidden_size,device=torch.device('cuda:0')), torch.zeros(4,batch_size, self.hidden_size,device=torch.device('cuda:0')))
    def forward(self,input):
        #self.lstm1.flatten_parameters()
        x,hidden = self.lstm1(input)
        x = self.relu(x)
        #x,hidden = self.lstm2(x)
        #x = self.relu(x)
        #x,hidden = self.lstm2(x)
        #x = self.relu(x)
        return x



class Merger(torch.nn.Module):
    def __init__(self):
        super(Merger, self).__init__()
        self.hiddensize = 300
        self.blstm1 = BiLSTM2(input_size=39,hidden_size=self.hiddensize)
        self.blstm2 = BiLSTM2(input_size=195,hidden_size=self.hiddensize)
        self.blstm3 = BiLSTM2(input_size=23, hidden_size=self.hiddensize)
        self.blstm4 = BiLSTM2(input_size=115, hidden_size=self.hiddensize)
        self.fc1 = nn.Linear(in_features=self.hiddensize*4,out_features=2)
        self.fc2 = nn.Linear(in_features=self.hiddensize,out_features=3)
        self.fc3 = nn.Linear(in_features=self.hiddensize*4,out_features=self.hiddensize)
        self.fc4 = nn.Linear(in_features=self.hiddensize*2,out_features=self.hiddensize)
        self.fc5 = nn.Linear(in_features=self.hiddensize,out_features=1)
        self.fc6 =nn.Linear(in_features=self.hiddensize*8,out_features=4)
        self.fc7 =nn.Linear(in_features=1000,out_features=4)
        self.fc8 =nn.Linear(in_features=self.hiddensize*2,out_features=self.hiddensize)
        self.fc9 =nn.Linear(in_features=self.hiddensize,out_features=3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.6)
        self.bn1 = nn.BatchNorm1d(self.hiddensize)
        self.bn2 = nn.BatchNorm1d(4)


    def forward(self,x1,x2,x3,x4,device):
        def atten_60(x):
            batch_size, time_step, hidden_step = x.shape
            #print(x.shape)
            x11 = self.fc4(x)
            x11 = self.relu(x11)
            x11 = self.fc5(x11)
            #x11 = self.softmax(x11)
            x11 =x11.cpu().data.numpy()
            num =np.argmax(x11,axis=1)
            output = np.zeros((batch_size, 1, hidden_step))
            x = x.cpu().data.numpy()
            for i in range(batch_size):
                output[i, :, :] = x[i, num[i][0], :]
            output = output.reshape(-1, hidden_step)
            output = torch.Tensor(output).to(device)
            return output
        def wholeatten(lens,x1,x2,x3,x4,x0):
            x0 = self.fc6(x0)
            #x0 = self.relu(x0)
            #x0 = self.fc7(x0)
            #x0 =self.bn2(x0)
            x0 = self.softmax(x0)
            x111 = x0[:, 0]
            x112 = x0[:, 1]
            x113 = x0[:, 2]
            x114 = x0[:, 3]
            x =torch.Tensor(np.zeros((x1.shape[0],x1.shape[1]))).to(device)
            for i in range(lens):
                x[i, :] = x1[i, :] * x111[i] + x2[i,:]*x112[i] + x3[i,:]*x113[i] + x4[i,:]*x114[i]
            return x
        def finalatten(lens,x1,x2,x0):
            x0 = self.fc1(x0)
            x0 = self.relu(x0)
            #x0 = self.fc2(x0)
            #x0 = self.softmax(x0)
            x111 = x0[:, 0]
            x112 = x0[:, 1]
            x =torch.Tensor(np.zeros((x1.shape[0],x1.shape[1]))).to(device)
            for i in range(lens):
                x[i, :] = x1[i, :] * x111[i] + x2[i,:]*x112[i]
            return x
        x1_batch_size, x1_time_step, x1_fea_len = x1.shape
        #print(x1.shape)
        x1 = self.blstm1(x1)#,self.hidden1,x1_batch_size)
        x2 = self.blstm2(x2)#,self.hidden2,x1_batch_size)
        x3 = self.blstm3(x3)
        x4 = self.blstm4(x4)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)
        x4 = self.dropout(x4)
        x111=atten_60(x1)
        x112=atten_60(x2)
        x113=atten_60(x3)
        x114=atten_60(x4)
        x5 = x1[:,0,:]
        x6 = x2[:,0,:]
        x7 = x3[:,0,:]
        x8 = x4[:,0,:]
        x1 = x1[:,-1,:]
        x2 = x2[:,-1,:]
        x3 = x3[:,-1,:]
        x4 = x4[:,-1,:]
        x1 = torch.cat([x1[:, :self.hiddensize], x5[:, self.hiddensize:]], dim=1)
        x2 = torch.cat([x2[:, :self.hiddensize], x6[:, self.hiddensize:]], dim=1)
        x3 = torch.cat([x3[:, :self.hiddensize], x7[:, self.hiddensize:]], dim=1)
        x4 = torch.cat([x4[:, :self.hiddensize], x8[:, self.hiddensize:]], dim=1)
        x11 =torch.cat([x111,x112,x113,x114],dim=1)
        x12 = torch.cat([x1, x2, x3, x4], dim=1)
        lens = x1_batch_size
        x0 = wholeatten(lens,x111,x112,x113,x114,x11)
        x2 = wholeatten(lens,x1,x2,x3,x4,x12)       
        x = torch.cat([x0,x2],dim=1)
        #x = finalatten(lens,x0,x2,x)
        x = self.fc3(x)
        x = self.relu(self.dropout(x))
        x = self.fc9(x)
        '''
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc9(x)
        '''
        #x = torch.nn.functional.log_softmax(x,dim=-1)
        return x