import pandas as pd
import librosa
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
import vggish
from vggish import Vggish
from IS10 import tryFC
import random
from torch.optim import lr_scheduler
from sklearn import metrics
seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MFCC(wave,sr=16000,n_fft=1024,hop_length=512):#calculate features "MFCC"
    mfcc=librosa.feature.mfcc(wave,sr,n_mfcc=13,n_fft=n_fft,hop_length=hop_length)
    mfcc_delta=librosa.feature.delta(mfcc)
    mfcc_delta2=librosa.feature.delta(mfcc,order=2)
    mfccs=np.vstack((mfcc,mfcc_delta,mfcc_delta2))
    return mfccs
def splitwav(X,y,second,sr):#split wav into smaller wav
    X,sr=librosa.load(X,sr=sr)
    xshape=X.shape[0]
    length=second*sr
    remainder = xshape % length
    constant = math.floor(xshape/length)
    if remainder != 0:
        if remainder % 2 != 0:
            remainder += 1
            remainder1=int(remainder/2)
            Y =X[remainder1-1:xshape-remainder1,]
        else:
            remainder1=int(remainder/2)
            Y =X[remainder1:xshape-remainder1,]
    else:
        Y = X
    temp_X=[]
    temp_y=[]
    for i in range(0,constant):
        spsong=Y[int(i*length):int((i+1)*length),]
        mfccs=MFCC(spsong)
        ex=[y]
        temp_X.append(mfccs)
        temp_y.append(ex)

    return np.array(temp_X), np.array(temp_y)


class MyDataset(Dataset):
    def __init__(self,data,label):
        self.data=np.load(data)
        self.label=np.load(label)
        self.len=self.label.shape[0]
    def __getitem__(self, item):
        return torch.Tensor(self.data[item,:]), torch.Tensor(self.label[item,:])

    def __len__(self):
        return self.len

train_data = r'train_data.npy'
train_label=r'train_label.npy'
valid_data=r'develop_data.npy'
valid_label=r'develop_label.npy'
batch_size=256
STEP_SIZE = 5  # 1
LR_DECAY = 0.5  # 0.7
#load data
train_dataset=MyDataset(train_data,train_label)
datasettrain=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

valid_dataset=MyDataset(valid_data,valid_label)
datasetvalid=DataLoader(dataset=valid_dataset,batch_size=8,shuffle=None)
model = tryFC()
model = nn.DataParallel(model).to(device)
WD =1e-4
criterion= torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD)
accuracy=0
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)
       
#train
def train(epoch):
    global model
    model.train()
    running_loss=0
    correct = 0
    total = 0
    pred_all,actu_all =[],[]
    for j, data in enumerate(datasettrain):
        inputs, labels =data
        labels = labels.view(labels.shape[0])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        y_pred = model(inputs)
        loss3 = criterion(y_pred,labels.long())
        loss3.backward()
        optimizer.step()
        total += labels.size(0)
        _, predicted = torch.max(y_pred.data, dim=1)
        actu = labels.cpu().data.numpy()
        pred = torch.max(y_pred.cpu().data, 1)[1].numpy()
        pred_all = pred_all + list(pred)
        actu_all = actu_all + list(actu)
        correct += (predicted == labels).sum().item()
        running_loss+= loss3.item()
    print('第',str(epoch),'代结果')
    test_result=metrics.accuracy_score(actu_all, pred_all)
    cl_result=metrics.recall_score(actu_all, pred_all,average='macro')
    print('training accuracy:',str(test_result),' runningloss:',str(running_loss))
    print('classical accuracy:',str(cl_result))
    print('actu:',actu_all)
    print('pred:',pred_all)
    scheduler.step()

#test
def val(epoch):
    global model
    global accuracy
    correct=0
    total=0
    running_loss=0
    model.eval()
    pred_all,actu_all =[],[]
    with torch.no_grad():
        for data in datasetvalid:
            inputs,labels = data
            labels = labels.view(labels.shape[0])
            inputs, labels = inputs.to(device), labels.to(device)
            outputs= model(inputs)
            loss = criterion(outputs,labels.long())
            _,predicted = torch.max(outputs.data,dim=1)
            pred = torch.max(outputs.cpu().data, 1)[1].numpy()
            actu = labels.cpu().data.numpy()
            pred = torch.max(outputs.cpu().data, 1)[1].numpy()
            pred_all = pred_all + list(pred)
            actu_all = actu_all + list(actu)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            running_loss += loss.item()

    test_result=metrics.accuracy_score(actu_all, pred_all)
    cl_result=metrics.recall_score(actu_all, pred_all,average='macro')
    print('Accuracy on test:  ',str(test_result),' runningloss:',str(loss))
    print('classical accuracy',str(cl_result))
    print('actu:',actu_all)
    print('pred:',pred_all)

def test(valid_file,data_csv):
    global model
    model.load_state_dict(torch.load('model-best.txt'))
    right_man=0
    total=0
    correct=0
    alldata = os.listdir(valid_file)
    df = pd.read_csv(data_csv)
    for i in range(len(alldata)):
        a=df[df.file_name.isin([alldata[i]])]
        tag=a.iat[0,1]
        file = valid_file + '/' + alldata[i]
        X1, y1 = splitwav(file, tag, second=3, sr=16000)
        X1=np.stack(X1,axis=-1)
        y1=np.stack(y1,axis=-1)
        np.save('X1.npy',X1)
        np.save('y1.npy',y1)
        a1='X1.npy'
        a2='y1.npy'
        test_dataset=MyDataset(a1,a2)
        batch_size=len(y1[0])
        test_dataloader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=None)
        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                labels = labels.view(labels.shape[0]).long()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                a3=predicted.cpu()
                a3=a3.numpy()
                num=Counter(a3)
                right_man+=(tag==num.most_common(1)[0][0])
    print('words accuracy:',str(100*correct/total))
    print('man accuracy:',str(100*right_man/len(alldata)))
if __name__=='__main__':
    for epoch in range(100):
        train(epoch)
        val(epoch)

