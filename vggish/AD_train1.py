import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from AD_model1 import Merger
#torch.backends.cudnn.enabled=False
from torch.optim import lr_scheduler
from torch.backends import cudnn
import random
import matplotlib.pyplot as plt
from sklearn import metrics
plt.switch_backend('agg')
plt.ion()
path_pic = 'picture/'
domain = 'AD_india_base_model'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
#Dataset devide
def paixu1(array):
    length = len(array)
    num = 1
    place = 0
    while num < 114:
        dataplace =[]
        for i in range(length):
            name = array[i].split('_')
            if int(name[0]) == num:
                dataplace.append(i)

        for j in range(len(dataplace)):
            temp =array[place]
            array[place] = array[dataplace[j]]
            array[dataplace[j]] = temp
            place += 1
        num +=1
    num =1
    while num < 114:
        dataplace= []
        data = []
        kankan=[]
        tet=[]
        for i in range(length):
            name = array[i].split('_')
            if int(name[0]) ==num:
                dataplace.append(i)
                data.append(array[i])
        place = dataplace[0]
        badmemory =0
        for j in range(len(dataplace)):
            name = array[dataplace[j]].split('_')
            if name[2] =='BadMemory':
                temp = array[place]
                array[place] =array[dataplace[j]]
                kankan.append(array[dataplace[j]])
                array[dataplace[j]] = temp
                place +=1
                badmemory +=1
        videorecord = len(dataplace) - badmemory
        name = array[dataplace[0]].split('_')
        place = dataplace[0]
        for j in range(badmemory):
            array[place] =name[0]+'_'+name[1]+'_'+'BadMemory'+'_'+str(j)+'_'+name[4]
            place+=1
        for j in range(videorecord):
            array[place] = name[0]+'_'+name[1]+'_'+'VideoRecall'+'_'+str(j)+'_'+name[4]
            place +=1
        num +=1
    return array
def devide(input_file,train_csv,develop_csv,output1_name,output2_name):
    filename_train = np.array(pd.read_csv(train_csv).iloc[:,0])
    filename_develop = np.array(pd.read_csv(develop_csv).iloc[:,0])
    filetrain=[]
    filedeve=[]
    input_data_name = os.listdir(input_file)
    intput_data_name = paixu1(input_data_name)
    for i in filename_train:       
        num = 0
        name1 = i.split('_')
        for j in input_data_name:
            name2 = j.split('_')
            #print(name1)
            #print(name2)
            names = name1[2]
            if name1[0]+name1[1]+names[:-4]==name2[0]+name2[1]+name2[2] :
                file = np.load(input_file+'/'+j)
                filetrain.append(file)
                num += 1
    for k in filename_develop:
        num = 0
        name1 = k.split('_')
        for l in input_data_name:
            name2 = l.split('_')
            names = name1[2]
            if name1[0]+name1[1]+names[:-4]==name2[0] + name2[1] + name2[2]:
                file = np.load(input_file+'/'+l)
                filedeve.append(file)
                num += 1
    filetrain = np.stack(filetrain,axis=0)
    filedev = np.stack(filedeve,axis=0)
    print('filetrain.shape',filetrain.shape)
    print('filedev.shape',filedev.shape)
    np.save(output1_name,filetrain)
    np.save(output2_name,filedev)

def gettag(input_train,input_dev,output):
    wholelen = r'../features/hsf_egemaps'
    wholelen = os.listdir(wholelen)
    wholelen = paixu1(wholelen)
    train = np.array(pd.read_csv(input_train).iloc[:,1])
    train_name = np.array(pd.read_csv(input_train).iloc[:,0])
    dev = np.array(pd.read_csv(input_dev).iloc[:,1])
    dev_name = np.array(pd.read_csv(input_dev).iloc[:,0])
    train1 = []
    dev1 = []
    for i in range(len(train_name)):
        name = train_name[i]
        name1 = name.split('_')
        names1 =name1[2]
        for j in wholelen:
            name2 = j.split('_')
            if name1[0]+name1[1]+names1[:-4] == name2[0]+name2[1]+name2[2]:
                train1.append([train[i]])
    for i in range(len(dev_name)):
        name = dev_name[i]
        name1 = name.split('_')
        names1 = name1[2]
        for j in wholelen:
            name2 = j.split('_')
            if name1[0]+name1[1]+names1[:-4] == name2[0]+name2[1]+name2[2]:
                dev1.append([dev[i]])
    train1 = np.stack(train1,axis=0)
    dev1 = np.stack(dev1,axis =0)
    print(train1.shape)
    print(dev1.shape)
    np.save(output+'trainlabel.npy',train1)
    np.save(output+'devlabel.npy',dev1)


if not os.path.exists('trainlabel.npy'):
    devide(r'../features/hsf_mfcc','train.csv','develop.csv','mfcc_hsf_train.npy','mfcc_hsf_deve.npy')
    devide(r'../features/hsf_egemaps','train.csv','develop.csv','egemaps_hsf_train.npy','egemaps_hsf_deve.npy')
    devide(r'../features/lld_mfcc','train.csv','develop.csv','mfcc_lld_train.npy','mfcc_lld_deve.npy')
    devide(r'../features/lld_egemaps','train.csv','develop.csv','egemaps_lld_train.npy','egemaps_lld_deve.npy')
    gettag('train.csv','develop.csv','./')
class AD_Dataset(Dataset):
    def __init__(self,raw_mfcc,hsf_mfcc,raw_ege,hsf_ege,tag):
        self.raw_mfcc = np.load(raw_mfcc)
        self.hsf_mfcc = np.load(hsf_mfcc)
        self.raw_ege = np.load(raw_ege)
        self.hsf_ege = np.load(hsf_ege)
        self.tag = np.load(tag)
    def __getitem__(self, item):
        return torch.Tensor(self.raw_mfcc[item,:,:]),torch.Tensor(self.hsf_mfcc[item,:,:]),torch.Tensor(self.raw_ege[item,:,:]),torch.Tensor(self.hsf_ege[item,:,:]),torch.Tensor(self.tag[item,:])
    def __len__(self):
        return self.tag.shape[0]
train_raw_mfcc=r'mfcc_lld_train.npy'
train_hsf_mfcc=r'mfcc_hsf_train.npy'
train_raw_ege =r'egemaps_lld_train.npy'
train_hsf_ege =r'egemaps_hsf_train.npy'
train_tag=r'trainlabel.npy'
valid_raw_mfcc=r'mfcc_lld_deve.npy'
valid_hsf_mfcc=r'mfcc_hsf_deve.npy'
valid_raw_ege =r'egemaps_lld_deve.npy'
valid_hsf_ege =r'egemaps_hsf_deve.npy'
valid_tag=r'devlabel.npy'
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
trains = AD_Dataset(train_raw_mfcc,train_hsf_mfcc,train_raw_ege,train_hsf_ege,train_tag)
tests =AD_Dataset(valid_raw_mfcc,valid_hsf_mfcc,valid_raw_ege,valid_hsf_ege,valid_tag)

trainDataset = DataLoader(dataset=trains,batch_size=64,shuffle=True)
testDataset = DataLoader(dataset=tests,batch_size=8,shuffle=None)
model = Merger()
model = nn.DataParallel(model).to(device)
criterion= torch.nn.CrossEntropyLoss()
WD =1e-3
cudnn.benchmark = True
STEP_SIZE = 10  # 1
LR_DECAY = 0.5  # 0.7
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD)
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)
EPOCH=100
loss_train, loss_dev = [], []
wa_train, wa_dev = [], []
ua_train, ua_dev = [], []

#train&develop
for epoch in range(EPOCH):
    #train
    model.train()
    loss_tr = 0.0
    pred_all, actu_all = [], []
    start_time = time.time()
    for item,data in enumerate(trainDataset):
        inputs_raw_mfcc,inputs_hsf_mfcc,inputs_raw_ege,inputs_hsf_ege,tag =data
        tag = tag.view(tag.shape[0]).long()
        tag =tag.to(device)
        inputs_raw_mfcc,inputs_hsf_mfcc,inputs_raw_ege,inputs_hsf_ege = inputs_raw_mfcc.to(device),inputs_hsf_mfcc.to(device),inputs_raw_ege.to(device),inputs_hsf_ege.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        out = model(inputs_raw_mfcc,inputs_hsf_mfcc,inputs_raw_ege,inputs_hsf_ege,device)


        err = criterion(out, tag)
        err.backward(retain_graph=True)
        optimizer.step()
        pred = torch.max(out.cpu().data, 1)[1].numpy()
        actu = tag.cpu().data.numpy()
        pred_all = pred_all + list(pred)
        actu_all = actu_all + list(actu)

        loss_tr += err.cpu().item()
    loss_tr = loss_tr / len(trainDataset.dataset)
    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    wa_tr = metrics.accuracy_score(actu_all, pred_all)
    ua_tr = metrics.recall_score(actu_all, pred_all, average='macro')

    loss_train.append(loss_tr)
    wa_train.append(wa_tr)
    ua_train.append(ua_tr)
    print('act:',actu_all)
    print('pre:',pred_all)
    print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| wa: %.3f' % wa_tr, '| ua: %.3f' % ua_tr)
    print(metrics.classification_report(pred_all,actu_all))
    scheduler.step()
    #develop:

    model.eval()
    loss_de = 0.0
    pred_all, actu_all = [], []
    for item,data in enumerate(testDataset):
        inputs_raw_mfcc,inputs_hsf_mfcc,inputs_raw_ege,inputs_hsf_ege,tag =data
        tag = tag.view(tag.shape[0]).long()
        tag = tag.to(device)
        inputs_raw_mfcc,inputs_hsf_mfcc,inputs_hsf_ege,inputs_hsf_ege = inputs_raw_mfcc.to(device),inputs_hsf_mfcc.to(device),inputs_raw_ege.to(device),inputs_hsf_ege.to(device)
        out = model(inputs_raw_mfcc,inputs_hsf_mfcc,inputs_raw_ege,inputs_hsf_ege,device)
        err = criterion(out,tag)
        pred = torch.max(out.cpu().data, 1)[1].numpy()
        actu = tag.cpu().data.numpy()
        pred_all = pred_all + list(pred)
        actu_all = actu_all + list(actu)
        loss_de +=err.cpu().item()
    loss_de = loss_de / len(testDataset.dataset)

    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    #pred_all = pred_all.reshape(-1, len(testDataset.dataset))
    #actu_all = actu_all.reshape(-1, len(testDataset.dataset))
    #pred_all = [np.argmax(np.bincount(pred_all[i])) for i in range(len(pred_all))]
    #actu_all = [np.argmax(np.bincount(actu_all[i])) for i in range(len(actu_all))]
    print('act:',actu_all)
    print('pre:',pred_all)
    wa_de = metrics.accuracy_score(actu_all, pred_all)
    ua_de = metrics.recall_score(actu_all, pred_all, average='macro')

    loss_dev.append(loss_de)
    wa_dev.append(wa_de)
    ua_dev.append(ua_de)
    print('TEST:: Epoch: ', epoch, '| Loss: %.3f' % loss_de, '| wa: %.3f' % wa_de, '| ua: %.3f' % ua_de)
    print(metrics.classification_report(actu_all,pred_all))
    time_epoch = time.time() - start_time
    print('Epoch {:.0f} complete in {:.0f}m {:.0f}s'.format(epoch, time_epoch // 60, time_epoch % 60))
fig,axes=plt.subplots(2,3)
ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[0,2]
ax4 = axes[1,0]
ax5 = axes[1,1]
ax6 = axes[1,2]

ax1.plot(wa_train, color="blue", lw = 2.5, linestyle="-")
ax2.plot(ua_train, color="red", lw = 2.5, linestyle="-")
ax3.plot(loss_train, color="green", lw = 2.5, linestyle="-")
ax4.plot(wa_dev, color="blue", lw = 2.5, linestyle="-")
ax5.plot(ua_dev, color="red", lw = 2.5, linestyle="-")
ax6.plot(loss_dev, color="green", lw = 2.5, linestyle="-")

fig.savefig(path_pic + domain + '_' + str(seed) +'.png')


