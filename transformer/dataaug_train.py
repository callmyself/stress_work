import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn as nn
from dataaug_model import Transformers
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
from torch.optim import lr_scheduler
from torch.backends import cudnn
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from pytorch_transformers import BertModel,BertConfig,BertTokenizer
from model import TextNet
from FocalLoss1 import FocalLosss
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark= False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

model = Transformers()
model = nn.DataParallel(model).to(device)

textnet = TextNet(code_length=128)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
class stress_Dataset(Dataset):
    def __init__(self,audio_data1,audio_data2,text_data,label):
        self.audio_data1 = np.load(audio_data1)
        self.audio_data2 = np.load(audio_data2)
        self.text_data = np.load(text_data)
        self.label = np.load(label)
    def __getitem__(self,item):
        return torch.Tensor(self.audio_data1[item,:,:,:]),torch.Tensor(self.audio_data2[item,:,:,:]),torch.Tensor(self.text_data[item,:,:]),torch.Tensor(self.label[item,:])
    def __len__(self):
        return len(self.label)
def hsf_data(data,callen,interval):
    finallen = 500//(callen-interval) -1
    lens = data.shape[0]
    feas = data.shape[1]
    times = lens//(callen-interval)-1
    finaldata = np.zeros((times,feas*5))
    times2 = finallen //times
    remain = finallen % times
    finaldata2 = np.zeros((finallen,feas*5))
    for i in range(times):
        if i !=times-1:
            data1 = data[i*(callen-interval):i*(callen-interval)+20,:]
            maxdata = np.max(data1,axis=0)[np.newaxis,:]
            mindata = np.min(data1,axis=0)[np.newaxis,:]
            meandata = np.mean(data1,axis=0)[np.newaxis,:]
            stddata = np.std(data1,axis=0)[np.newaxis,:]
            meddata = np.median(data1,axis=0)[np.newaxis,:]
            calculatedata = np.concatenate((maxdata,mindata,meandata,stddata,meddata),axis=1)
            finaldata[i,:] = calculatedata
        else:
            data1 = data[i*(callen-interval):,:]
            maxdata = np.max(data1,axis=0)[np.newaxis,:]
            mindata = np.min(data1,axis=0)[np.newaxis,:]
            meandata = np.mean(data1,axis=0)[np.newaxis,:]
            stddata = np.std(data1,axis=0)[np.newaxis,:]
            meddata = np.median(data1,axis=0)[np.newaxis,:]
            calculatedata = np.concatenate((maxdata,mindata,meandata,stddata,meddata),axis=1)
            finaldata[i,:] = calculatedata
    for j in range(times2):
        finaldata2[j*times:(j+1)*times,:] = finaldata
    finaldata2[times2*times:,:]  = finaldata[finaldata.shape[0]-remain:,:]

    return finaldata2
def datacopy(be_fea,af_fea):
    be_len = be_fea.shape[0]
    af_len = af_fea.shape[0]
    times = af_len // be_len
    remain = af_len % be_len
    for i in range(times):
        af_fea[i*be_len:(i+1)*be_len,:] = be_fea
    af_fea[times*be_len:,:] = be_fea[be_len-remain:,:]
    return af_fea
def gen_data(csv,fea_path,feacsv,fealen):
    label = np.array(pd.read_csv(csv).iloc[:,1])
    name = np.array(pd.read_csv(csv).iloc[:,0])
    feas = np.zeros((len(name),3,49,fealen*5))
    for i in range(len(name)):
        fea_name = name[i]
        for j in range(3):
            temp = np.array(pd.read_csv(fea_path+fea_name[:-4]+'_%02d'%j+'_'+feacsv+'.csv',sep=';').iloc[:,2:])
            fea = hsf_data(temp,20,10)
            feas[i,j,:,:] = fea
    label = label.reshape((len(label),1))
    return feas,label
def gen_data2(csv,fea_path):
    label = np.array(pd.read_csv(csv).iloc[:,1])
    name = np.array(pd.read_csv(csv).iloc[:,0])
    feas = np.zeros((len(name),3,498,512))
    for i in range(len(name)):
        fea_name = name[i]
        for j in range(3):
            fea = np.load(fea_path+fea_name[:-4]+'_%02d'%j+'.npy')
            newfea = np.zeros((498,512))
            fea = datacopy(fea,newfea)
            feas[i,j,:,:] = fea
    label = label.reshape((len(label),1))
    return feas,label
def gen_data1(csv,fea_path):
    label = np.array(pd.read_csv(csv).iloc[:,1])
    name = np.array(pd.read_csv(csv).iloc[:,0])
    feas = np.zeros((len(name),995,76))
    for i in range(len(name)):
        fea = np.zeros((995,76))
        fea_name = name[i]
        temp = np.array(pd.read_csv(fea_path+fea_name[:-4]+'_avec2013.csv',sep=';').iloc[:,2:])
        fea = datacopy(temp,fea)
        feas[i,:,:] = fea
    label = label.reshape((len(label),1))
    return feas,label
def data_text(input_dir,input_csv):
    dataname = np.array(pd.read_csv(input_csv,sep=',').iloc[:,0])
    text = []
    for i in dataname:
        f = open(input_dir+i[:-4]+'.txt','rb')
        data = f.read()
        data = data.decode('gbk')
        if 'onebest' in data:
            splitdata = data.split('onebest')
            juzi = ''
            for data_seg in splitdata:
                if 'speaker' in data_seg:
                    finalplace = data_seg.find('speaker')
                    finaldata = data_seg[3:finalplace-3]
                    juzi = juzi + finaldata
            text.append(juzi)
        else:
            text.append('')
    return text
def gen_features(texts,textnet,tokenizer):
    textnet.eval()
    tokens,segments,input_masks = [],[],[]
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] *len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))
    max_len = 69
    for j in range(len(tokens)):
        padding = [0] *(max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding
    tokens_tensor = torch.tensor(tokens)
    segments_tensors = torch.tensor(segments)
    input_masks_tensors = torch.tensor(input_masks)
    text_hashCodes = textnet(tokens_tensor,segments_tensors,input_masks_tensors)
    return text_hashCodes
def label2label(label):
    newlabel =[]
    for i in range(len(label)):
        if label[i] !=0:
            newlabel.append(1)
        else:
            newlabel.append(0)
    return torch.Tensor(np.array(newlabel))
def voting(x):
    result = np.zeros((x.shape[0],x.shape[1]),dtype=int)
    finalresult = np.zeros((x.shape[0]),dtype=int)
    for i in range(x.shape[1]):
        result[:,i]=torch.max(x[:,i,:].cpu().data, 1)[1].numpy()
    for j in range(x.shape[0]):
        finalresult[j] = np.argmax(np.bincount(result[j,:]))
    return np.array(finalresult)   
# if not os.path.exists('train_audio_data.npy'):
#产生特征矩阵
'''
'''
# traincsv = r'./train.csv'
# developcsv = r'./develop.csv'
# fea_path2 = r'../../../../cnn/folder1/feature/5s_lld_avec20131/' 
# text_path = r'../../../../../data_asr/'
# fea_path = r'../../../../cnn/folder1/feature/5s_wav2vec/'
# # train_audio_data1,train_label = gen_data1(traincsv,fea_path2)
# # develop_audio_data1,develop_label = gen_data1(developcsv,fea_path2)
# train_audio_data1,train_label = gen_data(traincsv,fea_path2,'avec2013',76)
# develop_audio_data1,develop_label = gen_data(developcsv,fea_path2,'avec2013',76)
# train_audio_data2,_ =gen_data2(traincsv,fea_path)
# develop_audio_data2,_ = gen_data2(developcsv,fea_path)
# texts_train = data_text(text_path,traincsv)
# texts_develop = data_text(text_path,developcsv)
# train_text_data= gen_features(texts_train,textnet,tokenizer)
# develop_text_data = gen_features(texts_develop,textnet,tokenizer)
# print('train-audio1-shape',train_audio_data1.shape)
# print('train-audio2-shape',train_audio_data2.shape)
# print('train-text-shape',train_text_data.shape)
# print('train-label-shape',train_label.shape)
# print('develop-audio1-shape',develop_audio_data1.shape)
# print('develop-audio2-shape',develop_audio_data2.shape)
# print('develop-text-shape',develop_text_data.shape)
# print('develop-label-shape',develop_label.shape)
# train_label = train_label.reshape((-1,1))
# develop_label = develop_label.reshape((-1,1))
# np.save(r'./train_audio_data1.npy',train_audio_data1)
# np.save(r'./train_label.npy',train_label)
# np.save(r'./develop_audio_data1.npy',develop_audio_data1)
# np.save(r'./train_audio_data2.npy',train_audio_data2)
# np.save(r'./develop_audio_data2.npy',develop_audio_data2)
# np.save(r'./develop_label.npy',develop_label)
# np.save(r'./train_text_data.npy',train_text_data.detach().numpy())
# np.save(r'./develop_text_data.npy',develop_text_data.detach().numpy())
# '''
# '''
train_audio_data1= r'train_audio_data1.npy'
train_label = r'train_label.npy'
train_audio_data2 = r'train_audio_data2.npy'
develop_audio_data1 = r'develop_audio_data1.npy'
develop_label = r'develop_label.npy'
develop_audio_data2 = r'develop_audio_data2.npy'
train_text_data= r'train_text_data.npy'
develop_text_data = r'develop_text_data.npy'
train_dataset = stress_Dataset(train_audio_data1,train_audio_data2,train_text_data,train_label)
develop_dataset = stress_Dataset(develop_audio_data1,develop_audio_data2,develop_text_data,develop_label)
trainDataset = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
developDataset = DataLoader(dataset=develop_dataset,batch_size=8,shuffle=None)

WD = 1e-5
LR_DECAY = 0.3
EPOCH = 60
STEP_SIZE=10
optimizer = torch.optim.Adam(model.parameters(), lr=7e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum = 0.8,weight_decay=WD)
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)
loss = nn.CrossEntropyLoss().to(device)
# loss = FocalLosss().to(device)
for epoch in range(EPOCH):
    model.train()
    loss_tr = 0.0
    start_time = time.time()
    pred_all,actu_all = [],[]
    for step, (data_audio1,data_audio2,data_text,labels) in enumerate(trainDataset, 0):
        data_audio1,data_audio2,data_text = data_audio1.to(device),data_audio2.to(device),data_text.to(device)
        labels = labels.view(len(labels))
        # labels = label2label(labels)
        labels = labels.to(device)
        out = model(data_audio1,data_audio2,data_text)
        optimizer.zero_grad()
        err1 = loss(out[:,0,:],labels.long())
        err2 = loss(out[:,1,:],labels.long())
        err3 = loss(out[:,2,:],labels.long())
        err4 = loss(out[:,3,:],labels.long())
        err = err1 + err2 + err3 + err4
        err.backward()
        optimizer.step()
        pred = voting(out)
        actu = labels.cpu().data.numpy()
        pred_all += list(pred)
        actu_all += list(actu)
        loss_tr += err.cpu().item()
    loss_tr = loss_tr / len(trainDataset.dataset)
    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    wa_tr = metrics.accuracy_score(actu_all, pred_all)
    ua_tr = metrics.recall_score(actu_all, pred_all,average='macro')
    end_time = time.time()
    print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| wa: %.3f' % wa_tr, '| ua: %.3f' % ua_tr)  
    print('所耗时长:',str(end_time-start_time),'s')
    scheduler.step()
    #----------------------------------------------eval---------------------------------------
    model.eval()
    loss_de = 0.0
    start_time = time.time()
    pred_all,actu_all = [],[]
    for step, (data_audio1,data_audio2,data_text,labels) in enumerate(developDataset, 0):
        data_audio1,data_audio2,data_text = data_audio1.to(device),data_audio2.to(device),data_text.to(device)
        labels = labels.view(len(labels))
        # labels = label2label(labels)
        labels = labels.to(device)
        out = model(data_audio1,data_audio2,data_text)
        err1 = loss(out[:,0,:],labels.long())
        err2 = loss(out[:,1,:],labels.long())
        err3 = loss(out[:,2,:],labels.long())
        err4 = loss(out[:,3,:],labels.long())
        err = err1 + err2 + err3 + err4
        pred = voting(out)
        actu = labels.cpu().data.numpy()
        pred_all += list(pred)
        actu_all += list(actu)
        loss_de += err.cpu().item()
    loss_de = loss_de / len(developDataset.dataset)
    pred_all, actu_all = np.array(pred_all), np.array(actu_all,dtype=int)
    wa_de = metrics.accuracy_score(actu_all, pred_all)
    ua_de = metrics.recall_score(actu_all, pred_all,average='macro')
    end_time = time.time()
    print('TEST:: Epoch: ', epoch, '| Loss: %.3f' % loss_de, '| wa: %.3f' % wa_de, '| ua: %.3f' % ua_de)  
    print('所耗时长:  ',str(end_time-start_time),'s')    
    print('actu_all:',actu_all)
    print('pred_all:',pred_all)