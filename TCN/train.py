import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import time
import random
from sklearn import metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.backends import cudnn
from model import TCN
from sklearn.preprocessing import MinMaxScaler

gen_data = True
voting = True
#------------------------------------------------限制随机------------------------------------------
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark= False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False


#------------------------------------------------生成训练数据，验证数据-----------------------------
def gentraindata(inputdir,inputcsv,outputdir):
    csv = pd.read_csv(inputcsv)
    names = csv.name.values
    labels = csv.label.values
    newdata = np.zeros((len(names),80000))
    for i in range(len(names)):
        data,_ = librosa.load(inputdir+'/'+names[i],sr=16000)
        if len(data)>80000:
            data = data[:80000]
        elif len(data)<80000:
            data1 = np.zeros((80000))
            data1[:len(data)] =data
            data = data1 
        newdata[i,:] = data
    newdata = newdata.reshape(-1,1,80000)
    newlabel = np.array(labels)
    newlabel = newlabel.reshape(-1,1)
    np.save(outputdir+'traindata.npy',newdata)
    np.save(outputdir+'trainlabel.npy',newlabel)
def gendevelopdata(inputdir,inputcsv,outputdir):
    csv = pd.read_csv(inputcsv)
    names = csv.name.values
    labels = csv.label.values
    newdata = np.zeros((len(names),80000))
    for i in range(len(names)):
        data,_ = librosa.load(inputdir + '/'+names[i],sr=16000)
        if len(data)>80000:
            data = data[:80000]
        elif len(data)<80000:
            data1 = np.zeros((80000))
            data1[:len(data)] =data
            data = data1 
        newdata[i,:] = data
    newdata = newdata.reshape(-1,1,80000)
    newlabel = np.array(labels)
    newlabel = newlabel.reshape(-1,1)
    np.save(outputdir+'developdata.npy',newdata)
    np.save(outputdir+'developlabel.npy',newlabel)
def genavecdata(inputdir,inputcsv,outputdir,type,scaler):
    csv = pd.read_csv(inputcsv)
    names = csv.name.values
    labels = csv.label.values
    newdata = np.zeros((len(names),76,496))
    # newdata = np.zeros((len(names),76,996))
    for i in range(len(names)):
        name = names[i]
        data = np.array(pd.read_csv(inputdir+name[:-4]+'_avec2013.csv',sep=';').iloc[:,2:])
        scaler.fit(data)
        data = scaler.transform(data)
        newdata[i,:,:data.shape[0]] = data.T
    newlabel = np.array(labels)
    newlabel = newlabel.reshape(-1,1)
    np.save(outputdir+type+'avecdata.npy',newdata)
    np.save(outputdir+type+'aveclabel.npy',newlabel)
if gen_data ==True:
    scaler = MinMaxScaler()
    traincsv = r'newtrain.csv'
    developcsv = r'newdevelop.csv'
    inputdir = r'/148Dataset/data-chen.shuaiqi/stress/project/cnn/folder1/cut5s_data2/'
    inputdir2 = r'/148Dataset/data-chen.shuaiqi/stress/project/cnn/folder1/feature/5s_lld_avec20132/'
    inputdir3 = r'/148Dataset/data-chen.shuaiqi/stress/project/cnn/folder1/feature/lld_avec2013/'
    outputdir = r'./'
    # gentraindata(inputdir,traincsv,outputdir)
    # gendevelopdata(inputdir,developcsv,outputdir)
    genavecdata(inputdir2,traincsv,outputdir,'train',scaler)
    genavecdata(inputdir2,developcsv,outputdir,'develop',scaler)
    print('finish!')
#对验证数据做voting
def vote_item(input_csv1):
    '''
    输出相同数据的位置
    '''
    csv = pd.read_csv(input_csv1)
    names = csv.name.values
    outputitem = []
    outputitem.append(0)
    name1= names[0]
    name1 = name1[:-7]
    for j in range(len(names)):
        name2 = names[j]        
        if name1 != name2[:-7]:
            name1 = name2[:-7]
            outputitem.append(j)
    namedao1 = names[-1]
    namedao2 = names[-2]
    if namedao1[:-7] == namedao2[:-7]:
        outputitem.append(len(names)-1)
    return outputitem
def vote_data(pred,actu,outputitem):
    '''
    根据outputitem进行投票
    '''
    actu_new=[]
    pred_new=[]
    for i in range(len(outputitem)-1):
        actu_data1 = actu[outputitem[i]:outputitem[i+1]]
        pred_data1 = pred[outputitem[i]:outputitem[i+1]]
        actu_new.append(int(np.argmax(np.bincount(actu_data1))))
        pred_new.append(int(np.argmax(np.bincount(pred_data1))))
    return actu_new,pred_new
input_csv1 = r'./newdevelop.csv'
outputitem = vote_item(input_csv1)
#------------------------------------------------模型定义-------------------------------------------
# model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')
# model.fc =nn.Linear(in_features = 32,out_features=3)
# cp = torch.load('./wav2vec_small.pt', map_location=device)
# wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
# wav2vec.load_state_dict(cp['model'],False)
# wav2vec.fc = nn.Linear(in_features = 256, out_features = 3)
# wav2vec = wav2vec.to(device)
# for para in list(wav2vec.parameters())[:-1]:
#     para.requires_grad=False

model = TCN(76, 3,[76]*7, 5, 0.25).to(device)
# model = nn.DataParallel(model).to(device)#多卡
# model = wav2vec2().to(device)#单gpu
#-------------------------------------------------生成DataLoader-----------------------------------

class ADDataset(Dataset):
    def __init__(self,data,label):
        self.data = np.load(data)
        self.label = np.load(label)
    def __getitem__(self,item):
        return torch.Tensor(self.data[item,:,:]),torch.Tensor(self.label[item,:])
    def __len__(self):
        return len(self.label)


#--------------------------------------------------数据加载---------------------------------
traindata = 'trainavecdata.npy'
trainlabel = 'trainaveclabel.npy'
developdata = 'developavecdata.npy'
developlabel = 'developaveclabel.npy'
train_dataset = ADDataset(traindata,trainlabel)
develop_dataset = ADDataset(developdata,developlabel)
trainDataset = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
developDataset = DataLoader(dataset=develop_dataset,batch_size=8,shuffle=False)
#---------------------------------------------------参数设置---------------------------------

WD = 1e-4
LR_DECAY = 0.4
EPOCH = 60
STEP_SIZE=8

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=WD)
# optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=e-4, momentum=0.9, weight_decay=WD)
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY)
loss = nn.CrossEntropyLoss().to(device)

#--------------------------------------------------train--------------------------------------------
for epoch in range(EPOCH):
    model.train()
    loss_tr = 0.0
    start_time = time.time()
    pred_all,actu_all = [],[]
    for step, (datas,labels) in enumerate(trainDataset, 0):
        datas = datas.to(device)
        labels = labels.view(len(labels))
        labels = labels.to(device)
        out = model(datas)
        optimizer.zero_grad()
        err1 = loss(out,labels.long())
        err1.backward()
        optimizer.step()
        pred = torch.max(out.cpu().data, 1)[1].numpy()
        actu = labels.cpu().data.numpy()
        pred_all += list(pred)
        actu_all += list(actu)
        loss_tr += err1.cpu().item()
    loss_tr = loss_tr / len(trainDataset.dataset)
    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    wa_tr = metrics.accuracy_score(actu_all, pred_all)
    ua_tr = metrics.recall_score(actu_all, pred_all,average='macro')
    end_time = time.time()
    print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| wa: %.3f' % wa_tr, '| ua: %.3f' % ua_tr)  
    print('所耗时长:',str(end_time-start_time),'s')
    scheduler.step()
#---------------------------------------------------develop-----------------------------------------
    model.eval()
    loss_de = 0.0
    start_time = time.time()
    pred_all,actu_all = [],[]
    for step, (datas,labels) in enumerate(developDataset, 0):
        datas = datas.to(device)
        labels = labels.view(len(labels))
        labels = labels.to(device)
        out = model(datas)
        err1 = loss(out,labels.long())
        pred = torch.max(out.cpu().data, 1)[1].numpy()
        actu = labels.cpu().data.numpy()
        pred_all += list(pred)
        actu_all += list(actu)
        loss_de += err1.cpu().item()
    loss_de = loss_de / len(developDataset.dataset)
    if voting == True:
        actu_all,pred_all = vote_data(pred_all,actu_all,outputitem)
    pred_all, actu_all = np.array(pred_all,dtype=int), np.array(actu_all,dtype=int)
    wa_de = metrics.accuracy_score(actu_all, pred_all)
    ua_de = metrics.recall_score(actu_all, pred_all,average='macro')
    end_time = time.time()
    print('TEST:: Epoch: ', epoch, '| Loss: %.3f' % loss_de, '| wa: %.3f' % wa_de, '| ua: %.3f' % ua_de)  
    print('所耗时长:  ',str(end_time-start_time),'s')    
    print('actu_all:',actu_all)
    print('pred_all:',pred_all)
