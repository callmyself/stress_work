from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np 
from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from numpy import *
from pandas import DataFrame

def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance
def train(X_train,y_train,X_valid,y_valid):
    classifier = svm.SVC(kernel='rbf', probability = True, gamma = 'auto')
    classifier.fit(X_train,y_train)
    #估算Accuracy,precision,recall,f1值
    train_score = classifier.score(X_train,y_train)
    print('train Accuracy：',train_score)
    predict_validation = classifier.predict(X_valid)
    performance = evaluate(y_valid, predict_validation)
    print('validation :',performance)

def preprocess_data(datanpy,label_csv):
    dataname = np.array(pd.read_csv(label_csv,sep=',').iloc[:,0])
    datalabel = np.array(pd.read_csv(label_csv,sep=',').iloc[:,1])
    outputdata = np.zeros((len(dataname),512))
    for i in range(len(dataname)):
        name = dataname[i]
        data =np.load(datanpy+name[:-4]+'.npy')
        data = np.mean(data,axis=0)
        outputdata[i,:] =data
    return outputdata,datalabel
a = r'./newtrain4.csv'
b = r'./newdevelop4.csv'
d = r'./save_data/'
traindata,trainlabel = preprocess_data(d,a)
developdata,developlabel = preprocess_data(d,b)
# X_train = StandardScaler().fit_transform(traindata)
# X_test = StandardScaler().fit_transform(developdata)
# train(X_train,trainlabel,X_test,developlabel)
train(traindata,trainlabel,developdata,developlabel)