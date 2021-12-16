import numpy as np
import librosa
import pandas as pd
import lmdb
def audio2(audio,length,sr):
    newaudio = np.zeros((length*sr))
    if audio.shape[0] >= newaudio.shape[0]:
        newaudio = audio[:length*sr]
    else:
        newaudio[:audio.shape[0]] = audio
    return newaudio 

def combineaudio(csv1,csv2,inputdir,outputdir,mode):#logmel 不做mean 效果 (313,128)
    env = lmdb.open(outputdir+mode,map_size = 30995116270)
    csv1 = pd.read_csv(csv1)
    csv2 = pd.read_csv(csv2)
    names1 = csv1.name.values
    names2 = csv2.name.values
    label = csv2.label.values
    num = 0
    with env.begin(write=True) as txn:
        for j in range(len(names1)):
            name1 = names1[j].split('_')
            for k in range(len(names2)):
                name2 = names2[k].split('_')
                if name1[0] + name1[1] == name2[0] + name2[1]:
                    audio1,sr = librosa.load(inputdir+names1[j],sr = 16000)
                    audio1 = audio2(audio1,10,16000)
                    audio11,sr = librosa.load(inputdir+names2[k],sr = 16000)
                    audio11 = audio2(audio11,10,16000)
                    mel1 = librosa.feature.melspectrogram(audio1,sr = sr)
                    mel1_delta = librosa.feature.delta(mel1).T
                    mel2 = librosa.feature.melspectrogram(audio11,sr = sr)
                    mel2_delta = librosa.feature.delta(mel2).T
                    key_fea = "data-%06d"%num
                    key_label = "label-%06d"%num
                    txn.put(key_fea.encode(), np.float32(np.ascontiguousarray(np.concatenate((mel1,mel2),0))))
                    txn.put(key_label.encode(), np.float32(label[k]))
                    num += 1
    env.close()

def combineaudio1(csv1,csv2,inputdir,outputdir,mode):#delta_logmel 不做mean 效果(313,128)
    env = lmdb.open(outputdir+mode,map_size = 30995116270)
    csv1 = pd.read_csv(csv1)
    csv2 = pd.read_csv(csv2)
    names1 = csv1.name.values
    names2 = csv2.name.values
    label = csv2.label.values
    num = 0
    with env.begin(write=True) as txn:
        for j in range(len(names1)):
            name1 = names1[j].split('_')
            for k in range(len(names2)):
                name2 = names2[k].split('_')
                if name1[0] + name1[1] == name2[0] + name2[1]:
                    audio1,sr = librosa.load(inputdir+names1[j],sr = 16000)
                    audio1 = audio2(audio1,10,16000)
                    audio11,sr = librosa.load(inputdir+names2[k],sr = 16000)
                    audio11 = audio2(audio11,10,16000)
                    mel1 = librosa.feature.melspectrogram(audio1,sr = sr)
                    mel1_delta = librosa.feature.delta(mel1).T
                    mel2 = librosa.feature.melspectrogram(audio11,sr = sr)
                    mel2_delta = librosa.feature.delta(mel2).T
                    key_fea1 = "data1-%06d"%num
                    key_fea2 = "data2-%06d"%num
                    key_label = "label-%06d"%num
                    txn.put(key_fea1.encode(), np.float32(np.ascontiguousarray(mel1_delta)))
                    txn.put(key_fea2.encode(), np.float32(np.ascontiguousarray(mel2_delta)))
                    txn.put(key_label.encode(), np.float32(label[k]))
                    num += 1
    env.close()

def combineaudio2(csv1,csv2,inputdir,outputdir,mode):#logmel 做mean(313,128)
    env = lmdb.open(outputdir+mode,map_size = 30995116270)
    csv1 = pd.read_csv(csv1)
    csv2 = pd.read_csv(csv2)
    names1 = csv1.name.values
    names2 = csv2.name.values
    label = csv2.label.values
    num = 0
    with env.begin(write=True) as txn:
        name1 = names1[0].split('_')
        mel = 0
        nums = 0
        for j in range(len(names1)):
            name11 = names1[j].split('_')
            if j != len(names1)-1:
                if name1[0] + name1[1] == name11[0] + name11[1]:
                    audio1,sr = librosa.load(inputdir+names1[j],sr=16000)
                    audio1 = audio2(audio1,10,16000)
                    mel1 = librosa.feature.melspectrogram(audio1,sr=sr).T
                    mel += mel1
                    nums +=1
                else:
                    mel = np.float32(mel/nums)
                    for k in range(len(names2)):
                        name2 = names2[k].split('_')
                        if name1[0] + name1[1] == name2[0] + name1[1]:
                            audio11,sr = librosa.load(inputdir+names2[k],sr=16000)
                            audio11 = audio2(audio11,10,16000)
                            mel2 = librosa.feature.melspectrogram(audio11,sr=sr).T
                            key_fea1 = "data1-%06d"%num
                            key_fea2 = "data2-%06d"%num
                            key_label = "label-%06d"%num
                            txn.put(key_fea1.encode(), np.float32(np.ascontiguousarray(mel)))
                            txn.put(key_fea2.encode(), np.float32(np.ascontiguousarray(mel2)))
                            txn.put(key_label.encode(), np.float32(label[k]))
                            num += 1
                    name1 = name11
                    nums = 1
                    audio1,sr = librosa.load(inputdir+names1[j],sr = 16000)
                    audio1 = audio2(audio1,10,16000)
                    mel = librosa.feature.melspectrogram(audio1,sr = sr).T
                   
            else:
                if name1[0] + name1[1] == name11[0] + name11[1]:
                    audio1,sr = librosa.load(inputdir+names1[j],sr = 16000)
                    audio1 = audio2(audio1,10,16000)
                    mel1 = librosa.feature.melspectrogram(audio1,sr=sr).T
                    mel += mel1
                    nums +=1
                    mel = np.float32(mel/nums)
                    for k in range(len(names2)):
                        name2 = names2[k].split('_')
                        if name1[0] + name1[1] == name2[0] + name1[1]:
                            audio11,sr = librosa.load(inputdir+names2[k],sr = 16000)
                            audio11 = audio2(audio11,10,16000)
                            mel2 = librosa.feature.melspectrogram(audio11,sr = sr).T
                            key_fea1 = "data1-%06d"%num
                            key_fea2 = "data2-%06d"%num
                            key_label = "label-%06d"%num
                            txn.put(key_fea1.encode(), np.float32(np.ascontiguousarray(mel)))
                            txn.put(key_fea2.encode(), np.float32(np.ascontiguousarray(mel2)))
                            txn.put(key_label.encode(), np.float32(label[k]))
                            num += 1
                else:
                    mel = np.float32(mel/nums)
                    for k in range(len(names2)):
                        name2 = names2[k].split('_')
                        if name1[0] + name1[1] == name2[0] + name1[1]:
                            audio11,sr = librosa.load(inputdir+names2[k],sr = 16000)
                            audio11 = audio2(audio11,10,16000)
                            mel2 = librosa.feature.melspectrogram(audio11,sr = sr).T
                            key_fea1 = "data1-%06d"%num
                            key_fea2 = "data2-%06d"%num
                            key_label = "label-%06d"%num
                            txn.put(key_fea1.encode(), np.float32(np.ascontiguousarray(mel)))
                            txn.put(key_fea2.encode(), np.float32(np.ascontiguousarray(mel2)))
                            txn.put(key_label.encode(), np.float32(label[k]))
                            num += 1
                    audio1,sr = librosa.load(inputdir+names1[j],sr = 16000)
                    audio1 = audio2(audio1,10,16000)
                    mel = librosa.feature.melspectrogram(audio1,sr = sr).T
                    name1 = name11
                    for k in range(len(names2)):
                        name2 = names2[k].split('_')
                        if name1[0] + name1[1] == name2[0] + name1[1]:
                            audio11,sr = librosa.load(inputdir+names2[k],sr = 16000)
                            audio11 = audio2(audio11,10,16000)
                            mel2 = librosa.feature.melspectrogram(audio11,sr = sr).T
                            key_fea1 = "data1-%06d"%num
                            key_fea2 = "data2-%06d"%num
                            key_label = "label-%06d"%num
                            txn.put(key_fea1.encode(), np.float32(np.ascontiguousarray(mel)))
                            txn.put(key_fea2.encode(), np.float32(np.ascontiguousarray(mel2)))
                            txn.put(key_label.encode(), np.float32(label[k]))
                            num += 1                    

    env.close()


def logmel(csv,inputdir,name):
    csv = pd.read_csv(csv)
    names = csv.name.values
    labels = csv.label.values
    new_data1 = np.zeros((len(names),313,128))
    new_data2 = np.zeros((len(names),313,128))
    for i in range(len(names)):
        audio,sr = librosa.load(inputdir+names[i],sr = 16000)
        audio = audio2(audio,10,16000)
        mel = librosa.feature.melspectrogram(audio, sr=sr)
        mel_delta = librosa.feature.delta(mel).T
        mel_delta2=librosa.feature.delta(mel,order=2).T
        new_data1[i,:,:] = mel_delta
        new_data2[i,:,:] = mel_delta2
    labels = labels.reshape(-1,1)
    np.save(name+'data1.npy',new_data1)
    np.save(name+'data2.npy',new_data2)
    np.save(name+'label.npy',labels)
def feature(csv,inputdir,name):
    jitterfile = r'../../../project/cnn/folder1/feature/features/'
    csv = pd.read_csv(csv)
    names = csv.name.values
    labels = csv.label.values
    newdata = np.zeros((len(names),313,40))
    for i in range(len(names)):
        name1 = names[i]
        audio,sr = librosa.load(inputdir+names[i],sr = 16000)
        audio = audio2(audio,10,16000)
        mfcc = librosa.feature.mfcc(audio,sr = sr,n_mfcc = 13).T
        mfcc_delta=librosa.feature.delta(mfcc)
        mfcc_delta2=librosa.feature.delta(mfcc,order=2)
        f0, _, _ = librosa.pyin(audio,sr = sr,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
        # mfcc = np.array(pd.read_csv(jitterfile+name1[:-4]+'_mfcc.csv',sep=';').iloc[:,2:])
        # f0 = np.array(pd.read_csv(jitterfile+name1[:-4]+'_egemaps.csv',sep=';').iloc[:,12])
        # jitter = np.array(pd.read_csv(jitterfile+name1[:-4]+'_egemaps.csv',sep=';').iloc[:,13])
        # lens = min(len(mfcc),len(f0),len(jitter))
        # newdata[i,:lens,:] = np.hstack((mfcc[:lens,:],f0[:lens].reshape(-1,1),jitter[:lens].reshape(-1,1)))
        newdata[i,:,:] = np.hstack((mfcc,mfcc_delta,mfcc_delta2,f0.reshape(-1,1)))
    labels = labels.reshape(-1,1)
    np.save(name+'data.npy',newdata)
    np.save(name+'label.npy',labels)
#voting   new_data = np.zeros((len(names),157,128))
# traincsv = r'newtrain.csv'
# developcsv = r'newdevelop.csv'
# inputdir = r'../../cnn/folder1/cut5s_data2/'
# logmel(traincsv,inputdir,'train')
# logmel(developcsv,inputdir,'develop')
#not voting new_data = np.zeros((len(names),313,128))
csv1 = r'newtrain1_VideoRecall.csv'
csv2 = r'newtrain1_BadMemory.csv'
csv3 = r'newdevelop1_VideoRecall.csv'
csv4 = r'newdevelop1_BadMemory.csv'
inputdir = r'../../../project/cnn/folder1/final/'
combineaudio2(csv1,csv2,inputdir,r'./f1/','train')
combineaudio2(csv3,csv4,inputdir,r'./f1/','develop')
# feature(traincsv,inputdir,'train')
# feature(developcsv,inputdir,'develop')
# pd.read_csv('a.csv',sep=';').iloc[:,13]