import numpy as np
import librosa
import pandas as pd
def audio2(audio,length,sr):
    newaudio = np.zeros((length*sr))
    if audio.shape[0] >= newaudio.shape[0]:
        newaudio = audio[:length*sr]
    else:
        newaudio[:audio.shape[0]] = audio
    return newaudio 
def logmel(csv,inputdir,name):
    csv = pd.read_csv(csv)
    names = csv.name.values
    labels = csv.label.values
    new_data = np.zeros((len(names),157,128))
    for i in range(len(names)):
        audio,sr = librosa.load(inputdir+names[i],sr = 16000)
        audio = audio2(audio,5,16000)
        mel = librosa.feature.melspectrogram(audio, sr=sr).T
        new_data[i,:,:] = mel
    labels = labels.reshape(-1,1)
    np.save(name+'data.npy',new_data)
    np.save(name+'label.npy',labels)
#voting   new_data = np.zeros((len(names),157,128))
traincsv = r'newtrain3.csv'
developcsv = r'newdevelop3.csv'
inputdir = r'../../cnn/folder1/cut5s_data/'
logmel(traincsv,inputdir,'train')
logmel(developcsv,inputdir,'develop')
#not voting new_data = np.zeros((len(names),313,128))
# traincsv = r'train.csv'
# developcsv = r'develop.csv'
# inputdir = r'../../cnn/folder1/final/'
# logmel(traincsv,inputdir,'train')
# logmel(developcsv,inputdir,'develop')