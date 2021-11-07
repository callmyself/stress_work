from scipy import io
import pandas as pd
import soundfile
import scipy.signal as signal
import torch
import numpy as np
import os
from fairseq.models.wav2vec import Wav2VecModel

sample_rate = 16000
audio_names = []

df = pd.read_csv('train.csv')
for row in df.iterrows():
    audio_names.append(row[1]['name'])

print('Total files: ', len(audio_names)*3)

cp = torch.load('./wav2vec_large.pt', map_location='cuda:0')
# cp = torch.load('/home/lab-chen.weidong/project2/model/vq-wav2vec/vq-wav2vec.pt', map_location='cpu')

wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
wav2vec.load_state_dict(cp['model'],False)
wav2vec.eval()

save_dir = './5s_wav2vec1/'
names = os.listdir('../../cnn/folder1/cut5s_data2')
for name in names:
    wavs,fs = soundfile.read('../../cnn/folder1/cut5s_data2/'+name)
    if fs != sample_rate:
        result = int((wavs.shape[0]) / fs*sample_rate)
        wavs = signal.resample(wavs,result)
    if wavs.ndim>1 :
        wavs = np.mean(wavs,axis=1)
    wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)
    z = wav2vec.feature_extractor(wavs)
    feature_wav = wav2vec.feature_aggregator(z)
    feature_wav = feature_wav.transpose(1,2).squeeze().detach().numpy()
    print(feature_wav.shape)
    np.save(save_dir+name[:-4]+'.npy',feature_wav)
# for name in audio_names:
#     mat_name = name
#     audio_file = name
#     for i in range(3):
#         wavs, fs = soundfile.read('../../cnn/folder1/cut5s_data2/'+audio_file[:-4]+'_%02d'%i+'.wav')
#         if fs != sample_rate:
#             result = int((wavs.shape[0]) / fs * sample_rate)
#             wavs = signal.resample(wavs, result)
        
#         if wavs.ndim > 1:
#             wavs = np.mean(wavs, axis=1)

#         wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)    # (B, S)
        
#         # feature_wav = wav2vec.feature_extractor(wavs)
#         z = wav2vec.feature_extractor(wavs)
#         feature_wav = wav2vec.feature_aggregator(z)
#         feature_wav = feature_wav.transpose(1,2).squeeze().detach().numpy()   # (t, 512)
#         # save = {'wav': feature_wav}
#         # io.savemat(os.path.join(save_dir, mat_name), save)
        
#         print(mat_name, feature_wav.shape)
#         np.save(save_dir+mat_name[:-4]+'_%02d'%i+'.npy',feature_wav)