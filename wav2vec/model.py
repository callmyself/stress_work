import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC
from fairseq.models.wav2vec import Wav2Vec2Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class wav2vec2(nn.Module):
    def __init__(self):
        super(wav2vec2,self).__init__()
        self.cp = torch.load('./wav2vec_small.pt', map_location=device)
        self.wav2vec = Wav2Vec2Model.build_model(self.cp['args'], task=None)
        self.wav2vec.load_state_dict(self.cp['model'],False)
        self.fc = nn.Linear(in_features=512,out_features=3)
        self.avgpool = nn.AvgPool1d(249)
        self.maxpool = nn.MaxPool1d(249)
    def forward(self,x):
        batchsize = x.shape[0]
        x = self.wav2vec.feature_extractor(x)
        # x = self.avgpool(x)
        x = self.maxpool(x)
        x = x.view(batchsize,-1)
        x = self.fc(x)
        return x