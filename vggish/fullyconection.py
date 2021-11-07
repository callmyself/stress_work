import torch
import torch.nn as nn
class tryFC(nn.Module):
    def __init__(self):
        super(tryFC,self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(69*768,4096)
        self.fc2 = nn.Linear(4096,3)
        self.fc3 = nn.Linear(256,3)
        self.dropout1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        '''
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc3(x)
        '''
        return x