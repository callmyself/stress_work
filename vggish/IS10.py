import torch
import torch.nn as nn
class tryFC(nn.Module):
    def __init__(self):
        super(tryFC,self).__init__()
        self.fc1 = nn.Linear(1582,1582//4)
        self.fc11 = nn.Linear(1582//4,1582)
        self.fc2 = nn.Linear(1582,3)
        self.fc3 = nn.Linear(256,3)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,3)
        self.dropout1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self,x):
        #x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc11(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        # x = self.relu(x)
        # x = self.fc5(x)
        # x = self.dropout1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        '''
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc3(x)
        '''
        return x