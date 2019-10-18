import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import pandas as pd
import numpy as np
import datetime


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(35, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 35)
        self.fc4 = nn.Linear(35, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class MyDataset(data.Dataset):
    def __init__(self, csv_path, chunkSize):
        self.chunksize = chunkSize
        self.reader = pd.read_csv(csv_path, sep='|', chunksize=self.chunksize,
                                  header=None, iterator=True)

    def __len__(self):
        return self.chunksize

    def __getitem__(self, index):
        data = self.reader.get_chunk(self.chunksize)
        input = data.iloc[:, 1:-1]
        input = np.array(input)
        input = input.astype('float')
        label = data.iloc[:, -1:]
        label = np.array(label)
        label = label.astype('float')
        return input, label


model = Net().double()
model.train()
batch_size = 100
lr = 0.1

custom_data_from_csv = MyDataset('/Users/recep/Documents/NumericoTech/multi-agent-incidents-prediction/data/train_data/280312011.csv', batch_size)
train_loader = data.DataLoader(dataset=custom_data_from_csv)

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.MSELoss()

lenght_dataset = 1052636
iterations = lenght_dataset // batch_size

for batch_idx in range(iterations):
    for _, (input, label) in enumerate(train_loader):
        optimizer.zero_grad()
        net_out = model(input)
        loss = criterion(net_out, label)
        loss.backward()
        optimizer.step()
    print(f'{datetime.datetime.now()} | Iteration: {batch_idx}/'
          f'{iterations} | Loss: {loss}')
