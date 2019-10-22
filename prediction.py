import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import os
import sys
import datetime
import h5py
import torch.multiprocessing as multiprocessing
import psutil

this_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = os.path.dirname(this_dir)
if projects_dir not in sys.path: sys.path.append(projects_dir)


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
        return self.fc4(x)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(35, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 70)
        self.fc4 = nn.Linear(70, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DatasetH5(data.Dataset):
    def __init__(self, in_file):
        self.file = h5py.File(in_file, 'r')
        self.data = self.file['data']['table']

    def __getitem__(self, index):
        data = self.data[index][1]
        input = data[:-1]
        label = data[-1:]
        return torch.from_numpy(input).float(), torch.from_numpy(label).float()

    def __len__(self):
        return self.data.shape[0]


def single_training_step(network, input, label):
    network.train()
    optimizer = optim.SGD(network.parameters(), lr=lr)
    optimizer.zero_grad()
    net_out = network(input)
    loss = criterion(net_out, label)
    loss.backward()
    optimizer.step()
    print(f'{datetime.datetime.now()} | Iteration: {batch_idx}/'
          f'{iterations} | Loss: {loss}')
    return net_out.detach()


batch_size = 1
lr = 0.1
length_dataset = 1052636
iterations = length_dataset // batch_size

dataset = DatasetH5('data/train_data/280312011.h5')
dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size)

model = Net()
model2 = Net2()
criterion = nn.BCEWithLogitsLoss()

for batch_idx, (input, label) in enumerate(dataloader):
    args_list = [[model, input, label], [model2, input, label]]
    with multiprocessing.Pool(processes=psutil.cpu_count()-1) as pool:
        results = []
        for result in pool.starmap(single_training_step, args_list):
            # pred = single_training_step(model, input, label)
            results.append(result)
    print(results)
