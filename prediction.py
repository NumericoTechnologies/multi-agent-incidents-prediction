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
import pandas as pd
import numpy as np

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
        return torch.sigmoid(self.fc4(x))


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


def single_training_step(network, segment_ID, input, label):
    network.train()
    optimizer = optim.SGD(network.parameters(), lr=lr)
    optimizer.zero_grad()
    net_out = network(input)
    loss = criterion(net_out, label)
    loss.backward()
    optimizer.step()
    return [segment_ID, net_out.detach(), loss.detach()]


adjacency_matrix = pd.DataFrame(
    {'segments': ['278313080', '277315031', '277315032#277316021#277317004'],
     '278313080': [0, 1, 0], '277315031': [1, 0, 1],
     '277315032#277316021#277317004': [0, 1, 0]})

batch_size = 100
lr = 0.1
length_dataset = 1052636
iterations = length_dataset // batch_size

file_dir = '/Users/recep/Numerico Dropbox/Projects/2019 - NEC RWS 8 Exploration Agent Based Incident Prediction/python_out/train_data_hdf5'
segment_features = sorted([f for f in os.listdir(file_dir)])

models = dict()
dataloaders = dict()
adjacent_segments = dict()

# assign data loaders and models with respect to segments
for segment in segment_features:
    segment_ID = segment[:-3]
    adjacent_segments[segment_ID] = adjacency_matrix['segments'].where(
        adjacency_matrix[segment_ID] == 1).dropna().tolist()
    models[segment_ID] = Net()
    models[segment_ID].fc1 = nn.Linear(35 + len(adjacent_segments[segment_ID]),
                                       100)
    dataset = DatasetH5(os.path.join(file_dir, segment))
    dataloaders[segment_ID] = data.DataLoader(dataset=dataset,
                                              batch_size=batch_size)

model = Net()
criterion = nn.BCELoss()

preds = dict(zip(adjacent_segments.keys(),
                 [torch.tensor([[0.]] * batch_size)] * len(segment_features)))
for idx, data in enumerate(zip(*dataloaders.values())):
    segments_values = dict(zip(dataloaders.keys(), data))
    args_list = []
    for segment in segments_values:
        # get predictions of segment neighbours of previous timestamp(s)
        # 1 Neighbour (BS=2): tensor([[T1., T2.]]) —> tensor([[T1.],[T2.]])
        # 2 Neighbours (BS=2): tensor([[T1., T2.], [T1., T2.]])
        #                   —> tensor([[T1., T1.], [T2., T2.]])
        # 3 Neighbours (BS=4): tensor([[T1., T2., T3., T4.],
        #                      [T1., T2., T3., T4.], [T1., T2., T3., T4.]])
        #                   —> tensor([[T1., T1., T1.], [T2., T2., T2.],
        #                      [T3., T3., T3.], [T4., T4., T4.]])
        prev_preds = torch.stack(
            [preds[x] for x in adjacent_segments[segment]], 1).reshape(
            [batch_size, len(adjacent_segments[segment])])
        model = models[segment]
        input = segments_values[segment][0]
        # feed back previous predictions to input data
        input = torch.cat([input, prev_preds], 1)
        label = segments_values[segment][1]
        args_list.append([model, segment, input, label])
    with multiprocessing.Pool(processes=psutil.cpu_count()-1) as pool:
        preds = dict()
        loss = list()
        for pred in pool.starmap(single_training_step, args_list):
            preds[pred[0]] = pred[1]
            loss.append(pred[2])
    print(f'{datetime.datetime.now()} | Iteration: {idx}/{iterations} | Loss: '
          f'{np.mean(loss)}')
