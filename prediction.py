import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import os
import argparse
import sys
import datetime
import h5py
import torch.multiprocessing as multiprocessing
import pandas as pd
import numpy as np

this_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = os.path.dirname(this_dir)
if projects_dir not in sys.path: sys.path.append(projects_dir)

adjacency_matrix = pd.DataFrame(
    {'segments': ['278313080', '277315031', '277315032#277316021#277317004'],
     '278313080': [0, 1, 0], '277315031': [1, 0, 1],
     '277315032#277316021#277317004': [0, 1, 0]})


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(25, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 25)
        self.fc4 = nn.Linear(25, 1)

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


def single_training_step(network, segment_ID, input, label, criterion, lr):
    network.train()
    optimizer = optim.SGD(network.parameters(), lr=lr)
    optimizer.zero_grad()
    net_out = network(input)
    loss = criterion(net_out, label)
    loss.backward()
    optimizer.step()
    return [segment_ID, net_out.detach(), loss.detach()]


def single_evaluation_step(network, segment_ID, input, label, criterion, lr):
    network.eval()
    net_out = network(input)
    loss = criterion(net_out, label)
    return [segment_ID, net_out.detach(), loss.detach()]


def network_iteration(segments_values, predictions, models, function,
                      adjacent_segments, connected, batch_size, criterion,
                      lr):
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
        input = segments_values[segment][0]
        label = segments_values[segment][1]
        model = models[segment]
        if connected:
            prev_preds = torch.stack(
                [predictions[x] for x in adjacent_segments[segment]],
                1).reshape([batch_size, len(adjacent_segments[segment])])
            avg_prev_preds = torch.mean(prev_preds, 1).reshape([batch_size, 1])
            if len(input) == len(avg_prev_preds):
                # feed back previous predictions to input data
                input = torch.cat([input, avg_prev_preds], 1)
            else:
                return 0
        args_list.append([model, segment, input, label, criterion, lr])
    with multiprocessing.Pool(
            processes=multiprocessing.cpu_count() - 1) as pool:
        predictions = dict()
        loss = dict()
        for pred in pool.starmap(function, args_list):
            predictions[pred[0]] = pred[1]
            loss[pred[0]] = pred[2]
    return predictions, loss


def agent_based_prediction(train_dir, test_dir, epochs, batch_size, lr,
                           undersampling, undersampler_threshold, connected):
    segment_features = sorted([f for f in os.listdir(train_dir)])
    # define dictionaries for data, models & adjacent segments
    models = dict()
    dataloaders_train = dict()
    dataloaders_test = dict()
    adjacent_segments = dict()
    criterion = nn.BCELoss()

    # assign data loaders and models with respect to segments
    for segment in segment_features:
        segment_ID = segment[:-3]
        models[segment_ID] = Net()
        if connected:
            adjacent_segments[segment_ID] = adjacency_matrix['segments'].where(
                adjacency_matrix[segment_ID] == 1).dropna().tolist()
            models[segment_ID].fc1 = nn.Linear(
                25 + 1, 100)
        dataset_train = DatasetH5(os.path.join(train_dir, segment))
        dataset_test = DatasetH5(os.path.join(test_dir, segment))
        dataloaders_train[segment_ID] = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=batch_size)
        dataloaders_test[segment_ID] = torch.utils.data.DataLoader(
            dataset=dataset_test, batch_size=batch_size)

    # get number of iterations for train & test
    iterations_train = dataloaders_train[
        list(dataloaders_train.keys())[0]].__len__()
    iterations_test = dataloaders_test[
        list(dataloaders_test.keys())[0]].__len__()

    # start of training & evaluating the networks
    for epoch in range(epochs):
        print(f'Training Epoch {epoch}:')
        undersampler_counter = 0
        predictions = dict(zip(adjacent_segments.keys(),
                               [torch.tensor([[0.]] * batch_size)] * len(
                                   segment_features)))
        # training of the models
        for idx, data in enumerate(zip(*dataloaders_train.values())):
            if undersampling:
                if torch.stack([x[1] for x in data]).sum() == 0:
                    undersampler_counter += 1
                else:
                    undersampler_counter = 0
                if undersampler_counter > undersampler_threshold:
                    continue
            segments_values = dict(zip(dataloaders_train.keys(), data))
            predictions, loss = network_iteration(segments_values, predictions,
                                                  models, single_training_step,
                                                  adjacent_segments,
                                                  connected, batch_size,
                                                  criterion, lr)
            print(f'{datetime.datetime.now()} | Epoch: {epoch}/{epochs} | '
                  f'Iteration: {idx}/{iterations_train} | Loss: '
                  f'{np.mean([*loss.values()])}')
        # testing of the models
        print('-' * 100)
        print(f'Evaluation Epoch {epoch}:')
        cum_loss = list()
        predictions = dict(zip(adjacent_segments.keys(),
                               [torch.tensor([[0.]] * batch_size)] * len(
                                   segment_features)))
        for idx, data in enumerate(zip(*dataloaders_test.values())):
            segments_values = dict(zip(dataloaders_test.keys(), data))
            _, loss = network_iteration(segments_values, predictions, models,
                                        single_evaluation_step,
                                        adjacent_segments, connected,
                                        batch_size, criterion, lr)
            cum_loss.append([*loss.values()])
            if idx % 100 == 0:
                print(f'Iteration: {idx}/{iterations_test} | Loss: '
                      f'{np.mean(cum_loss)}')
                cum_loss = list()
        print('-' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('--train_dir', required=True, type=str,
                        help='Directory of HDF5 training data')
    parser.add_argument('--test_dir', required=True, type=str,
                        help='Directory of HDF5 testing data')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of Epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch Size')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning Rate')
    parser.add_argument('--undersampling', default=False, type=bool,
                        help='Apply undersampling to the datasets')
    parser.add_argument('--undersampler_threshold', default=100, type=int,
                        help='Threshold for undersampling of training data')
    parser.add_argument('--connected', default=False, type=bool,
                        help='Network of Segments are training independently')
    args = parser.parse_args()
    kwargs_ = dict(vars(args))
    agent_based_prediction(**kwargs_)
