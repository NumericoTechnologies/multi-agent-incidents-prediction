import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

from sklearn.metrics import roc_auc_score

import os
import argparse
import sys
import datetime
import random
import h5py
import csv
import torch.multiprocessing as multiprocessing
import numpy as np

this_dir = os.path.abspath(os.path.dirname(__file__))
projects_dir = os.path.dirname(this_dir)
if projects_dir not in sys.path: sys.path.append(projects_dir)

from common_utils.general_utils import csv_to_df


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
    return segment_ID, net_out.detach(), loss.detach()


def single_evaluation_step(network, segment_ID, input, label, criterion, lr):
    network.eval()
    net_out = network(input)
    loss = criterion(net_out, label)
    return segment_ID, net_out.detach(), loss.detach()


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
            # feed back previous predictions to input data
            input = torch.cat([input, avg_prev_preds], 1)
        args_list.append([model, segment, input, label, criterion, lr])
    with multiprocessing.Pool(
            processes=multiprocessing.cpu_count() - 1) as pool:
        predictions = dict()
        loss = dict()
        for pred in pool.starmap(function, args_list):
            predictions[pred[0]] = pred[1]
            loss[pred[0]] = pred[2]
    return predictions, loss


def area_under_roc(label, preds):
    label = torch.cat(label).numpy()
    preds = torch.cat(preds).numpy()
    if len(np.unique(label)) == 1:
        return 0
    return roc_auc_score(label, preds)


def confusion(label, preds, inc_threshold):
    # Probability when it is count as incident or not
    preds = preds.clone().detach()
    preds[(preds >= inc_threshold)] = 1.0
    preds[(preds < inc_threshold)] = 0.0
    confusion_vector = preds / label
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    return true_positives, false_positives, true_negatives, false_negatives


def agent_based_prediction(train_dir, test_dir, out_dir, epochs, batch_size,
                           lr, sampling, sampling_rate, inc_threshold,
                           connected, adjacency_matrix_csv):
    os.makedirs(out_dir, exist_ok=True)
    segment_features = sorted([f for f in os.listdir(train_dir)])
    # define dictionaries for data, models, adjacent segments ...
    adjacency_matrix = csv_to_df(adjacency_matrix_csv, sep=',')
    models = dict()
    dataloaders_train = dict()
    dataloaders_test = dict()
    adjacent_segments = dict()
    criterion = nn.BCELoss()

    # drop all segments in adjacent matrix where no dataset is available
    segment_ids = list()
    segment_ids.append('0')
    for segment_id in segment_features:
        segment_ids.append(segment_id[:-3])
    adjacency_matrix = adjacency_matrix[segment_ids]
    adjacency_matrix = adjacency_matrix[
        adjacency_matrix['0'].isin(segment_ids)]

    # assign data loaders and models with respect to segments
    for segment in segment_features:
        segment_id = segment[:-3]
        models[segment_id] = Net()
        if connected:
            adjacent_segments[segment_id] = adjacency_matrix['0'].where(
                adjacency_matrix[segment_id] == 1).dropna().tolist()
            models[segment_id].fc1 = nn.Linear(
                25 + 1, 100)
        dataset_train = DatasetH5(os.path.join(train_dir, segment))
        dataset_test = DatasetH5(os.path.join(test_dir, segment))
        dataloaders_train[segment_id] = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=batch_size, drop_last=True)
        dataloaders_test[segment_id] = torch.utils.data.DataLoader(
            dataset=dataset_test, batch_size=batch_size, drop_last=True)

    # get number of iterations for train & test
    iterations_train = dataloaders_train[
        list(dataloaders_train.keys())[0]].__len__()
    iterations_test = dataloaders_test[
        list(dataloaders_test.keys())[0]].__len__()

    # start of training & evaluating the networks
    for epoch in range(epochs):
        print(f'Training Epoch {epoch}:')
        predictions = dict(zip(adjacent_segments.keys(),
                               [torch.tensor([[0.]] * batch_size)] * len(
                                   segment_features)))
        # training of the models
        avg_list = list()
        for idx, data in enumerate(zip(*dataloaders_train.values())):
            if sampling:
                num_ones = torch.stack([x[1] for x in data]).sum()
                if num_ones == 0:
                    if sampling_rate < random.uniform(0, 1):
                        continue
                avg = num_ones / batch_size
                avg_list.append(avg)
            segments_values = dict(zip(dataloaders_train.keys(), data))
            predictions, loss = network_iteration(segments_values,
                                                  predictions, models,
                                                  single_training_step,
                                                  adjacent_segments,
                                                  connected, batch_size,
                                                  criterion, lr)
            print(f'{datetime.datetime.now()} | Epoch: {epoch}/{epochs} | '
                  f'Iteration: {idx}/{iterations_train} | Loss: '
                  f'{np.mean([*loss.values()])}')
            break
        print(f'Ratio of incident & non-incident: {np.mean(avg_list)}')
        # testing of the models
        print('-' * 100)
        print(f'Evaluation Epoch {epoch}:')
        out_preds_name = f'Preds&Labels_{datetime.datetime.now()}_epoch' \
            f'{epoch}_batchsize{batch_size}_lr{lr}_sr{sampling_rate}' \
            f'_connected{connected}.csv'
        with open(os.path.join(out_dir, out_preds_name), 'w',
                  newline='') as out_preds:
            preds_writer = csv.DictWriter(out_preds, predictions.keys())
            preds_writer.writeheader()
            cum_loss = list()
            predictions = dict(zip(adjacent_segments.keys(),
                                   [torch.tensor([[0.]] * batch_size)] * len(
                                       segment_features)))
            # dict which stores confusion matrices & labels for all segments
            confusion_matrix = dict()
            for segment_id in segment_ids[1:]:
                confusion_matrix[segment_id] = [0, 0, 0, 0]
            for idx, data in enumerate(zip(*dataloaders_test.values())):
                segments_values = dict(zip(dataloaders_test.keys(), data))
                predictions, loss = network_iteration(segments_values,
                                                      predictions, models,
                                                      single_evaluation_step,
                                                      adjacent_segments,
                                                      connected, batch_size,
                                                      criterion, lr)
                cum_loss.append([*loss.values()])
                # store all preds & labels in dict to write them to csv
                preds_labels = dict()
                for segment_id in segment_ids[1:]:
                    labels = segments_values[segment_id][1]
                    preds = predictions[segment_id]
                    # convert to list & remove dimension of preds and labels
                    preds_labels[segment_id] = [sum(preds.tolist(), []),
                                                sum(labels.tolist(), [])]
                    tp, fp, tn, fn, = confusion(labels, preds, inc_threshold)
                    confusion_matrix[segment_id] = np.add(
                        confusion_matrix[segment_id],
                        [tp, fp, tn, fn]).tolist()
                preds_writer.writerow(preds_labels)
                if idx % 1000 == 0:
                    print(f'{datetime.datetime.now()} | Iteration: {idx}/'
                          f'{iterations_test} | Loss: {np.mean(cum_loss)}')
                    cum_loss = list()
        # Write result to output csv
        with open(os.path.join(out_dir, out_cm_name), 'w',
                  newline='') as out_cm:
            out_cm_name = f'ConfusionMatrix_{datetime.datetime.now()}_epoch' \
                f'{epoch}_batchsize{batch_size}_lr{lr}_sr{sampling_rate}' \
                f'_connected{connected}.csv'
            cm_writer = csv.writer(out_cm, delimiter='|')
            cm_writer.writerow(['segment_id', 'type', 'count'])
            for segment_id in segment_ids[1:]:
                cm_writer.writerow(
                    [segment_id, 'tp', confusion_matrix[segment_id][0]])
                cm_writer.writerow(
                    [segment_id, 'fp', confusion_matrix[segment_id][1]])
                cm_writer.writerow(
                    [segment_id, 'tn', confusion_matrix[segment_id][2]])
                cm_writer.writerow(
                    [segment_id, 'fn', confusion_matrix[segment_id][3]])
        print('-' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument('--train_dir', required=True, type=str,
                        help='Directory of HDF5 training data')
    parser.add_argument('--test_dir', required=True, type=str,
                        help='Directory of HDF5 testing data')
    parser.add_argument('--out_dir', default='results/', type=str,
                        help='Directory of HDF5 testing data')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of Epochs')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch Size')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning Rate')
    parser.add_argument('--sampling', default=True, type=bool,
                        help='Apply undersampling to the datasets')
    parser.add_argument('--sampling_rate', default=0.01, type=float,
                        help='Threshold for undersampling of training data')
    parser.add_argument('--inc_threshold', default=0.5, type=float,
                        help='Threshold for prediction to be an incident')
    parser.add_argument('--connected', default=0, type=int,
                        help='Network of Segments are connected during '
                             'training')
    parser.add_argument('--adjacency_matrix_csv', required=True, type=str,
                        help='csv adjacency matrix for Numerico segments')
    args = parser.parse_args()
    kwargs_ = dict(vars(args))
    agent_based_prediction(**kwargs_)
