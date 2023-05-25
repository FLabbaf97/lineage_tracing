import os
import wandb
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
# read in data
from utils import get_matrix_features, generate_all_permutations, flatten_3d_array

class BudDataset(Dataset):
    def __init__(self, data, augment=True):
        X = data['features'].to_numpy()
        labels = data['parent_index_in_candidates'].to_numpy()
        if(augment):
            X, labels = generate_all_permutations(X, labels)
        X = flatten_3d_array(X)
        self.data = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.zeros(len(labels), 5)  # initialize labels as zeros
        for i, label in enumerate(labels):
            if label != -1:
                # set the position of the correct parent to 1
                self.labels[i][label] = 1.0

    def __getitem__(self, index):
        data = self.data[index]
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class LineageNN(nn.Module):
    def __init__(self, layers):
        super(LineageNN, self).__init__()
        self.layers = nn.ModuleList()  # create an empty nn.ModuleList
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


def train_nn(train_df, eval_df, save_path='bst_nn.pth', config={},):
    if config == {}:
        config = {'epoch_n': 100, 'patience': 10, 'save_path': 'bst_nn.pth',
                  'augment': True, 'batch_size': 16, 'lr': 0.001, 'layers': [40, 32, 5]}
    # Initialize wandb
    wandb.init(project="lineage_tracing", config=config)
    # initialize neural network
    net = LineageNN(layers=config['layers'])

    # define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    train_bud_dataset = BudDataset(train_df, augment=config['augment'])
    train_bud_dataloader = DataLoader(
        train_bud_dataset, batch_size=config['batch_size'], shuffle=True)
    eval_bud_dataset = BudDataset(eval_df, augment=False)
    eval_bud_dataloader = DataLoader(
        eval_bud_dataset, batch_size=config['batch_size'], shuffle=True)

    # train your neural network
    patient = 0
    best_accuracy = 0.0
    for epoch in range(config['epoch_n']):
        running_loss = 0.0
        # training loop
        predicted_all = []
        labels_all = []
        net.train()
        for i, data in enumerate(train_bud_dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            # forward pass
            outputs = net(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)
            _, labels = torch.max(labels.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            predicted_all.extend(predicted)
            labels_all.extend(labels)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_accuracy = accuracy_score(labels_all, predicted_all)
        # eval loop
        predicted_all = []
        labels_all = []
        net.eval()
        for i, data in enumerate(eval_bud_dataloader, 0):
            inputs, labels = data

            with torch.no_grad():
                outputs = net(inputs)
            _, labels = torch.max(labels.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            predicted_all.extend(predicted)
            labels_all.extend(labels)
        eval_accuracy = accuracy_score(labels_all, predicted_all)
        if(eval_accuracy > best_accuracy):
            best_accuracy = eval_accuracy
            best_model = net
            torch.save(net.state_dict(), save_path)
            patient = 0
        else:
            patient += 1
        if(patient > config['patience']):
            # print('early stopping at ', epoch)
            break
            # print('Epoch' , i, ' loss: ', running_loss / len(bud_dataloader))
        wandb_log = {'epoch': epoch, 'patience': patient, 'eval_accuracy': eval_accuracy,
                     'train_accuracy': train_accuracy, 'best_accuracy': best_accuracy}
        wandb.log(wandb_log)

    # print('patient', patient)
    # print('best accuracy', best_accuracy)
    return best_model, best_accuracy


def test_nn(model, test_df):
    bud_dataset = BudDataset(test_df, augment=False)
    bud_dataloader = DataLoader(
        bud_dataset, batch_size=len(test_df), shuffle=False)
    for i, data in enumerate(bud_dataloader, 0):
        if (i > 0):
            print('more than one batch')
        inputs, labels = data
        with torch.no_grad():
            outputs = model(inputs)
        _, labels = torch.max(labels.data, 1)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(predicted, labels)
    print('test accuracy', accuracy)
    test_df['predicted'] = predicted
    return test_df, accuracy


def cv_nn(df, config={}):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df['features'].to_numpy()
    y = df['parent_index_in_candidates'].to_numpy()
    # repeat df because it necessary for the function to have two arguments
    skf.get_n_splits(X, y)
    accuracies = []
    models = []
    i = 0
    # repeat df because it necessary for the function to have two arguments
    for train_index, test_index in skf.split(X, y):
        i=i+1
        config['cv_number'] = i
        print('config: ', config)
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        net, accuracy = train_nn(train_df, test_df, config=config)
        accuracies.append(accuracy)
        models.append(net)
    
    print('accuracy: ', np.mean(accuracies), '+/-', np.std(accuracies))
    return models, accuracies
