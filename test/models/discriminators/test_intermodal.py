from __future__ import print_function

import pickle

import numpy as np

import torch
import torch.nn.parallel

from settings import DATA_SETS
from src.metrics import AccuracyBinary, PrecisionBinary, F1Binary, RecallBinary
from src.models.discriminators.intermodal import InterModalDiscriminator, InterModalDiscriminatorOneHot
from src.utils.data import split


def test_binary_classification(n_gpu = 1):
    data = pickle.load(open(DATA_SETS['sample_vectors']['pickle'], 'rb')) # Load sample data
    train, test = split(data)

    print('\n')
    print(
        'length: %s class 1: %s %% class 0: %s %%' %
        (len(train), sum(train['class']) / len(train) * 100, (len(train) - sum(train['class'])) / len(train) * 100)
    )
    print(
        'length: %s class 1: %s %% class 0: %s %%' %
        (len(test), sum(test['class']) / len(test) * 100, (len(test) - sum(test['class'])) / len(test) * 100)
    )

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    # Create data tensors
    x_train = torch.tensor(np.vstack(train['vector']), dtype=torch.float, device=device)
    y_train = torch.tensor(np.vstack(train['class']), dtype=torch.float, device=device)
    x_test = torch.tensor(np.vstack(test['vector']), dtype=torch.float, device=device)
    y_test = torch.tensor(np.vstack(test['class']), dtype=torch.float, device=device)

    loss_fn = torch.nn.BCELoss()

    learning_rate = 1e-1

    net = InterModalDiscriminator(input_dimension=100, n_gpu=n_gpu)
    net.to(device)

    print(net)

    # Define metrics
    metrics = [AccuracyBinary(), PrecisionBinary()] #, RecallBinary(), F1Binary()]

    epoch_size = 10
    epochs = 10

    for epoch in range(epochs):
        for t in range(epoch_size):
            y_pred = net(x_train)
            loss = loss_fn(y_pred, y_train)
            net.zero_grad()
            loss.backward()
            # Update metric object
            for metric in metrics: metric(net(x_test), y_test, loss)
            if t % 4 == 1:
                print('epoch: %s t: %s loss: %s' % (epoch, epoch * epoch_size, loss.item()))
                for metric in metrics:
                    print(metric)
                    metric.reset()
            with torch.no_grad():
                for param in net.parameters():
                    param.data -= learning_rate * param.grad


def test_onehot_classification(n_gpu = 1):
    data = pickle.load(open(DATA_SETS['sample_vectors_onehot']['pickle'], 'rb'))  # Load sample data
    train, test = split(data)

    print('\n')
    print(
        'length: %s class 1: %s %% class 0: %s %%' %
        (len(train), sum(train['class'].apply(lambda x: x[0])) / len(train) * 100, (len(train) - sum(train['class'].apply(lambda x: x[0]))) / len(train) * 100)
    )
    print(
        'length: %s class 0: %s %% class 1: %s %%' %
        (len(test), sum(test['class'].apply(lambda x: x[0])) / len(test) * 100, (len(test) - sum(test['class'].apply(lambda x: x[0]))) / len(test) * 100))

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    # Create data tensors
    x_train = torch.tensor(np.vstack(train['vector']), dtype=torch.float, device=device)
    y_train = torch.tensor(np.vstack(train['class']), dtype=torch.float, device=device)
    x_test = torch.tensor(np.vstack(test['vector']), dtype=torch.float, device=device)
    y_test = torch.tensor(np.vstack(test['class']), dtype=torch.float, device=device)

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-1

    net = InterModalDiscriminatorOneHot(input_dimension=100, n_gpu=n_gpu)
    net.to(device)

    print(net)

    def accuracy():
        y_pred = net(x_test)
        predicted_classes = y_pred.argmax(1)
        return (predicted_classes == y_test.argmax(1)).sum().float() / float(len(predicted_classes))

    epoch_size = 10
    epochs = 10

    max_accuracy = 0

    for epoch in range(epochs):
        for t in range(epoch_size):
            y_pred = net(x_train)
            loss = loss_fn(y_pred, y_train.argmax(1))
            net.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in net.parameters():
                    param.data -= learning_rate * param.grad
        current_accuracy = accuracy()
        if current_accuracy < max_accuracy:
            break
        else:
            print('epoch: %s t: %s loss: %s accuracy: %s' % (epoch, epoch * epoch_size, loss.item(), current_accuracy))
            max_accuracy = current_accuracy