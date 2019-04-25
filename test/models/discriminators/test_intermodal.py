from __future__ import print_function

import pickle

import numpy as np

import torch
import torch.nn.parallel

from settings import DATA_SOURCES
from src.metrics.multi_class import Accuracy, MeanAveragePrecision, AveragePrecision, MeanAverageRecall, AverageRecall, \
    MeanAverageF1, AverageF1
from src.models.discriminators.intermodal import InterModalDiscriminatorOneHot
from src.utils.data import split


def test_onehot_classification(n_gpu = 1):
    data = pickle.load(open(DATA_SOURCES['sample_vectors_onehot']['pickle'], 'rb'))  # Load sample data
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

    # Define metrics
    metrics = [
        Accuracy(), AveragePrecision(), MeanAveragePrecision(), AverageRecall(), MeanAverageRecall(),
        AverageF1(), MeanAverageF1()
    ]

    epoch_size = 10
    epochs = 10

    for epoch in range(epochs):
        for t in range(epoch_size):
            y_pred = net(x_train)
            loss = loss_fn(y_pred, y_train.argmax(1))
            net.zero_grad()
            loss.backward()
            for metric in metrics: metric(net(x_test), y_test.argmax(1), loss)
            if t % 4 == 1:
                print('epoch: %s t: %s loss: %s' % (epoch, epoch * epoch_size, loss.item()))
                for metric in metrics:
                    print(metric)
                    metric.reset()
            with torch.no_grad():
                for param in net.parameters():
                    param.data -= learning_rate * param.grad