import torch

import torch.nn as nn
import torch.optim as optim

from src.datasets import get_dataset
from src.metrics.multi_class import Accuracy, MeanAverageF1, MeanAverageRecall, MeanAveragePrecision
from src.models.convolutional import ConvolutionalNetwork
from src.utils import get_device


def test_train_sketchy_cnn(workers=4, batch_size=16, n_gpu=0, epochs=2):
    """
    Train a classification Convolutional Neural Network for image classes.
    :param workers: number of workers for data_loader
    :type: int
    :param batch_size: batch size during training
    :type: int
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :param epochs: the number of epochs used for training
    :type: int
    :return: None
    """
    dataset = get_dataset('sketchy-test-photos')

    # Create the data_loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = get_device(n_gpu)

    net = ConvolutionalNetwork()
    net.to(device)
    print(net)

    # Define optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Define metrics
    metrics = [Accuracy(), MeanAveragePrecision(), MeanAverageRecall(), MeanAverageF1()]

    # Training
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update metric object
            for metric in metrics: metric(outputs, labels, loss)

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                for metric in metrics: print(metric)
                running_loss = 0.0

    print('Finished Training')