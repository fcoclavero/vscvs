import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from settings import DATA_SETS
from src.datasets.sketchy import Sketchy
from src.models.convolutional_network import ConvolutionalNetwork
from src.utils.data import dataset_split


def train_sketchy_cnn(workers=4, batch_size=16, n_gpu=0, epochs=2, train_test_split=1, train_validation_split=.8):
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
    :param train_test_split: proportion of the dataset that will be used for training. The remaining data will be used
    as the test set.
    :type: float
    :param train_validation_split: proportion of the training set that will be used for actual training. The remaining
    data will be used as the validation set.
    :type: float
    """
    dataset = Sketchy(DATA_SETS['sketchy_test']['photos'])

    train_set, validation_set, test_set = dataset_split(dataset, train_test_split, train_validation_split)

    # Create the data_loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    net = ConvolutionalNetwork()
    net.to(device)
    print(net)

    # Define optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(
        net,
        metrics={
            'accuracy': Accuracy(),
            'nll': Loss(criterion)
        },
        device=device
    )

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        pbar.desc = desc.format(trainer.state.output)
        pbar.update(1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        tqdm.write("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        tqdm.write("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))
        pbar.n = pbar.last_print_n = 0
        pbar.close()

    trainer.run(train_loader, max_epochs=epochs)

    print('Finished Training')