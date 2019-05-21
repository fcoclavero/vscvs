import os
import torch

import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from tqdm import tqdm

from ignite.handlers import Timer, ModelCheckpoint, TerminateOnNan
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from settings import ROOT_DIR, CHECKPOINT_NAME_FORMAT
from src.datasets import get_dataset
from src.models.convolutional_network import ConvolutionalNetwork
from src.utils.data import dataset_split, prepare_batch


def train_cnn(dataset_name, train_test_split=.7, train_validation_split=.8, learning_rate=.01, momentum=.8,
              batch_size=16, workers=4, n_gpu=0, epochs=2, resume=None):
    """
    Train a classification Convolutional Neural Network for image classes.
    :param dataset_name: the name of the Dataset to be used for training
    :type: str
    :param train_test_split: proportion of the dataset that will be used for training.
    The remaining data will be used as the test set.
    :type: float
    :param train_validation_split: proportion of the training set that will be used for actual
    training. The remaining data will be used as the validation set.
    :type: float
    :param learning_rate: learning rate for optimizers
    :type: float
    :param momentum: momentum parameter for SGD optimizer
    :type: float
    :param batch_size: batch size during training
    :type: int
    :param workers: number of workers for data_loader
    :type: int
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :param epochs: the number of epochs used for training
    :type: int
    :param resume: checkpoint folder name containing model and checkpoint .pth files containing the information
    needed for resuming training. Folder names correspond to dates with the following format: `%y-%m-%dT%H-%M`
    :type: str
    """
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    # Defaults
    checkpoint_directory = os.path.join(
        ROOT_DIR, 'static', 'checkpoints', 'cnn', datetime.now().strftime(CHECKPOINT_NAME_FORMAT)
    )
    net = ConvolutionalNetwork()
    net.to(device)
    start_epoch = 0

    if resume:
        try:
            print('Loading checkpoint %s.' % resume)
            checkpoint_directory = os.path.join(ROOT_DIR, 'static', 'checkpoints', 'cnn', resume)
            checkpoint = torch.load(os.path.join(checkpoint_directory, 'checkpoint.pth'))
            start_epoch = checkpoint['epochs']
            net = torch.load(os.path.join(checkpoint_directory, '_net_%s.pth' % start_epoch))
            print('Checkpoint loaded.')
        except FileNotFoundError:
            print('No checkpoint file found for checkpoint %s.' % resume)
            raise

    print(net)

    # Load data
    dataset = get_dataset(dataset_name)

    # Create the data_loaders
    train_loader, validation_loader, test_loader = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=workers)
        for subset in dataset_split(dataset, train_test_split, train_validation_split)
    ]

    # Define loss and optimizers
    loss = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Create the Ignite trainer
    trainer = create_supervised_trainer(
        net, optimizer, loss, device=device, prepare_batch=prepare_batch
    )

    # Create a model evaluator
    evaluator = create_supervised_evaluator(
        net,
        metrics={
            'accuracy': Accuracy(),
            'loss': Loss(loss)
        },
        device=device
    )

    # Timer that measures the average time it takes to process a single batch of examples
    timer = Timer(average=True)

    timer.attach(
        trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED
    )

    # tqdm progressbar definitions
    pbar_description = 'TRAINING => loss: {:.6f}'
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=pbar_description.format(0))

    # Summary writer for Tensorboard logging
    # Reference: https://pytorch.org/docs/stable/tensorboard.html
    writer = SummaryWriter(os.path.join(ROOT_DIR, 'static', 'logs', 'cnn'))

    # Save network graph to Tensorboard
    # writer.add_graph(net, train_set)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        writer.add_scalar('loss', trainer.state.output)
        pbar.desc = pbar_description.format(trainer.state.output)
        pbar.update(1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        pbar.n = pbar.last_print_n = 0
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        for key, value in metrics.items():
            writer.add_scalar(key, value)
        tqdm.write("\nTraining Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        pbar.n = pbar.last_print_n = 0
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        for key, value in metrics.items():
            writer.add_scalar(key, value)
        tqdm.write("\nValidation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.COMPLETED)
    def save_checkpoint(trainer):
        new_checkpoint = {
            'dataset_name': dataset_name,
            'epochs': start_epoch + epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'momentum': momentum,
            'model': ConvolutionalNetwork(),
            'optimizer': optimizer,
            'last_run': datetime.now(),
            'average_epoch_duration': timer.value()
        }
        torch.save(new_checkpoint, os.path.join(checkpoint_directory, 'checkpoint.pth'))
        pbar.close()
        print('Finished Training')

    # Create a Checkpoint handler that can be used to periodically save objects to disc.
    # Reference: https://pytorch.org/ignite/handlers.html?highlight=checkpoint#ignite.handlers.ModelCheckpoint
    checkpoint_saver = ModelCheckpoint(
        checkpoint_directory, filename_prefix='',
        save_interval=1, n_saved=5, atomic=True, create_dir=True, save_as_state_dict=False, require_empty=False
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, {'net': net})

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    trainer.run(train_loader, max_epochs=epochs)