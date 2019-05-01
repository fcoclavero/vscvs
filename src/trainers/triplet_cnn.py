import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data.dataloader import default_collate
from ignite.engine import Events

from src.datasets import get_dataset
from src.models.triplet_network import TripletNetwork
from src.utils.collators import triplet_collate
from src.models.convolutional_network import ConvolutionalNetwork
from src.trainers.engines.triplet_cnn import create_triplet_cnn_trainer
from src.utils.data import dataset_split, prepare_batch_gan


def train_triplet_cnn(dataset_name, vector_dimension, margin=.2, workers=4, batch_size=16, n_gpu=0, epochs=2,
                      train_test_split=1, train_validation_split=.8, learning_rate=0.0002, beta1=.5):
    """
    Train a triplet CNN that generates a vector space where vectors generated from similar (same class) images are close
    together and vectors from images of different classes are far apart.
    :param dataset_name: the name of the Dataset to be used for training
    :type: str
    :param vector_dimension: the dimensionality of the common vector space.
    :type: int
    :param margin: margin for the triplet loss
    :param workers: number of workers for data_loader
    :type: int
    :param batch_size: batch size during training
    :type: int
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :param epochs: the number of epochs used for training
    :type: int
    :param train_test_split: proportion of the dataset that will be used for training.
    The remaining data will be used as the test set.
    :type: float
    :param train_validation_split: proportion of the training set that will be used for actual training.
    The remaining data will be used as the validation set.
    :type: float
    :param learning_rate: learning rate for optimizers
    :type: float
    :param beta1: Beta1 hyper-parameter for Adam optimizers
    """
    dataset = get_dataset(dataset_name)

    train_set, validation_set, test_set = dataset_split(
        dataset, train_test_split, train_validation_split
    )

    # Create the data_loader
    collate = triplet_collate(default_collate)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=collate
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=collate
    )

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    # Instance adversarial models
    net = TripletNetwork(ConvolutionalNetwork())

    # Define loss and optimizers
    loss = nn.MarginRankingLoss(margin=margin)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    trainer = create_triplet_cnn_trainer(
        net, optimizer, loss, vector_dimension, device=device, prepare_batch=prepare_batch_gan
    )

    pbar_description = 'ITERATION - loss: {:.4f}'
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=pbar_description.format(0, 0))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        pbar.desc = pbar_description.format(trainer.state.output)
        pbar.update(1)

    trainer.run(train_loader, max_epochs=epochs)

    print('Finished Training')