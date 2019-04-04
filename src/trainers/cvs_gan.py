import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from src.datasets.sketchy import SketchyMixedBatches
from src.engines.cvs_gan import create_csv_gan_trainer
from src.models.discriminators.intermodal import InterModalDiscriminator
from src.models.generators.images import ImageEncoder
from src.utils.data import dataset_split, prepare_batch_gan


def train_cvs_gan(vector_dimension, workers=4, batch_size=16, n_gpu=0, epochs=2,
                  train_test_split=1, train_validation_split=.8):
    """
    Train a classification Convolutional Neural Network for image classes.
    :param vector_dimension: the dimensionality of the common vector space.
    :type: int
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
    :param train_validation_split: proportion of the training set that will be used for actual
    training. The remaining data will be used as the validation set.
    :type: float
    """
    dataset = SketchyMixedBatches('sketchy')

    train_set, validation_set, test_set = dataset_split(
        dataset, train_test_split, train_validation_split
    )

    # Create the data_loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    generator = ImageEncoder(feature_depth=64, output_dimension=vector_dimension)
    discriminator = InterModalDiscriminator(input_dimension=vector_dimension)

    # Define optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(generator.parameters(), lr=0.01, momentum=0.8)

    trainer = create_csv_gan_trainer(
        generator, discriminator, optimizer, criterion, device=device, prepare_batch=prepare_batch_gan
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

    trainer.run(train_loader, max_epochs=epochs)

    print('Finished Training')