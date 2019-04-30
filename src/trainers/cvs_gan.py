import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from ignite.engine import Events

from src.datasets import get_dataset
from src.utils.collators import sketchy_mixed_collate
from src.trainers.engines.cvs_gan import create_csv_gan_trainer
from src.models.discriminators.intermodal import InterModalDiscriminator
from src.models.generators.images import ImageEncoder
from src.utils.data import dataset_split, prepare_batch_gan
from src.utils.initialize_weights import initialize_weights


def train_cvs_gan(dataset_name, vector_dimension, workers=4, batch_size=16, n_gpu=0, epochs=2,
                  train_test_split=1, train_validation_split=.8, learning_rate=0.0002, beta1=.5):
    """
    Train a GAN that generates a common vector space between photos and sketches.
    :param dataset_name: the name of the Dataset to be used for training
    :type: str
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
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=sketchy_mixed_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=sketchy_mixed_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, collate_fn=sketchy_mixed_collate
    )

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

    # Instance adversarial models
    generator = ImageEncoder(feature_depth=64, output_dimension=vector_dimension)
    discriminator = InterModalDiscriminator(input_dimension=vector_dimension)

    # Initialize weights, as done in the DCGAN paper
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    # Define loss and optimizers
    gan_loss = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    trainer = create_csv_gan_trainer(
        generator, discriminator, generator_optimizer, discriminator_optimizer, gan_loss,
        vector_dimension, device=device, prepare_batch=prepare_batch_gan
    )

    pbar_description = 'ITERATION - Generator loss: {:.4f} Discriminator loss: {:.4f}'
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=pbar_description.format(0, 0))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        pbar.desc = pbar_description.format(*trainer.state.output)
        pbar.update(1)

    trainer.run(train_loader, max_epochs=epochs)

    print('Finished Training')