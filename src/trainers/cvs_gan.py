import os
import torch

import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.handlers import Timer, ModelCheckpoint, TerminateOnNan
from ignite.engine import Events

from settings import ROOT_DIR, CHECKPOINT_NAME_FORMAT
from src.datasets import get_dataset
from src.utils.collators import sketchy_mixed_collate
from src.trainers.engines.cvs_gan import create_csv_gan_trainer
from src.models.discriminators.intermodal import InterModalDiscriminator
from src.models.generators.images import ImageEncoder
from src.utils.data import dataset_split, prepare_batch_gan
from src.utils.initialize_weights import initialize_weights


def train_cvs_gan(dataset_name, vector_dimension, train_test_split=.7, train_validation_split=.8, learning_rate=0.0002,
                  beta1=.5, batch_size=16, workers=4, n_gpu=0, epochs=2, resume=None):
    """
    Train a GAN that generates a common vector space between photos and sketches.
    :param dataset_name: the name of the Dataset to be used for training
    :type: str
    :param vector_dimension: the dimensionality of the common vector space.
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
        ROOT_DIR, 'static', 'checkpoints', 'csv_gan', datetime.now().strftime(CHECKPOINT_NAME_FORMAT)
    )
    # Instance adversarial models
    generator = ImageEncoder(feature_depth=64, output_dimension=vector_dimension)
    discriminator = InterModalDiscriminator(input_dimension=vector_dimension)
    # Initialize weights, as done in the DCGAN paper
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    generator.to(device)
    discriminator.to(device)
    start_epoch = 0

    if resume:
        try:
            print('Loading checkpoint %s.' % resume)
            checkpoint_directory = os.path.join(ROOT_DIR, 'static', 'checkpoints', 'csv_gan', resume)
            checkpoint = torch.load(os.path.join(checkpoint_directory, 'checkpoint.pth'))
            start_epoch = checkpoint['epochs']
            generator = torch.load(os.path.join(checkpoint_directory, 'generator_net_%s.pth' % start_epoch))
            discriminator = torch.load(os.path.join(checkpoint_directory, 'discriminator_net_%s.pth' % start_epoch))
            print('Checkpoint loaded.')
        except FileNotFoundError:
            print('No checkpoint file found for checkpoint %s.' % resume)
            raise

    print(generator)
    print(discriminator)

    # Load data
    dataset = get_dataset(dataset_name)

    # Create the data_loaders
    train_loader, validation_loader, test_loader = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=sketchy_mixed_collate)
        for subset in dataset_split(dataset, train_test_split, train_validation_split)
    ]

    # Define loss and optimizers
    gan_loss = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Create the Ignite trainer
    trainer = create_csv_gan_trainer(
        generator, discriminator, generator_optimizer, discriminator_optimizer, gan_loss,
        vector_dimension, device=device, prepare_batch=prepare_batch_gan
    )

    # Create a model evaluator
    # evaluator = create_cvs_gan_evaluator(
    #     net.embedding_network,
    #     metrics={
    #         'accuracy': TopKCategoricalAccuracy(k=50, output_transform=output_transform_gan)
    #     },
    #     device=device
    # )

    # Timer that measures the average time it takes to process a single batch of examples
    timer = Timer(average=True)

    timer.attach(
        trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED
    )

    pbar_description = 'ITERATION => Generator loss: {:.4f} Discriminator loss: {:.4f}'
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=pbar_description.format(0, 0))

    # Summary writer for Tensorboard logging
    # Reference: https://pytorch.org/docs/stable/tensorboard.html
    writer = SummaryWriter(os.path.join(ROOT_DIR, 'static', 'logs', 'cvs_gan'))

    # Save network graph to Tensorboard
    # writer.add_graph(generator, train_set)
    # writer.add_graph(discriminator, train_set)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        pbar.desc = pbar_description.format(*trainer.state.output)
        pbar.update(1)

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(trainer):
    #     pbar.n = pbar.last_print_n = 0
    #     evaluator.run(train_loader)
    #     metrics = evaluator.state.metrics
    #     tqdm.write("\nTraining Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #                .format(trainer.state.epoch, metrics['accuracy'], metrics['triplet_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        pbar.n = pbar.last_print_n = 0
        # evaluator.run(validation_loader)
        # metrics = evaluator.state.metrics
        # tqdm.write("\nValidation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
        #            .format(trainer.state.epoch, metrics['accuracy'], metrics['triplet_loss']))
        tqdm.write(' - Epoch complete')

    @trainer.on(Events.COMPLETED)
    def save_checkpoint(trainer):
        new_checkpoint = {
            'dataset_name': dataset_name,
            'vector_dimension': vector_dimension,
            'epochs': start_epoch + epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'beta1': beta1,
            'generator': ImageEncoder(feature_depth=64, output_dimension=vector_dimension),
            'discriminator': InterModalDiscriminator(input_dimension=vector_dimension),
            'generator_optimizer': generator_optimizer,
            'discriminator_optimizer': discriminator_optimizer,
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
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_saver, {'generator': generator, 'discriminator': discriminator}
    )

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    trainer.run(train_loader, max_epochs=epochs)