__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Creation of image embeddings given a trained model. """


import click
import os
import torch

from src.utils.embeddings import create_embeddings
from settings import ROOT_DIR

@click.group()
def embed():
    """ Image embedding creation click group. """
    pass


@embed.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--embeddings_name', prompt='Embeddings name', help='Name of file where the embeddings will be saved.')
@click.option('--in_channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell_size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--n_bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed_gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
@click.option('--batch_size', prompt='Batch size', help='The batch size for the embedding routine.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def hog(dataset_name, embeddings_name, in_channels, cell_size, n_bins, signed_gradients,
        batch_size, workers, n_gpu):
    click.echo('HOG embeddings for {} dataset'.format(dataset_name))
    from src.models.hog import HOG
    model = HOG(in_channels, cell_size, n_bins, signed_gradients)
    create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu)


@embed.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--embeddings_name', prompt='Embeddings name', help='Name of file where the embeddings will be saved.')
@click.option('--checkpoint', prompt='Checkpoint date', help='Checkpoint date (corresponds to its directory name.')
@click.option('--epoch', prompt='Checkpoint epoch', help='Epoch corresponding to the model state to be loaded.')
@click.option('--batch_size', prompt='Batch size', help='The batch size for the embedding routine.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def cnn(dataset_name, embeddings_name, checkpoint, epoch, batch_size, workers, n_gpu):
    from src.models.convolutional.classification import ConvolutionalNetwork
    click.echo('CNN embeddings for {} dataset'.format(dataset_name))
    # Load the model checkpoint
    checkpoint_directory = os.path.join(ROOT_DIR, 'static', 'checkpoints', 'cnn', checkpoint)
    net = torch.load(os.path.join(checkpoint_directory, '_net_{}.pth'.format(epoch)))
    # This CNN is a classification model, so we will eliminate the last few layers to obtain embeddings with it
    state_dict = net.state_dict() # contains network layers
    del state_dict['fully_connected_3.weight'] # delete last layer weights
    del state_dict['fully_connected_3.bias'] # and bias
    model = ConvolutionalNetwork() # new model object with same structure as checkpoint, but without last layer
    model = model.load_state_dict(state_dict, strict=False) # load the trained model weights into it
    create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu)
