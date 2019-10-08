__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Image retrieval given a query image. """


import click
import os
import torch

from settings import ROOT_DIR
from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from src.utils.embeddings import query_embeddings


@click.group()
@click.option(
    '--query_image_filename', prompt='Query image full path.', help='The dataset will be queried for similar images.'
)
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--embeddings_name', prompt='Embeddings name', help='Name of file where the embeddings will be saved.')
@click.option('--k', prompt='Top k', help='The amount of top results to be retrieved', default=10)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def retrieve(context, **kwargs):
    """ Image retrieval click group. """
    # Use the `_filenames` variant of the selected dataset to allow retrieval by filename
    context.obj['dataset_name'] = context.obj['dataset_name'] + '_filenames'


@retrieve.command()
@pass_context_to_kwargs
@click.option('--in_channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell_size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--n_bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed_gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
def hog(_, query_image_filename, dataset_name, embeddings_name, k, n_gpu,
        in_channels, cell_size, n_bins, signed_gradients):
    click.echo('Querying {} embeddings'.format(embeddings_name))
    from src.models.hog import HOG
    model = HOG(in_channels, cell_size, n_bins, signed_gradients)
    query_embeddings(model, query_image_filename, dataset_name, embeddings_name, k, n_gpu)


@retrieve.command()
@pass_context_to_kwargs
@click.option('--checkpoint', prompt='Checkpoint date', help='Checkpoint date (corresponds to its directory name.')
@click.option('--epoch', prompt='Checkpoint epoch', help='Epoch corresponding to the model state to be loaded.')
def cnn(_, query_image_filename, dataset_name, embeddings_name, k, n_gpu, checkpoint, epoch):
    click.echo('CNN embeddings for {} dataset'.format(dataset_name))
    # Load the model checkpoint
    checkpoint_directory = os.path.join(ROOT_DIR, 'data', 'checkpoints', 'cnn', checkpoint)
    net = torch.load(os.path.join(checkpoint_directory, '_net_{}.pth'.format(epoch)))
    # This CNN is a classification model, so we will eliminate the last few layers to obtain embeddings with it
    query_embeddings(net.embedding_network, query_image_filename, dataset_name, embeddings_name, k, n_gpu)
