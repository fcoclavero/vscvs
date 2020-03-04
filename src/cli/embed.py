__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Creation of image embeddings given a trained model. """


import click
import os
import torch

from datetime import datetime

from settings import CHECKPOINT_NAME_FORMAT, ROOT_DIR
from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from src.embeddings import create_embeddings
from src.models import CNN, ResNet, ResNext
from src.utils import get_checkpoint_directory, remove_last_layer, get_out_features_from_state_dict


@click.group()
@click.option('--dataset-name', prompt='Dataset name', help='The name of the dataset to be embedded.',
              type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches']))
@click.option('--embeddings-name', prompt='Embeddings name', help='Name of file where the embeddings will be saved.')
@click.option('--batch-size', prompt='Batch size', help='The batch size for the embedding routine.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def embed(context, **kwargs):
    """ Image embedding creation click group. """
    pass


@embed.command()
@pass_context_to_kwargs
@click.option('--in-channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell-size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed-gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
def hog(_, dataset_name, embeddings_name, batch_size, workers, n_gpu,
        in_channels, cell_size, bins, signed_gradients):
    click.echo('HOG embeddings for {} dataset'.format(dataset_name))
    from src.models import HOG
    model = HOG(in_channels, cell_size, bins, signed_gradients)
    create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu)


@embed.command()
@pass_context_to_kwargs
@click.option('--date', prompt='Checkpoint date', help='Checkpoint date (corresponds to the directory name.')
@click.option('--checkpoint', prompt='Checkpoint name', help='Name of the checkpoint to be loaded.')
@click.option('--tag', help='Optional tag for model checkpoint and tensorboard logs.')
def cnn(_, dataset_name, embeddings_name, batch_size, workers, n_gpu, date, checkpoint, tag):
    click.echo('CNN embeddings for {} dataset'.format(dataset_name))
    checkpoint_directory = os.path.join(ROOT_DIR, 'data', 'checkpoints', 'CNN', checkpoint)
    state_dict = torch.load(os.path.join(checkpoint_directory, '{}.pth'.format(checkpoint)))
    model = CNN()
    model.load_state_dict(state_dict)
    model = remove_last_layer(model)
    create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu)


@embed.command()
@pass_context_to_kwargs
@click.option('--date', prompt='Checkpoint date', help='Checkpoint date (corresponds to the directory name.')
@click.option('--checkpoint', prompt='Checkpoint name', help='Name of the checkpoint to be loaded.')
@click.option('--tag', help='Optional tag for model checkpoint and tensorboard logs.')
def resnet(_, dataset_name, embeddings_name, batch_size, workers, n_gpu, date, checkpoint, tag):
    click.echo('ResNet embeddings for {} dataset'.format(dataset_name))
    date = datetime.strptime(date, CHECKPOINT_NAME_FORMAT)
    checkpoint_directory = get_checkpoint_directory('ResNext', tag=tag, date=date)
    state_dict = torch.load(os.path.join(checkpoint_directory, '{}.pth'.format(checkpoint)))
    out_features = get_out_features_from_state_dict(state_dict)
    model = ResNet(out_features=out_features)
    model.load_state_dict(state_dict)
    create_embeddings(model.resnet_base, dataset_name, embeddings_name, batch_size, workers, n_gpu)


@embed.command()
@pass_context_to_kwargs
@click.option('--date', prompt='Checkpoint date', help='Checkpoint date (corresponds to the directory name.')
@click.option('--checkpoint', prompt='Checkpoint name', help='Name of the checkpoint to be loaded.')
@click.option('--tag', help='Optional tag for model checkpoint and tensorboard logs.')
def resnext(_, dataset_name, embeddings_name, batch_size, workers, n_gpu, date, checkpoint, tag):
    click.echo('ResNext embeddings for {} dataset'.format(dataset_name))
    date = datetime.strptime(date, CHECKPOINT_NAME_FORMAT)
    checkpoint_directory = get_checkpoint_directory('ResNext', tag=tag, date=date)
    state_dict = torch.load(os.path.join(checkpoint_directory, '{}.pth'.format(checkpoint)))
    out_features = get_out_features_from_state_dict(state_dict)
    model = ResNext(out_features=out_features)
    model.load_state_dict(state_dict)
    create_embeddings(model.resnext_base, dataset_name, embeddings_name, batch_size, workers, n_gpu)
