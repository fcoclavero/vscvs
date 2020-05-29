__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Creation of image embeddings given a trained model. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from vscvs.embeddings import create_embeddings
from vscvs.utils import load_classification_model_from_checkpoint, remove_last_layer


@click.group()
@click.option('--dataset-name', prompt='Dataset name', help='The name of the dataset to be embedded.',
              type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches']))
@click.option('--embeddings-name', prompt='Embeddings name', help='Name of file where the embeddings will be saved.')
@click.option('--batch-size', prompt='Batch size', help='The batch size for the embedding routine.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def embed(*_, **__):
    """ Image embedding creation. """
    pass


@embed.command()
@pass_context_to_kwargs
@click.option('--in-channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell-size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed-gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
def hog(dataset_name, embeddings_name, batch_size, workers, n_gpu, in_channels, cell_size, bins, signed_gradients):
    """ Create image embeddings with the HOG model. """
    click.echo('HOG embeddings for {} dataset'.format(dataset_name))
    from vscvs.models import HOG
    model = HOG(in_channels, cell_size, bins, signed_gradients)
    create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu)


@embed.command()
@pass_context_to_kwargs
@click.option('--checkpoint', prompt='Checkpoint name', help='Name of the checkpoint directory.')
@click.option('--date', prompt='Checkpoint date', help='Checkpoint date (corresponds to the directory name.')
@click.option('--state-dict', prompt='State dict', help='The state_dict file to be loaded.')
@click.option('-t', '--tag', help='Optional tag for model checkpoint and tensorboard logs.', multiple=True)
def cnn(dataset_name, embeddings_name, batch_size, workers, n_gpu, checkpoint, date, state_dict, tag):
    """ Create image embeddings with the CNN model. """
    click.echo('CNN embeddings for {} dataset'.format(dataset_name))
    from vscvs.models import CNN
    model = load_classification_model_from_checkpoint(CNN, state_dict, checkpoint, date, *tag)
    model = remove_last_layer(model)
    create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu)


@embed.command()
@pass_context_to_kwargs
@click.option('--checkpoint', prompt='Checkpoint name', help='Name of the checkpoint directory.')
@click.option('--date', prompt='Checkpoint date', help='Checkpoint date (corresponds to the directory name.')
@click.option('--state-dict', prompt='State dict', help='The state_dict file to be loaded.')
@click.option('-t', '--tag', help='Optional tag for model checkpoint and tensorboard logs.', multiple=True)
def resnet(dataset_name, embeddings_name, batch_size, workers, n_gpu, checkpoint, date, state_dict, tag):
    """ Create image embeddings with the ResNet model. """
    click.echo('ResNet embeddings for {} dataset'.format(dataset_name))
    from vscvs.models import ResNet
    model = load_classification_model_from_checkpoint(ResNet, state_dict, checkpoint, date, *tag)
    create_embeddings(model.base, dataset_name, embeddings_name, batch_size, workers, n_gpu)


@embed.command()
@pass_context_to_kwargs
@click.option('--checkpoint', prompt='Checkpoint name', help='Name of the checkpoint directory.')
@click.option('--date', prompt='Checkpoint date', help='Checkpoint date (corresponds to the directory name.')
@click.option('--state-dict', prompt='State dict', help='The state_dict file to be loaded.')
@click.option('-t', '--tag', help='Optional tag for model checkpoint and tensorboard logs.', multiple=True)
def resnext(dataset_name, embeddings_name, batch_size, workers, n_gpu, checkpoint, date, state_dict, tag):
    """ Create image embeddings with the ResNext model. """
    from vscvs.models import ResNext
    click.echo('ResNext embeddings for {} dataset'.format(dataset_name))
    model = load_classification_model_from_checkpoint(ResNext, state_dict, checkpoint, date, *tag)
    create_embeddings(model.base, dataset_name, embeddings_name, batch_size, workers, n_gpu)
