__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Creation of image embeddings given a trained siamese model. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from vscvs.embeddings import create_embeddings
from vscvs.utils import load_siamese_model_from_checkpoint


@click.group()
@click.option('--branch', prompt='Siamese branch.', help='The siamese branch to be used to embed.',
              default=0, type=click.Choice(['0', '1']))
@pass_kwargs_to_context
def siamese(context, *_, **__):
    """ Image embedding creation. """
    context.obj['branch'] = int(context.obj['branch']) # `click.Choice` only admits `str`, so we must cast manually


@siamese.command()
@pass_context_to_kwargs
@click.option('--checkpoint', prompt='Checkpoint name', help='Name of the checkpoint directory.')
@click.option('--date', prompt='Checkpoint date', help='Checkpoint date (corresponds to the directory name.')
@click.option('--state-dict', prompt='State dict', help='The state_dict file to be loaded.')
@click.option('-t', '--tag', help='Optional tag for model checkpoint and tensorboard logs.', multiple=True)
def resnext(branch, dataset_name, embeddings_name, batch_size, workers, n_gpu, checkpoint, date, state_dict, tag):
    """ Create image embeddings with the ResNext model. """
    from vscvs.models import ResNextNormalized
    click.echo('Siamese ResNext embeddings for {} dataset'.format(dataset_name))
    model = load_siamese_model_from_checkpoint(ResNextNormalized, ResNextNormalized, state_dict, checkpoint, date, *tag)
    embedding_model = model.embedding_network_1 if branch else model.embedding_network_0
    create_embeddings(embedding_model.base, dataset_name, embeddings_name, batch_size, workers, n_gpu)
