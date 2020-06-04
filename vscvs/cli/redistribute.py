__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Offline embedding redistribution. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context


@click.group()
@click.option('--embeddings-name', prompt='Embeddings name', help='Name of file where the embeddings will be saved.')
@pass_kwargs_to_context
def redistribute(*_, **__):
    """ Image embedding redistribution. """
    pass


@redistribute.command()
@pass_context_to_kwargs
def gcn(embeddings_name):
    """ Redistribute using GCN over batch clique graphs. """
    click.echo('Siamese ResNext embeddings for {} embeddings'.format(embeddings_name))
