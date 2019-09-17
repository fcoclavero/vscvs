__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Image retrieval given a query image. """


import click
import os
import pickle

from datetime import datetime
from sklearn.neighbors import NearestNeighbors


def retrieve(n_gpu):
    """

    :param n_gpu: number of available GPUs. If zero, the CPU will be used
    :type: int
    :return:
    """
    pass


@click.group()
def retrieve():
    """ Image retrieval click group. """
    pass


@retrieve.command()
@click.option(
    '--embedding_directory_name', prompt='Embedding directory', help='Static directory where embeddings are saved.'
)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def hog(embedding_directory_name, n_gpu):
    click.echo('Querying {} embeddings'.format(embedding_directory_name))
    from src.models.hog import HOG
    model = HOG()
