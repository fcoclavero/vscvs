import click

from settings import DATA_SETS

from src.trainers.sketchy_cnn import train_sketchy_cnn
from src.trainers.cvs_gan import train_cvs_gan


@click.group()
def train():
    """ Train a model. """
    pass


@train.command()
def sketchy_cnn():
    click.echo('sketchy cnn')
    train_sketchy_cnn(DATA_SETS['sketchy_test']['images'], DATA_SETS['sketchy_test']['dimensions'][0])


@train.command()
def cvs_gan():
    click.echo('sketchy cnn')
    train_cvs_gan(DATA_SETS['sketchy_test']['images'], DATA_SETS['sketchy_test']['dimensions'][0])