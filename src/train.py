import click

from settings import DATA_SETS

from src.trainers.sketchy_cnn import train_sketchy_cnn


@click.group()
def train():
    """ Train a model. """
    pass


@train.command()
def sketchy_cnn():
    click.echo('sketchy cnn')
    train_sketchy_cnn(DATA_SETS['sketchy_test']['images'], DATA_SETS['sketchy_test']['dimensions'][0])
