__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Click groups for different optimizers. """


import click

from src.cli.decorators import pass_kwargs_to_context


@click.group()
@click.option('--learning_rate', prompt='Learning rate', help='Learning rate for the optimizer', default=2e-4)
@click.option('--momentum', prompt='Momentum', help='Momentum parameter for SGD optimizer.', default=.2)
@pass_kwargs_to_context
def sgd(context, **kwargs):
    """ Train models using an SGD optimizer. """
    pass