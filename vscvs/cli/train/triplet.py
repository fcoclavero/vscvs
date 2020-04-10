__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Triplet model training entry point. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from vscvs.loss_functions import ReductionMixin


@click.group()
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy', 'sketchy-test'])
)
@click.option(
    '--loss-reduction', prompt='Loss reduction', help='Reduction function for the loss function.',
    type=click.Choice(ReductionMixin.reduction_choices)
)
@click.option('--margin', prompt='Margin', help='The margin for the triplet loss.', default=.2)
@pass_kwargs_to_context
def triplet(context, **kwargs):
    """ Train a triplet model. """
    context.obj['dataset_name'] = context.obj['dataset_name'] + '-triplet'


@triplet.command()
@pass_context_to_kwargs
def cnn(_, *args, **kwargs):
    from vscvs.trainers.triplet import train_triplet_cnn
    click.echo('triplet cnn - {} dataset'.format(kwargs['dataset_name']))
    train_triplet_cnn(*args, **kwargs)\


@triplet.command()
@pass_context_to_kwargs
def resnet(_, *args, **kwargs):
    from vscvs.trainers.triplet import train_triplet_resnet
    click.echo('triplet resnet - {} dataset'.format(kwargs['dataset_name']))
    train_triplet_resnet(*args, **kwargs)\


@triplet.command()
@pass_context_to_kwargs
def resnext(_, *args, **kwargs):
    from vscvs.trainers.triplet import train_triplet_resnext
    click.echo('triplet resnext - {} dataset'.format(kwargs['dataset_name']))
    train_triplet_resnext(*args, **kwargs)
