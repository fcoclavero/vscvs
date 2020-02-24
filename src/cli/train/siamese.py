__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Siamese model training entry point. """


import click

from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context


@click.group()
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy', 'sketchy-test'])
)
@click.option('--margin', prompt='Margin', help='The margin for the Contrastive Loss.', default=.2)
@pass_kwargs_to_context
def siamese(context, **kwargs):
    """ Train a siamese model. """
    context.obj['dataset_name'] = context.obj['dataset_name'] + '-siamese'


@siamese.command()
@pass_context_to_kwargs
def cnn(_, *args, **kwargs):
    from src.trainers.siamese import train_siamese_cnn
    click.echo('siamese cnn - {} dataset'.format(kwargs['dataset_name']))
    train_siamese_cnn(*args, **kwargs)


@siamese.command()
@pass_context_to_kwargs
def resnet(_, *args, **kwargs):
    from src.trainers.siamese import train_siamese_resnet
    click.echo('siamese resnet - {} dataset'.format(kwargs['dataset_name']))
    train_siamese_resnet(*args, **kwargs)


@siamese.command()
@pass_context_to_kwargs
def resnext(_, *args, **kwargs):
    from src.trainers.siamese import train_siamese_resnext
    click.echo('siamese resnext - {} dataset'.format(kwargs['dataset_name']))
    train_siamese_resnext(*args, **kwargs)
