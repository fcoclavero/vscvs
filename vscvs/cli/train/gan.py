__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" GAN model training entry point. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context


@click.group()
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy', 'sketchy-test'])
)
@pass_kwargs_to_context
def gan(context, **kwargs):
    """ Train a generative adversarial model. """
    context.obj['dataset_name'] = context.obj['dataset_name'] + '-mixed-batches'


@gan.command()
@pass_context_to_kwargs
@click.option(
    '--loss-reduction', prompt='Loss reduction', help='Reduction function for the loss function.',
    type=click.Choice(['none', 'mean', 'sum'])
)
@click.option('--loss-weight', prompt='Loss reduction', help='Reduction function for the loss function.', default=None)
@pass_kwargs_to_context
def cvs(_, *args, **kwargs):
    """ Train a generative adversarial model for common vector space creation. """
    from vscvs.trainers.gan import train_gan_cvs
    train_gan_cvs(*args, **kwargs)
