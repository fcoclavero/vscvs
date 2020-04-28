__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" GAN model training entry point. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from vscvs.loss_functions import ReductionMixin


@click.group()
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy', 'sketchy-test'])
)
@pass_kwargs_to_context
def gan(context, **kwargs):
    """ Train a generative adversarial model. """
    context.obj['dataset_name'] = context.obj['dataset_name'] + '-multimodal'


@gan.group()
@click.option(
    '--loss-reduction', prompt='Loss reduction', help='Reduction function for the loss function.',
    type=click.Choice(ReductionMixin.reduction_choices)
)
@click.option('--loss-weight', help='Reduction function for the loss function.', default=None)
@pass_kwargs_to_context
def multimodal(_, *args, **kwargs):
    """ Train a generative adversarial model for common vector space creation. """
    pass


@multimodal.command()
@pass_context_to_kwargs
def default(_, *args, **kwargs):
    """ Train a generative adversarial model for common vector space creation. """
    from vscvs.trainers.gan import train_gan_multimodal
    train_gan_multimodal(*args, **kwargs)


@multimodal.command()
@pass_context_to_kwargs
@click.option('--margin', prompt='Margin', help='The margin for the contrastive loss.', default=.2)
def siamese(_, *args, dataset_name=None, **kwargs):
    """
    Train a generative adversarial model for common vector space creation, adding a contrastive term to the
    generator network loss.
    """
    from vscvs.trainers.gan import train_gan_multimodal
    dataset_name = dataset_name + 'siamese'
    train_gan_multimodal(*args, dataset_name=dataset_name, **kwargs)

