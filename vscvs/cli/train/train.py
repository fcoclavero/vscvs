__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Model training entry point. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from vscvs.cli.train.optimizers import adabound, adam, adam_w, rms_prop, sgd
from vscvs.cli.train.gan import gan
from vscvs.cli.train.siamese import siamese
from vscvs.cli.train.triplet import triplet


""" Simple trainer commands. """


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--early-stopping-patience', prompt='Patience', help='Early stopping patience, in epochs', default=5)
def cnn(_,  *args, **kwargs):
    from vscvs.trainers.cnn import train_cnn
    click.echo('cnn - {} dataset'.format(kwargs['dataset_name']))
    train_cnn(*args, **kwargs)


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--pretrained', prompt='Pretrained', help='Whether to use pretrained model weights.', default=False)
@click.option('--early-stopping-patience', prompt='Patience', help='Early stopping patience, in epochs', default=5)
def resnet(_, *args, **kwargs):
    from vscvs.trainers.resnet import train_resnet
    click.echo('resnet - {} dataset'.format(kwargs['dataset_name']))
    train_resnet(*args, **kwargs)


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--pretrained', prompt='Pretrained', help='Whether to use pretrained model weights.', default=False)
@click.option('--early-stopping-patience', prompt='Patience', help='Early stopping patience, in epochs', default=5)
def resnext(_, *args, **kwargs):
    from vscvs.trainers.resnext import train_resnext
    click.echo('resnext - {} dataset'.format(kwargs['dataset_name']))
    train_resnext(*args, **kwargs)


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--processes', prompt='Number of parallel workers for batch graph creation', default=1,
              help='The number of parallel workers to be used for creating batch graphs.')
def classification_gcn(_, *args, **kwargs):
    from vscvs.trainers.classification_gcn import train_classification_gcn
    dataset_name = kwargs.pop('dataset_name')
    click.echo('classification GCN - {} dataset'.format(dataset_name))
    dataset_name = dataset_name + '-binary'
    train_classification_gcn(*args, dataset_name=dataset_name, **kwargs)


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--in-channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell-size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed-gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
@click.option('--processes', prompt='Number of parallel workers for batch graph creation', default=1,
              help='The number of parallel workers to be used for creating batch graphs.')
def hog_gcn(_, *args, **kwargs):
    from vscvs.trainers.hog_gcn import train_hog_gcn
    click.echo('HOG GCN - {} dataset'.format(kwargs['dataset_name']))
    train_hog_gcn(*args, **kwargs)


""" Global trainer group. """


@click.group()
@click.option('--resume-date', help='Date of the training checkpoint to be resumed.', default=None)
@click.option('--resume-checkpoint', help='Model checkpoint to be resumed.', default=None)
@click.option('--train-validation-split', prompt='Train/validation split',
              help='proportion of the training set that will be used for training.', default=.8)
@click.option('--batch-size', prompt='Batch size', help='The batch size during training.', default=16)
@click.option('--epochs', prompt='Number of epochs', help='The number of training epochs.', type=int)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@click.option('--tag', help='Optional tag for model checkpoint and tensorboard logs.')
@pass_kwargs_to_context
def train(context, **kwargs):
    """ Train a model. """
    pass


""" Add trainer commands to each optimizer trainer group, then add the optimizer groups to the global `train` group. """


for optimizer_group in [adabound, adam, adam_w, rms_prop, sgd]:
    # Compound trainer commands
    optimizer_group.add_command(gan)
    optimizer_group.add_command(siamese)
    optimizer_group.add_command(triplet)
    # Simple trainer commands
    optimizer_group.add_command(cnn)
    optimizer_group.add_command(resnet)
    optimizer_group.add_command(resnext)
    optimizer_group.add_command(classification_gcn)
    optimizer_group.add_command(hog_gcn)
    # Add optimizer group to global `train` group
    train.add_command(optimizer_group)
