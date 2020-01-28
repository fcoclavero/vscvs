__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Model training entry point. """


import click

from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from src.cli.train.optimizers import sgd
from src.cli.train.siamese import siamese
from src.cli.train.triplet import triplet


""" Simple trainer commands. """


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--early_stopping_patience', prompt='Patience', help='Early stopping patience, in epochs', default=5)
def cnn(_,  *args, **kwargs):
    from src.trainers.cnn import train_cnn
    click.echo('cnn - {} dataset'.format(kwargs['dataset_name']))
    train_cnn(*args, **kwargs)


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--early_stopping_patience', prompt='Patience', help='Early stopping patience, in epochs', default=5)
def resnet(_, *args, **kwargs):
    from src.trainers.resnet import train_resnet
    click.echo('resnet - {} dataset'.format(kwargs['dataset_name']))
    train_resnet(*args, **kwargs)


@click.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--early_stopping_patience', prompt='Patience', help='Early stopping patience, in epochs', default=5)
def resnext(_, *args, **kwargs):
    from src.trainers.resnext import train_resnext
    click.echo('resnext - {} dataset'.format(kwargs['dataset_name']))
    train_resnext(*args, **kwargs)


""" Global trainer group. """


@click.group()
@click.option('--resume_date', help='Epoch for checkpoint loading.', default=None)
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


""" Add every simple and compound trainer command to each optimizer trainer group """


for optimizer_group in [sgd]:
    # Compound trainer commands
    optimizer_group.add_command(siamese)
    optimizer_group.add_command(triplet)
    # Simple trainer commands
    optimizer_group.add_command(cnn)
    optimizer_group.add_command(resnet)
    optimizer_group.add_command(resnext)
    # optimizer_group.add_command(cvs_gan)
    # optimizer_group.add_command(classification_gcn)
    # optimizer_group.add_command(hog_gcn)


""" Add optimizer trainer groups to the global trainer group. """


train.add_command(sgd)


@train.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-mixed-batches', 'sketchy-test-mixed-batches'])
)
@click.option(
    '--vector-dimension', prompt='CVS dimensionality', help='Dimensionality of the common vector space.', default=300
)
def cvs_gan(_, resume, train_validation_split, batch_size, epochs, workers, n_gpu, dataset_name, vector_dimension):
    from src.trainers.cvs_gan import train_cvs_gan
    click.echo('cvs gan - %s dataset' % dataset_name)
    train_cvs_gan(dataset_name, vector_dimension, workers, batch_size, n_gpu, epochs)


@train.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--learning_rate', prompt='Learning rate', help='Learning rate for the optimizer', default=2e-4)
@click.option('--weight-decay', prompt='Weight decay', help='Weight decay parameter for Adam optimizer.', default=5e-4)
@click.option('--processes', prompt='Number of parallel workers for batch graph creation', default=1,
              help='The number of parallel workers to be used for creating batch graphs.')
def classification_gcn(_, *args, **kwargs):
    from src.trainers.classification_gcn import train_classification_gcn
    dataset_name = kwargs.pop('dataset_name')
    click.echo('classification GCN - {} dataset'.format(dataset_name))
    dataset_name = dataset_name + '-binary'
    train_classification_gcn(*args, dataset_name = dataset_name, **kwargs)


@train.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--in-channels', prompt='In channels', help='Number of image color channels.', default=3)
@click.option('--cell-size', prompt='Cell size', help='Gradient pooling size.', default=8)
@click.option('--bins', prompt='Number of histogram bins', help='Number of histogram bins.', default=9)
@click.option('--signed-gradients', prompt='Signed gradients', help='Use signed gradients?', default=False)
@click.option('--learning_rate', prompt='Learning rate', help='Learning rate for the optimizer', default=2e-4)
@click.option('--weight-decay', prompt='Weight decay', help='Weight decay parameter for Adam optimizer.', default=5e-4)
@click.option('--processes', prompt='Number of parallel workers for batch graph creation', default=1,
              help='The number of parallel workers to be used for creating batch graphs.')
def hog_gcn(_, *args, **kwargs):
    from src.trainers.hog_gcn import train_hog_gcn
    click.echo('HOG GCN - {} dataset'.format(kwargs['dataset_name']))
    train_hog_gcn(*args, **kwargs)
