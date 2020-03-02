__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Triplet model training entry point. """


import click

from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context


@click.group()
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy', 'sketchy-test'])
)
@click.option('--margin', prompt='Margin', help='The margin for the triplet loss.', default=.2)
@pass_kwargs_to_context
def triplet(context, **kwargs):
    """ Train a triplet model. """
    context.obj['dataset_name'] = context.obj['dataset_name'] + '-triplets'


@triplet.command()
@pass_context_to_kwargs
@click.option(
    '--vector-dimension', prompt='CVS dimensionality', help='Dimensionality of the vector space.', default=300
)
@click.option('--margin', prompt='Margin', help='The margin for the Triplet Loss.', default=.2)
@click.option('--lr', prompt='Learning rate', help='Learning rate for the optimizer', default=2e-4)
@click.option('--beta1', prompt='Beta 1', help='Decay parameter for Adam optimizer.', default=.2)
def cnn(_, resume, train_validation_split, batch_size, epochs, workers, n_gpu, dataset_name, vector_dimension,
        margin, lr, beta1):
    from src.trainers.triplet import train_triplet_cnn
    click.echo('triplet cnn - %s dataset' % dataset_name)
    train_triplet_cnn(dataset_name, vector_dimension, resume, margin, workers, batch_size, n_gpu, epochs, lr, beta1)
