__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Entry script for the entire project. """


import click
import warnings

from src.cli.embed import embed
from src.cli.measure import measure
from src.cli.retrieve import retrieve
from src.cli.show import show
from src.cli.train import train


# Suppress gensim 'detected Windows; aliasing chunkize to chunkize_serial' warning
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


# Create a nested command from command groups in the src package
@click.group()
def cli():
    pass


@click.command()
@click.option('--dataset-name', prompt='Dataset name', help='Name of the dataset for which classes must be created.',
              type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches']))
@click.option('--tsne-dimension', default=2, help='The target dimensionality for the lower dimension projection.')
def create_classes(dataset_name, tsne_dimension):
    """" Create and pickle a new classes data frame. """
    from src.preprocessing import create_classes_data_frame # import here to avoid loading word vectors on every command
    from src.datasets import get_dataset
    create_classes_data_frame(get_dataset(dataset_name), tsne_dimension)


@click.command()
@click.option('--n', prompt='Number of samples', help='The number of sample vectors to be created.', type=int)
@click.option('--dimension', prompt='Sample dimensionality', help='The dimension of sample vectors.', type=int)
def create_sample_vectors(n, dimension):
    """ Create and pickle a new classes data frame. """
    from src.preprocessing import create_sample_vectors
    create_sample_vectors(n, dimension)


# We must use add_command instead of CommandCollection to get a nested structure.
# https://stackoverflow.com/a/39416589
cli.add_command(create_classes)
cli.add_command(create_sample_vectors)
cli.add_command(embed)
cli.add_command(measure)
cli.add_command(retrieve)
cli.add_command(show)
cli.add_command(train)


# Initialize the command line interface
if __name__ == '__main__':
    cli()
