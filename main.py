__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Entry script for the entire project. """


import click
import warnings

from src.cli.embed import embed
from src.cli.retrieve import retrieve
from src.cli.train import train


# Suppress gensim 'detected Windows; aliasing chunkize to chunkize_serial' warning
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


# Create a nested command from command groups in the src package
@click.group()
def cli():
    pass


@click.command()
@click.option("--data_set", prompt="Dataset name", help="The name of the dataset.")
@click.option("--tsne_dimension", default=2, help="The target dimensionality for the lower dimension projection.")
def create_classes(data_set, tsne_dimension):
    """ Create and pickle a new classes data frame. """
    from src.preprocessing import create_classes_data_frame # import here to avoid loading word vectors on every command
    create_classes_data_frame(data_set, tsne_dimension)


@click.command()
@click.option("--n", prompt="Number of samples", help="The number of sample vectors to be created.", type=int)
@click.option("--dimension", prompt="Sample dimensionality", help="The dimension of sample vectors.", type=int)
def create_sample_vectors(n, dimension):
    """ Create and pickle a new classes data frame. """
    from src.preprocessing import create_sample_vectors
    create_sample_vectors(n, dimension)


# We must use add_command instead of CommandCollection to get a nested structure.
# https://stackoverflow.com/a/39416589
cli.add_command(create_classes)
cli.add_command(create_sample_vectors)
cli.add_command(embed)
cli.add_command(retrieve)
cli.add_command(train)


# Initialize the command line interface
if __name__ == '__main__':
    cli()