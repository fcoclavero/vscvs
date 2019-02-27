import warnings

# Suppress gensim 'detected Windows; aliasing chunkize to chunkize_serial' warning
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import click

from src.train import train
from src.create_classes import create_classes_data_frame


# Create a nested command from command groups in the src package
@click.group()
def cli():
    pass


@click.command()
@click.option("--data_set", prompt="Dataset name", help="The name of the dataset.")
@click.option("--tsne_dimension", default=2, help="The target dimensionality for the lower dimension projection.")
def create_classes(data_set, tsne_dimension):
    """ Create and pickle a new classes data frame. """
    create_classes_data_frame(data_set, tsne_dimension)


# We must use add_command instead of CommandCollection to get a nested structure.
# https://stackoverflow.com/a/39416589
cli.add_command(train)
cli.add_command(create_classes)


# Initialize the command line interface
if __name__ == '__main__':
    cli()
