__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Entry script for the entire project. """


from dotenv import load_dotenv


# Load env
load_dotenv()


import click
import warnings

from src.cli import create, embed, gradient, measure, retrieve, show, train


# Suppress gensim 'detected Windows; aliasing chunkize to chunkize_serial' warning
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='tensorboard')


# Create a nested command from command groups in the src package
@click.group()
def cli():
    """ Click group for all of the project's scripts. """
    pass


# We must use add_command instead of CommandCollection to get a nested structure.
# https://stackoverflow.com/a/39416589
cli.add_command(gradient)

for group in [cli, gradient]:
    group.add_command(create)
    group.add_command(embed)
    group.add_command(measure)
    group.add_command(retrieve)
    group.add_command(show)
    group.add_command(train)

# Initialize the command line interface
if __name__ == '__main__':
    cli()
