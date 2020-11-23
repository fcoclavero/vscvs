__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Entry script for the entire project. """


import warnings

import click

from dotenv import load_dotenv

from vscvs.cli import create
from vscvs.cli import embed
from vscvs.cli import gradient
from vscvs.cli import measure
from vscvs.cli import retrieve
from vscvs.cli import show
from vscvs.cli import train


# Load env
load_dotenv()


# Suppress gensim 'detected Windows; aliasing chunkize to chunkize_serial' warning
warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="tensorboard")


# Create a nested command from command groups in the src package
@click.group()
def cli():
    """ Sketch/image common vector space with semantic information. """
    click.echo("")


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
if __name__ == "__main__":
    cli()
