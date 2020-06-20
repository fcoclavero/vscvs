__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Visualization. """


import os

from datetime import datetime

import click
import yaml

import torch

from vscvs.utils import CHECKPOINT_NAME_FORMAT
from vscvs.utils import get_checkpoint_path
from vscvs.utils import get_map_location

from .tensorboard import tensorboard


@click.group()
def show():
    """ Image visualization. """
    pass


show.add_command(tensorboard)


@show.command()
@click.option("--name", prompt="Checkpoint name", help="Name of the checkpoint directory.")
@click.option("--date", prompt="Checkpoint date", help="Checkpoint date (corresponds to the directory name).")
@click.option("-t", "--tag", help="Optional tag for model checkpoint and tensorboard logs.", multiple=True)
def checkpoint(name, date, tag):
    """ Show the contents of the specified trainer checkpoint. """
    date = datetime.strptime(date, CHECKPOINT_NAME_FORMAT)
    click.echo("Show the {} checkpoint".format(date))
    checkpoint_directory = get_checkpoint_path(name, *tag, date=date)
    trainer_checkpoint = torch.load(os.path.join(checkpoint_directory, "trainer.pt"), map_location=get_map_location())
    print(yaml.dump(trainer_checkpoint, allow_unicode=True, default_flow_style=False))  # yaml dump for pretty printing


@show.command()
@click.option("--path", prompt="Full path.", help="The full path to the image to be visualized.")
def image(path):
    """ Display the image in the given path. """
    from PIL import Image

    img = Image.open(path)
    img.show()


@show.command()
@click.option(
    "--dataset-name",
    prompt="Dataset name",
    help="The name of the dataset to be visualized.",
    type=click.Choice(["sketchy-photos", "sketchy-sketches", "sketchy-test-photos", "sketchy-test-sketches"]),
)
@click.option("--batch-size", prompt="Batch size", help="The batch size for the embedding routine.", default=16)
@click.option("--workers", prompt="Data loader workers", help="The number of workers for the data loader.", default=4)
def sample_batch(dataset_name, batch_size, workers):
    from vscvs.visualization import display_sample_batch

    display_sample_batch(dataset_name, batch_size, workers)


@show.command()
@click.option(
    "--dataset-name",
    prompt="Dataset name",
    help="The name of the dataset to be visualized.",
    type=click.Choice(["sketchy-photos", "sketchy-sketches", "sketchy-test-photos", "sketchy-test-sketches"]),
)
@click.option("--embeddings-name", prompt="Embedding directory", help="Static directory where embeddings are saved.")
@click.option("--load-projection", prompt="Load projection", help="Try to load pickled TSNE projections?", is_flag=True)
def embedding_tsne(dataset_name, embeddings_name, load_projection):
    from vscvs.visualization import plot_embedding_tsne

    click.echo("Display projection of the {} embeddings".format(embeddings_name))
    plot_embedding_tsne(dataset_name, embeddings_name, load_projection)
