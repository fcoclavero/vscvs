__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Create tensorboard visualizations. """


import os

import click

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from vscvs.embeddings import load_embeddings
from vscvs.utils import get_path


@click.group()
def tensorboard():
    """ Display in Tensorboard. """
    pass


@tensorboard.command()
@click.option(
    "--dataset-name",
    prompt="Dataset name",
    help="The name of the dataset from which the embeddings were created.",
    type=click.Choice(["sketchy-photos", "sketchy-sketches", "sketchy-test-photos", "sketchy-test-sketches"]),
)
@click.option("--embeddings-name", prompt="Embedding directory", help="Static directory where embeddings are saved.")
@click.option("--metadata", prompt="Dataset metadata.", help="The Dataset metadata.tsv.", type=click.Path(exists=True))
@click.option("-t", "--tag", help="Optional tags for organizing embeddings.", multiple=True)
def embeddings(dataset_name, embeddings_name, metadata, tag):
    """ Add the embeddings in the given path to Tensorboard. """
    from pandas import read_csv
    from vscvs.utils import sprite_tensor

    classes = read_csv(metadata, delimiter="\t")["class"]
    embeddings_tensor = load_embeddings(embeddings_name)
    writer = SummaryWriter(get_path("tensorboard", "embeddings", embeddings_name))
    writer.add_embedding(
        embeddings_tensor,
        metadata=classes,
        tag="/".join((embeddings_name,) + tag),
        label_img=sprite_tensor(dataset_name),
    )
    writer.close()
    click.echo("Embeddings added to Tensorboard: {}".format(embeddings_name))


@tensorboard.command()
@click.option("--path", prompt="Full path.", help="The full path to the image to be visualized.")
@click.option("-t", "--tag", help="Optional tags for organizing embeddings.", multiple=True)
def image(path, tag):
    """ Add the image in the given path to Tensorboard. """
    from PIL import Image

    image_name = os.path.basename(path)
    transform = transforms.ToTensor()
    writer = SummaryWriter(get_path("tensorboard", "images", *tag))
    writer.add_image(image_name, transform(Image.open(path)), 0)
    writer.close()
    click.echo("Image added to Tensorboard: {}".format(image_name))
