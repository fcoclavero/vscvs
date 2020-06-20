__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Image retrieval given a query image. """


import os

import click

import torch

from vscvs.cli.decorators import pass_context_to_kwargs
from vscvs.cli.decorators import pass_kwargs_to_context
from vscvs.embeddings import retrieve_top_k
from vscvs.utils import get_map_location
from vscvs.utils import get_path


@click.group()
@click.option(
    "--query-image-file-path", prompt="Query image full path.", help="The dataset will be queried for similar images."
)
@click.option(
    "--query-dataset-name",
    prompt="Query dataset name",
    help="The name of the dataset that contains the query image.",
    type=click.Choice(["sketchy-photos", "sketchy-sketches", "sketchy-test-photos", "sketchy-test-sketches"]),
)
@click.option(
    "--queried-dataset-name",
    prompt="Queried dataset name",
    help="The name of the dataset that will be queried.",
    type=click.Choice(["sketchy-photos", "sketchy-sketches", "sketchy-test-photos", "sketchy-test-sketches"]),
)
@click.option("--queried-embeddings-name", prompt="Queried embeddings name", help="Queried embeddings directory name.")
@click.option("--k", prompt="Top k", help="The amount of top results to be retrieved", default=10)
@click.option("--n-gpu", prompt="Number of gpus", help="The number of GPUs available. Use 0 for CPU mode.", default=0)
@pass_kwargs_to_context
def retrieve(context, *_, **__):
    """ Image retrieval. """
    click.echo("Querying {} embeddings".format(context.obj["queried_embeddings_name"]))
    context.obj["query_dataset_name"] = context.obj["query_dataset_name"] + "-file-paths"
    context.obj["queried_dataset_name"] = context.obj["queried_dataset_name"] + "-file-paths"


@retrieve.command()
@pass_context_to_kwargs
@click.option("--in-channels", prompt="In channels", help="Number of image color channels.", default=3)
@click.option("--cell-size", prompt="Cell size", help="Gradient pooling size.", default=8)
@click.option("--bins", prompt="Number of histogram bins", help="Number of histogram bins.", default=9)
@click.option("--signed-gradients", prompt="Signed gradients", help="Use signed gradients?", default=False)
def hog(
    query_image_file_path,
    query_dataset_name,
    queried_dataset_name,
    queried_embeddings_name,
    k,
    n_gpu,
    in_channels,
    cell_size,
    bins,
    signed_gradients,
):
    """ Image retrieval for the HOG model. """
    from vscvs.models import HOG

    model = HOG(in_channels, cell_size, bins, signed_gradients)
    retrieve_top_k(
        model, query_image_file_path, query_dataset_name, queried_dataset_name, queried_embeddings_name, k, n_gpu
    )


@retrieve.command()
@pass_context_to_kwargs
@click.option("--checkpoint", prompt="Checkpoint date", help="Checkpoint date (corresponds to its directory name.")
@click.option("--epoch", prompt="Checkpoint epoch", help="Epoch corresponding to the model state to be loaded.")
def cnn(
    query_image_file_path,
    query_dataset_name,
    queried_dataset_name,
    queried_embeddings_name,
    k,
    n_gpu,
    checkpoint,
    epoch,
):
    """ Image retrieval for the CNN model. """
    checkpoint_directory = get_path("checkpoints", "cnn", checkpoint)  # Load the model checkpoint
    net = torch.load(os.path.join(checkpoint_directory, "_net_{}.pt".format(epoch)), map_location=get_map_location())
    retrieve_top_k(
        net.embedding_network,
        query_image_file_path,
        query_dataset_name,
        queried_dataset_name,
        queried_embeddings_name,
        k,
        n_gpu,
    )
