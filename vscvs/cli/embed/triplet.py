__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Creation of image embeddings given a trained triplet model. """


import click

from vscvs.cli.decorators import pass_context_to_kwargs
from vscvs.cli.decorators import pass_kwargs_to_context
from vscvs.embeddings import create_embeddings
from vscvs.utils import load_triplet_model_from_checkpoint


@click.group()
@click.option(
    "--branch",
    prompt="Triplet branch.",
    help="The branch of the triplet to be used to embed.",
    default="anchor",
    type=click.Choice(["anchor", "positive", "negative"]),
)
@pass_kwargs_to_context
def triplet(context, *_, **__):
    """ Image embedding creation. """
    context.obj["branch"] = context.obj["branch"]


@triplet.command()
@pass_context_to_kwargs
@click.option("--checkpoint", prompt="Checkpoint name", help="Name of the checkpoint directory.")
@click.option("--date", prompt="Checkpoint date", help="Checkpoint date (corresponds to the directory name).")
@click.option("--state-dict", prompt="State dict", help="The state_dict file to be loaded.")
@click.option("-t", "--tag", help="Optional tag for model checkpoint and tensorboard logs.", multiple=True)
def resnext(branch, dataset_name, embeddings_name, batch_size, workers, n_gpu, checkpoint, date, state_dict, tag):
    """ Create image embeddings with the ResNext model. """
    from vscvs.models import ResNextNormalized

    click.echo("Triplet ResNext embeddings for {} dataset".format(dataset_name))
    model = load_triplet_model_from_checkpoint(
        ResNextNormalized, ResNextNormalized, ResNextNormalized, state_dict, checkpoint, date, *tag
    )
    embedding_network_model = {
        "anchor": model.anchor_embedding_network,
        "positive": model.positive_embedding_network,
        "negative": model.negative_embedding_network,
    }[branch]
    create_embeddings(embedding_network_model.base, dataset_name, embeddings_name, batch_size, workers, n_gpu)
