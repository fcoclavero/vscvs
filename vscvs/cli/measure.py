__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Image retrieval benchmarks. """


import click

from torch.utils.data import Subset

from vscvs.cli.decorators import pass_context_to_kwargs
from vscvs.cli.decorators import pass_kwargs_to_context
from vscvs.datasets import get_dataset
from vscvs.embeddings import average_class_recall
from vscvs.embeddings import average_class_recall_parallel
from vscvs.embeddings import load_embeddings
from vscvs.utils import random_simple_split


@click.group()
def measure():
    """ Image retrieval benchmarks adn metrics. """
    pass


@measure.command()
@pass_context_to_kwargs
@click.option(
    "--dataset-name",
    prompt="Dataset name",
    help="The name of the Sketchy dataset variant.",
    type=click.Choice(["sketchy", "sketchy-test"]),
)
@click.option("--photo-embeddings-name", prompt="Photo embeddings name", help="Photo embeddings directory name.")
@click.option("--sketch-embeddings-name", prompt="Sketch embeddings name", help="Sketch embeddings directory name.")
@click.option(
    "--test-split", prompt="Test split", default=0.2, help="Proportion of the dataset to be used for queries."
)
@click.option(
    "-k", "--top-k", prompt="Top k", help="The amount of top results to be retrieved", default=1, multiple=True
)
@click.option(
    "--distance", prompt="Distance", help="The distance measure to be used.", type=click.Choice(["cosine", "pairwise"])
)
@click.option(
    "--n-gpu",
    prompt="Number of gpus",
    default=0,
    help="The number of GPUs available. Use 0 for CPU mode. Windows does not support CUDA multiprocessing.",
)
def sketchy_suite(dataset_name, photo_embeddings_name, sketch_embeddings_name, test_split, top_k, distance, n_gpu):
    """ Same mode recall for photos and sketches, as well as cross-modal recall using both modes as queries. """
    photo_dataset = get_dataset(f"{dataset_name}-photos")
    sketch_dataset = get_dataset(f"{dataset_name}-sketches")
    photo_embeddings = load_embeddings(photo_embeddings_name)
    sketch_embeddings = load_embeddings(sketch_embeddings_name)

    for k in top_k:
        click.echo(f"Calculating class recall@{k}\n\nphoto to photo:")
        query_embeddings, queried_embeddings, query_indices, queried_indices = random_simple_split(
            photo_embeddings, test_split
        )
        query_dataset, queried_dataset = Subset(photo_dataset, query_indices), Subset(photo_dataset, queried_indices)
        average_class_recall(  # photo to photo average class recall
            photo_dataset, queried_dataset, query_embeddings, queried_embeddings, k, distance, n_gpu
        )

        click.echo(f"sketch to sketch:")
        query_embeddings, queried_embeddings, query_indices, queried_indices = random_simple_split(
            sketch_embeddings, test_split
        )
        query_dataset, queried_dataset = Subset(sketch_dataset, query_indices), Subset(sketch_dataset, queried_indices)
        average_class_recall(  # sketch to sketch average class recall
            sketch_dataset, queried_dataset, query_embeddings, queried_embeddings, k, distance, n_gpu
        )

        click.echo(f"photo to sketch:")
        average_class_recall(  # photo to sketch average class recall
            photo_dataset, sketch_dataset, photo_embeddings, sketch_embeddings, k, distance, n_gpu
        )

        click.echo(f"sketch to photo:")
        average_class_recall(  # sketch to photo average class recall
            sketch_dataset, photo_dataset, sketch_embeddings, photo_embeddings, k, distance, n_gpu
        )


@measure.group()
@click.option("--k", prompt="Top k", help="The amount of top results to be retrieved", default=1)
@click.option(
    "--distance", prompt="Distance", help="The distance measure to be used.", type=click.Choice(["cosine", "pairwise"])
)
@click.option("--n-gpu", prompt="Number of gpus", help="The number of GPUs available. Use 0 for CPU mode.", default=0)
@pass_kwargs_to_context
def recall(_, **__):
    """ Image recall benchmarks. """
    pass


@recall.command()
@pass_context_to_kwargs
@click.option(
    "--dataset-name",
    prompt="Dataset name",
    help="The name of the dataset that corresponds to the given embeddings.",
    type=click.Choice(["sketchy-photos", "sketchy-sketches", "sketchy-test-photos", "sketchy-test-sketches"]),
)
@click.option("--embeddings-name", prompt="Embeddings name", help="Name of the embeddings directory.")
@click.option(
    "--test-split", prompt="Test split", default=0.2, help="Proportion of the dataset to be used for queries."
)
def same_class(dataset_name, embeddings_name, test_split, k, distance, n_gpu):
    """ Image recall of same class elements. """
    click.echo("Calculating class recall@{} for {} embeddings".format(k, embeddings_name))
    dataset = get_dataset(dataset_name)
    embeddings = load_embeddings(embeddings_name)
    query_embeddings, queried_embeddings, query_indices, queried_indices = random_simple_split(embeddings, test_split)
    query_dataset, queried_dataset = Subset(dataset, query_indices), Subset(dataset, queried_indices)
    average_class_recall(query_dataset, queried_dataset, query_embeddings, queried_embeddings, k, distance, n_gpu)


@measure.group()
def cross_modal():
    """ Cross modal retrieval benchmarks. """
    pass


@cross_modal.group()
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
@click.option("--query-embeddings-name", prompt="Query embeddings name", help="Query embeddings directory name.")
@click.option("--queried-embeddings-name", prompt="Queried embeddings name", help="Queried embeddings directory name.")
@click.option("--k", prompt="Top k", help="The amount of top results to be retrieved", default=1)
@click.option(
    "--distance", prompt="Distance", help="The distance measure to be used.", type=click.Choice(["cosine", "pairwise"])
)
@click.option(
    "--n-gpu",
    prompt="Number of gpus",
    default=0,
    help="The number of GPUs available. Use 0 for CPU mode. Windows does not support CUDA multiprocessing.",
)
@click.option(
    "--processes", prompt="Number of parallel workers", default=1, help="The number of parallel workers to be used."
)
@pass_kwargs_to_context
def recall(_, **__):
    """ Cross-modal image recall benchmarks. """
    pass


@recall.command()
@pass_context_to_kwargs
def same_class(
    query_dataset_name,
    queried_dataset_name,
    query_embeddings_name,
    queried_embeddings_name,
    k,
    distance,
    n_gpu,
    processes,
):
    """ Cross-modal image recall of same class elements. """
    click.echo(
        "Calculating cross modal class recall@{} for the {} and {} embeddings".format(
            k, query_embeddings_name, queried_embeddings_name
        )
    )
    query_dataset, queried_dataset = get_dataset(query_dataset_name), get_dataset(queried_dataset_name)
    query_embeddings = load_embeddings(query_embeddings_name)
    queried_embeddings = load_embeddings(queried_embeddings_name)
    average_class_recall_parallel(
        query_dataset, queried_dataset, query_embeddings, queried_embeddings, k, distance, n_gpu, processes
    )
