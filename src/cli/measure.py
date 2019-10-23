__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'
 
 
""" Image retrieval benchmarks. """
 
 
import click

from torch.utils.data import Subset
 
from src.cli.decorators import pass_context_to_kwargs, pass_kwargs_to_context
from src.datasets import get_dataset
from src.utils.data import random_simple_split
from src.utils.embeddings import average_class_recall, load_embedding_pickles


@click.group()
def measure():
    """ Image retrieval benchmarks click group. """
    pass


@measure.group()
@click.option('--k', prompt='Top k', help='The amount of top results to be retrieved', default=10)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def recall(context, **kwargs):
    """ Image recall benchmarks click group. """
    pass


@recall.command()
@pass_context_to_kwargs
@click.option(
    '--dataset-name', prompt='Dataset name', help='The name of the dataset that corresponds to the given embeddings.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--embeddings-name', prompt='Embeddings name', help='Name of the embeddings directory.')
@click.option('--test-split', prompt='Test split', default=.2, help='Proportion of the dataset to be used for queries.')
def same_class(_, dataset_name, embeddings_name, test_split, k, n_gpu):
    click.echo('Calculating class recall@{} for {} embeddings'.format(k, embeddings_name))
    # Split dataset and embeddings into "query" and "queried" subsets
    dataset = get_dataset(dataset_name)
    embeddings = load_embedding_pickles(embeddings_name)
    query_embeddings, queried_embeddings, query_indices, queried_indices = random_simple_split(embeddings, test_split)
    query_dataset, queried_dataset = Subset(dataset, query_indices), Subset(dataset, queried_indices)
    average_class_recall(query_dataset, queried_dataset, query_embeddings, queried_embeddings, k, n_gpu)

 
@measure.group()
def cross_modal():
    """ Cross modal retrieval benchmarks click group. """
    pass
 
 
@cross_modal.group()
@click.option(
    '--query-dataset-name', prompt='Query dataset name', help='The name of the dataset that contains the query image.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option(
    '--queried-dataset-name', prompt='Queried dataset name', help='The name of the dataset that will be queried.',
    type=click.Choice(['sketchy-photos', 'sketchy-sketches', 'sketchy-test-photos', 'sketchy-test-sketches'])
)
@click.option('--query-embeddings-name', prompt='Query embeddings name', help='Query embeddings directory name.')
@click.option('--queried-embeddings-name', prompt='Queried embeddings name', help='Queried embeddings directory name.')
@click.option('--k', prompt='Top k', help='The amount of top results to be retrieved', default=10)
@click.option('--n-gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
@pass_kwargs_to_context
def recall(context, **kwargs):
    """ Cross-modal image recall benchmarks click group. """
    pass


@recall.command()
@pass_context_to_kwargs
def same_class(_, query_dataset_name, queried_dataset_name, query_embeddings_name, queried_embeddings_name, k, n_gpu):
    click.echo('Calculating cross modal class recall@{} for the {} and {} embeddings'.format(
        k, query_embeddings_name, queried_embeddings_name
    ))
    query_dataset, queried_dataset = get_dataset(query_dataset_name), get_dataset(queried_dataset_name)
    query_embeddings = load_embedding_pickles(query_embeddings_name)
    queried_embeddings = load_embedding_pickles(queried_embeddings_name)
    average_class_recall(query_dataset, queried_dataset, query_embeddings, queried_embeddings, k, n_gpu)
