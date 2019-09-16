__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Creation of image embeddings given a trained model. """


import click
import os
import pickle

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.datasets import get_dataset
from src.utils import get_device


def create_embeddings(embedding_directory_name, dataset_name, model, batch_size, workers, n_gpu):
    device = get_device(n_gpu)
    # Load data
    dataset = get_dataset(dataset_name)
    # Create the data_loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    # Create and save embeddings
    embedding_directory = os.path.join('static', 'embeddings', embedding_directory_name)
    if not os.path.exists(embedding_directory):
        os.makedirs(embedding_directory)
    for i, data in tqdm(enumerate(data_loader, 0), total=int(len(dataset) / batch_size)):  # iterate batches
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pickle.dump(outputs, open(os.path.join(embedding_directory, 'batch_{}.pickle'.format(i)), 'wb'))


@click.group()
def embed():
    """ Image embedding creation click group. """
    pass


@embed.command()
@click.option(
    '--embedding_directory_name', prompt='Embedding directory', help='Static directory where embeddings will be saved.'
)
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
@click.option('--batch_size', prompt='Batch size', help='The batch size during training.', default=16)
@click.option('--workers', prompt='Data loader workers', help='The number of workers for the data loader.', default=4)
@click.option('--n_gpu', prompt='Number of gpus', help='The number of GPUs available. Use 0 for CPU mode.', default=0)
def hog(embedding_directory_name, dataset_name, batch_size, workers, n_gpu):
    click.echo('HOG embeddings for %s dataset' % dataset_name)
    from src.models.hog import HOG
    create_embeddings(embedding_directory_name, dataset_name, HOG(), batch_size, workers, n_gpu)
