import click
import torch

from datetime import datetime
from sklearn.neighbors import NearestNeighbors


@click.group()
def retrieve():
    """ Train a model. """
    pass


@retrieve.command()
@click.option(
    '--dataset_name', prompt='Dataset name', help='The name of the dataset to be used for training.',
    type=click.Choice(['sketchy_photos', 'sketchy_sketches', 'sketchy_test_photos', 'sketchy_test_sketches'])
)
def hog(dataset_name):
    from src.trainers.cnn import train_cnn
    click.echo('cnn - %s dataset' % dataset_name)
    train_cnn(
        dataset_name, train_test_split, train_validation_split, lr, momentum, batch_size, workers, n_gpu, epochs, resume
    )


k = 16

start = datetime.now()

with torch.no_grad():
    results  = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine').fit(torch.stack(vectors).squeeze())

distances, indices = results.kneighbors(vectors[0].reshape(1,-1))

print('KNN search duration: %s' % (datetime.now() - start))