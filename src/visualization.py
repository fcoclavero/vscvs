__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for displaying images. """


import click
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torchvision.utils as vutils

from plotly.offline import plot
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import ROOT_DIR
from src.datasets import get_dataset, get_dataset_class_names


def display_sample_batch(dataset_name, batch_size, workers):
    """
    Output a random batch form the specified dataset.
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param batch_size: size of batches for the embedding process.
    :type: int
    :param workers: number of data loader workers.
    :type: int
    """
    # Load data
    dataset = get_dataset(dataset_name)
    # Create the data_loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    # Plot the batch
    plot_image_batch(next(iter(data_loader)))


def plot_image(image, figsize=(8, 8), title=''):
    """
    Display a pyplot figure showing a random batch form the specified dataset.
    :param image: a PyTorch formatted image with it's tensors in the first element.
    :type: list<torch.Tensor, ...>
    :param figsize: figure width and height (respectively), in inches.
    :type: tuple<float, float>
    :param title: title to be displayed over the batch image grid.
    :type: str
    """
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(image[0].cpu())
    plt.show()


def plot_image_batch(batch, figsize=(8, 8), title=''):
    """
    Display a pyplot figure showing a random batch form the specified dataset.
    :param batch: a PyTorch formatted image batch with image tensors in the first element.
    :type: list<torch.Tensor, ...>
    :param figsize: figure width and height (respectively), in inches.
    :type: tuple<float, float>
    :param title: title to be displayed over the batch image grid.
    :type: str
    """
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def plot_embedding_tsne(dataset_name, embedding_directory_name, load_projection=False):
    """
    Plot a 2D projection of embeddings in the specified embedding directory using plotly.
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param embedding_directory_name: the name of the subdirectory where the batch pickles will be saved.
    :type: str
    """
    from src.utils.embeddings import load_embedding_pickles # import here to avoid circular import
    dataset = get_dataset(dataset_name)
    embeddings = load_embedding_pickles(embedding_directory_name, 'cpu')
    image_class_names = get_dataset_class_names(dataset_name)
    projection_pickle_dir = os.path.join(ROOT_DIR, 'data', 'embeddings', embedding_directory_name)
    if load_projection:
        click.echo('Loading existing 2D projection from pickle.')
        projection = pickle.load(open(os.path.join(projection_pickle_dir, 'tsne.pickle'), 'rb'))
        dataset_class_names = pickle.load(open(os.path.join(projection_pickle_dir, 'tsne_class_names.pickle'), 'rb'))
    else:
        click.echo('Creating 2D projection of the embeddings using TSNE')
        projection = TSNE(n_components=2).fit_transform(embeddings)
        dataset_class_names = [image_class_names[tup[1]] for tup in tqdm(dataset, desc='Retrieving image class names')]
        pickle.dump(projection, open(os.path.join(projection_pickle_dir, 'tsne.pickle'), 'wb'))
        pickle.dump(dataset_class_names, open(os.path.join(projection_pickle_dir, 'tsne_class_names.pickle'), 'wb'))
    # Plot the resulting projection using plotly
    trace = go.Scattergl(
        x=projection[:, 0],
        y=projection[:, 1],
        text=dataset_class_names,
        mode='markers',
        marker=dict(
            size=16,
            color=np.random.randn(len(projection)),
            colorscale='Viridis'
        )
    )
    data = [trace]
    plot(data)
