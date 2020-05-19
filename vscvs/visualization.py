__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for displaying images. """


import click
import pickle
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import torchvision.utils as vutils

from plotly.offline import plot
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from vscvs.datasets import get_dataset
from vscvs.utils import get_path


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
    dataset = get_dataset(dataset_name) # load data
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers) # create the data_loader
    plot_image_batch(next(iter(data_loader))) # plot the batch


def plot_image_retrieval(query_image, query_image_class, query_dataset, queried_dataset, top_distances, top_indices):
    """
    Prints and plots the results of a retrieval query, showing the query image and the top results and distances.
    :param query_image: tensor with the original image pixels.
    :type: torch.Tensor
    :param query_image_class: name of the image's class.
    :type: str
    :param query_dataset: the Dataset that contains the query image.
    :type: torch.utils.data.Dataset
    :param queried_dataset: the Dataset that was queried.
    :type: torch.utils.data.Dataset
    :param top_distances: one-dimensional tensor with the distances of the query image's embedding to the top k most
    similar images' embeddings.
    :type: torch.Tensor
    :param top_indices: list of the indices of the top k most similar images in the dataset.
    :type: List[int]
    """
    aux = [queried_dataset[j] for j in top_indices]
    image_tensors = torch.stack([tup[0] for tup in aux])
    image_classes = [tup[1] for tup in aux]
    print('query image class = {}'.format(query_dataset.classes[query_image_class]))
    print('distances = {}'.format(top_distances))
    print('classes = {}'.format([queried_dataset.classes[class_name] for class_name in image_classes]))
    plot_image_batch([query_image, query_image_class])
    plot_image_batch([image_tensors, image_classes])


def plot_image(image, figsize=(8, 8), title=''):
    """
    Display a pyplot figure showing a random batch form the specified dataset.
    :param image: a PyTorch formatted image with its tensors in the first element.
    :type: List[torch.Tensor]
    :param figsize: figure width and height (respectively), in inches.
    :type: Tuple[float, float]
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
    :type: List[torch.Tensor]
    :param figsize: figure width and height (respectively), in inches.
    :type: Tuple[float, float]
    :param title: title to be displayed over the batch image grid.
    :type: str
    """
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def plot_embedding_tsne(dataset_name, embeddings_name, load_projection=False):
    """
    Plot a 2D projection of embeddings in the specified embedding directory using plotly.
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param embeddings_name: the name of the directory where the batch pickles will be saved.
    :type: str
    :param load_projection: load projections from pickles.
    :type: bool
    """
    from vscvs.embeddings import load_embeddings # import here to avoid circular import
    dataset = get_dataset(dataset_name)
    embeddings = load_embeddings(embeddings_name).to('cpu')
    projection_pickle_dir = get_path('embeddings', embeddings_name)
    if load_projection:
        click.echo('Loading existing 2D projection from pickle.')
        projection = pickle.load(open(os.path.join(projection_pickle_dir, 'tsne.pickle'), 'rb'))
        dataset_class_names = pickle.load(open(os.path.join(projection_pickle_dir, 'tsne_class_names.pickle'), 'rb'))
    else:
        click.echo('Creating 2D projection of the embeddings using TSNE')
        projection = TSNE(n_components=2).fit_transform(embeddings)
        dataset_class_names = [dataset.classes[tup[1]] for tup in tqdm(dataset, desc='Retrieving image class names')]
        pickle.dump(projection, open(os.path.join(projection_pickle_dir, 'tsne.pickle'), 'wb'))
        pickle.dump(dataset_class_names, open(os.path.join(projection_pickle_dir, 'tsne_class_names.pickle'), 'wb'))
    trace = go.Scattergl( # plot the resulting projection using plotly
        x=projection[:, 0],
        y=projection[:, 1],
        text=dataset_class_names,
        mode='markers',
        marker=dict(
            size=16,
            color=np.random.randn(len(projection)),
            colorscale='Viridis'))
    data = [trace]
    plot(data)
