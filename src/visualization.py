__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for displaying images. """


import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

from torch.utils.data import DataLoader

from src.datasets import get_dataset


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
