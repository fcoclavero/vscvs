__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for displaying images. """


import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

from torch.utils.data import DataLoader

from src.datasets import get_dataset
from src.utils import get_device


def display_sample_batch(dataset_name, batch_size, workers, n_gpu):
    """
    Output a random batch form the specified dataset.
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param batch_size: size of batches for the embedding process.
    :type: int
    :param workers: number of data loader workers.
    :type: int
    :param n_gpu: number of available GPUs. If zero, the CPU will be used.
    :type: int
    """
    device = get_device(n_gpu)
    # Load data
    dataset = get_dataset(dataset_name)
    # Create the data_loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    # Plot the batch
    plot_image_batch(next(iter(data_loader)), device)


def plot_image_batch(batch, device, figsize=(8, 8), title=''):
    """
    Display a pyplot figure showing a random batch form the specified dataset.
    :param batch: a PyTorch formatted image batch with image tensors in the first element.
    :type: list<torch.Tensor, ...>
    :param device: device type specification
    :type: str
    :param figsize: figure width and height (respectively), in inches.
    :type: tuple<float, float>
    :param title: title to be displayed over the batch image grid.
    :type: str
    """
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
