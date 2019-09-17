__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for handling image embeddings. """


import os
import pickle

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import get_dataset
from src.utils import get_device


def create_embeddings(embedding_directory_name, dataset_name, model, batch_size, workers, n_gpu):
    """
    Creates embedding vectors for each element in the given DataSet by batches, and saves each batch as a pickle
    file in the given directory name (which will be a subdirectory of the static directory).
    :param embedding_directory_name: the name of the subdirectory where the batch pickles will be saved
    :type: str
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param model: name of the model to be used for embedding the DataSet
    :type: torch.nn.Module
    :param batch_size: size of batches for the embedding process
    :type: int
    :param workers: number of data loader workers
    :type: int
    :param n_gpu: number of available GPUs. If zero, the CPU will be used
    :type: int
    """
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


def load_embedding_pickles():
    pass