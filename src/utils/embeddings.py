__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for handling image embeddings. """


import os
import pickle
import torch

from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import get_dataset
from src.utils import get_device
from src.visualization import plot_image_batch


def create_embeddings(model, dataset_name, embedding_directory_name, batch_size, workers, n_gpu):
    """
    Creates embedding vectors for each element in the given DataSet by batches, and saves each batch as a pickle
    file in the given directory name (which will be a subdirectory of the static directory).
    :param model: name of the model to be used for embedding the DataSet.
    :type: torch.nn.Module
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param embedding_directory_name: the name of the subdirectory where the batch pickles will be saved.
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
    # Create and save embeddings
    embedding_directory = os.path.join('static', 'embeddings', embedding_directory_name)
    if not os.path.exists(embedding_directory):
        os.makedirs(embedding_directory)
    for i, data in tqdm(enumerate(data_loader, 0), total=int(len(dataset) / batch_size)):  # iterate batches
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        pickle.dump(outputs, open(os.path.join(embedding_directory, 'batch_{}.pickle'.format(i)), 'wb'))


def query_embeddings(model, query_image_filename, dataset_name, embedding_directory_name, k, n_gpu):
    """
    Query the embeddings for a dataset with the given image. The image is embedded with the given model. Pairwise
    distances to the query image are computed for each embedding in the dataset, so the embeddings created by `model`
    must have the same length as the ones in the embedding directory. The `k` most similar images are displayed.
    :param model: name of the model to be used for embedding the DataSet.
    :type: torch.nn.Module
    :param query_image_filename: the complete file path and name to the image to be used as query. The images in the
    dataset that are most similar to this image will be displayed.
    :type: str
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param embedding_directory_name: the name of the subdirectory where the batch pickles will be saved.
    :type: str
    :param k: the number of most similar images that wil be displayed.
    :type: int
    :param n_gpu: number of available GPUs. If zero, the CPU will be used.
    :type: int
    """
    device = get_device(n_gpu)
    # Load data
    dataset = get_dataset(dataset_name)
    # Load embeddings from pickle directory
    embeddings = load_embedding_pickles(embedding_directory_name)
    # Get the query image and create the embedding for it
    image, _ = dataset.getitem_by_filename(query_image_filename)
    # Send elements to the specified device
    embeddings, image, model = embeddings.to(device), image.to(device), model.to(device)
    query_embedding = model(image.unsqueeze(0)) # unsqueeze to add the missing dimension expected by the model
    # Compute the distance to the query embedding for all images in the Dataset
    p_dist = PairwiseDistance(p=2)
    distances = p_dist(embeddings, query_embedding)
    # Return the top k results
    top_distances, top_indices = torch.topk(distances, k)
    print(top_distances)
    aux = [dataset[j] for j in top_indices]
    image_tensors = torch.stack([tup[0] for tup in aux])
    image_classes = [tup[1] for tup in aux]
    plot_image_batch([image_tensors, image_classes], device)


def load_embedding_pickles(embedding_directory_name):
    """
    Loads an embedding directory composed of pickled Tensors with image embeddings for a batch.
    :param embedding_directory_name: the name of the subdirectory where the batch pickles will be saved
    :type: str
    :return: a single Pytorch tensor with all the embeddings found in the provided embedding directory. The later must
    contain pickled tensor objects with image embeddings.
    :type: torch.Tensor
    """
    embedding_directory = os.path.join('static', 'embeddings', embedding_directory_name)
    return torch.cat([
        pickle.load(open(os.path.join(embedding_directory, f), 'rb')) for f in tqdm(os.listdir(embedding_directory))
    ])
