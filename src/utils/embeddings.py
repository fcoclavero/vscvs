__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for handling image embeddings. """


import os
import pickle
import shutil
import torch

from torch.nn import PairwiseDistance, CosineSimilarity
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets import get_dataset, get_dataset_class_names
from src.utils import get_device
from src.visualization import plot_image_batch, plot_image


def create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu):
    """
    Creates embedding vectors for each element in the given DataSet by batches, and saves each batch as a pickle
    file in the given directory name (which will be a subdirectory of the static directory).
    :param model: name of the model to be used for embedding the DataSet.
    :type: torch.nn.Module
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param embeddings_name: the name of the pickle file where the embeddings will be saved.
    :type: str
    :param batch_size: size of batches for the embedding process.
    :type: int
    :param workers: number of data loader workers.
    :type: int
    :param n_gpu: number of available GPUs. If zero, the CPU will be used.
    :type: int
    """
    device = get_device(n_gpu)
    model = model.to(device)
    # Load data
    dataset = get_dataset(dataset_name)
    # Create the data_loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    # Accumulate processed batches in a list, which will be converted into a single Tensor before pickling.
    embedding_batches = []
    for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader)):  # iterate batches
        inputs, _ = data
        embedding_batches.append(model(inputs.to(device)))
    # The embeddings are sent to CPU before pickling, as a GPU might not be available when they are loaded
    embeddings = torch.cat(embedding_batches).to('cpu')
    pickle.dump(embeddings, open(os.path.join('static', 'embeddings', '{}.pickle'.format(embeddings_name)), 'wb'))


def query_embeddings(model, query_image_filename, dataset_name, embeddings_name, k=16, distance='cosine', n_gpu=0):
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
    :param embeddings_name: the name of the pickle file where the embeddings will be saved.
    :type: str
    :param k: the number of most similar images that wil be displayed.
    :type: int
    :param distance: which distance function to be used for nearest neighbor computation. Either 'cosine' or 'pairwise'
    :type: str, either 'cosine' or 'pairwise'
    :param n_gpu: number of available GPUs. If zero, the CPU will be used.
    :type: int
    """
    device = get_device(n_gpu)
    # Load data
    dataset = get_dataset(dataset_name)
    # Load embeddings from pickle directory
    embeddings = load_embedding_pickles(embeddings_name, device)
    # Get the query image and create the embedding for it
    image, image_class = dataset.getitem_by_filename(query_image_filename)
    # Send elements to the specified device
    embeddings, image, model = [var.to(device) for var in [embeddings, image, model]]
    query_embedding = model(image.unsqueeze(0)) # unsqueeze to add the missing dimension expected by the model
    # Compute the distance to the query embedding for all images in the Dataset
    p_dist = PairwiseDistance(p=2) if distance == 'pairwise' else CosineSimilarity()
    distances = p_dist(embeddings, query_embedding)
    # Return the top k results
    top_distances, top_indices = torch.topk(distances, k)
    aux = [dataset[j] for j in top_indices]
    image_tensors = torch.stack([tup[0] for tup in aux])
    image_classes = [tup[1] for tup in aux]
    image_class_names = get_dataset_class_names(dataset_name)
    print('query image class = {}'.format(image_class_names[image_class]))
    print('distances = {}'.format(top_distances))
    print('classes = {}'.format([image_class_names[class_name] for class_name in image_classes]))
    plot_image_batch([image, image_class])
    plot_image_batch([image_tensors, image_classes])


def load_embedding_pickles(embeddings_name, device):
    """
    Loads an embedding directory composed of pickled Tensors with image embeddings for a batch.
    :param embeddings_name: the name of the pickle file where the embeddings are saved.
    :type: str
    :param device: device type specification
    :type: str
    :return: a single Pytorch tensor with all the embeddings found in the provided embedding directory. The later must
    contain pickled tensor objects with image embeddings.
    :type: torch.Tensor
    """
    return pickle.load(open(os.path.join('static', 'embeddings', '{}.pickle'.format(embeddings_name)), 'rb')).to(device)
