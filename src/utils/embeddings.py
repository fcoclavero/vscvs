__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utilities for handling image embeddings. """


import os
import pickle
import torch

from statistics import mean
from torch.nn import PairwiseDistance, CosineSimilarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import get_dataset
from src.utils import get_device, recreate_directory
from src.utils.data import random_simple_split
from src.visualization import plot_image_retrieval


def create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu):
    """
    Creates embedding vectors for each element in the given DataSet by batches, and saves each batch as a pickle
    file in the given directory name (which will be a subdirectory of the data directory).
    :param model: name of the model to be used for embedding the DataSet.
    :type: torch.nn.Module
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param embeddings_name: the name of the pickle directory where the embeddings will be saved.
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
    # Delete and recreate the embedding directory to ensure we are working with an empty dir
    embedding_directory = os.path.join('data', 'embeddings', embeddings_name)
    recreate_directory(embedding_directory)
    for i, data in tqdm(enumerate(data_loader, 0), total=len(data_loader), desc='Embedding data'):  # iterate batches
        inputs, _ = data
        inputs = inputs.to(device)
        pickle.dump( # we pickle per file, as joining batches online results in massive RAM requirements
            model(inputs).to('cpu'),  # embeddings are sent to CPU before pickling, as a GPU might not be available
            open(os.path.join(embedding_directory, '{}.pickle'.format(i)), 'wb') # when they are loaded
        )


def load_embedding_pickles(embeddings_name, device):
    """
    Loads an embedding directory composed of pickled Tensors with image embeddings for a batch.
    :param embeddings_name: the name of the pickle directory where the embeddings are saved.
    :type: str
    :param device: device type specification
    :type: str
    :return: a single Pytorch tensor with all the embeddings found in the provided embedding directory. The later must
    contain pickled tensor objects with image embeddings.
    :type: torch.Tensor
    """
    embedding_directory = os.path.join('data', 'embeddings', embeddings_name)
    return torch.cat([
        pickle.load(open(os.path.join(embedding_directory, f), 'rb')) for f in
        tqdm(sorted(os.listdir(embedding_directory), key=len), desc='Loading {} embeddings'.format(embeddings_name))
        if 'tsne' not in f  # skip possible projection pickle in the embedding directory
    ])


def get_top_k(query_embedding, embeddings, k, distance, device):
    """
    Returns the distances and indices of the k nearest embeddings in the `embeddings` tensor to the `query_embedding`
    tensor.
    :param query_embedding:
    :param embeddings:
    :param k:
    :param distance: which distance function to be used for nearest neighbor computation. Either 'cosine' or 'pairwise'
    :type: str, either 'cosine' or 'pairwise'
    :param device: device type specification
    :type: str
    :return: the closest k embeddings in the `embeddings` tensor to the `query_embedding`. A tuple with a list of their
    distances and indices are returned (respectively).
    """
    p_dist = PairwiseDistance(p=2) if distance == 'pairwise' else CosineSimilarity()
    distances = p_dist(embeddings, query_embedding)
    return torch.topk(distances, k) # return the top k results


def retrieve_top_k(model, query_image_filename, query_dataset_name, queried_dataset_name, queried_embeddings_name,
                   k=16, distance='cosine', n_gpu=0):
    """
    Query the embeddings for a dataset with the given image. The image is embedded with the given model. Pairwise
    distances to the query image are computed for each embedding in the dataset, so the embeddings created by `model`
    must have the same length as the ones in the embedding directory. The `k` most similar images are displayed.
    :param model: name of the model to be used for embedding the DataSet.
    :type: torch.nn.Module
    :param query_image_filename: the complete file path and name to the image to be used as query. The images in the
    dataset that are most similar to this image will be displayed.
    :type: str
    :param query_dataset_name: name of the registered dataset to which the query image belongs.
    :type: str
    :param queried_dataset_name: name of the registered dataset to be queried.
    :type: str
    :param queried_embeddings_name: the name of the pickle file where the embeddings will be saved.
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
    query_dataset = get_dataset(query_dataset_name)
    queried_dataset = get_dataset(queried_dataset_name)
    # Load embeddings from pickle directory
    queried_embeddings = load_embedding_pickles(queried_embeddings_name, device)
    # Get the query image and create the embedding for it
    image, image_class = query_dataset.getitem_by_filename(query_image_filename)
    # Send elements to the specified device
    image, model = [var.to(device) for var in [image, model]]
    query_embedding = model(image.unsqueeze(0)) # unsqueeze to add the missing dimension expected by the model
    # Compute the distance to the query embedding for all images in the Dataset
    queried_embeddings, query_embedding = [var.to(device) for var in [queried_embeddings, query_embedding]]
    top_distances, top_indices = get_top_k(query_embedding, queried_embeddings, k, distance, device)
    plot_image_retrieval(image, image_class, query_dataset, queried_dataset, top_distances, top_indices)


def average_class_recall(dataset_name, embeddings_name, test_split, k, distance='cosine', n_gpu=0):
    """
    Computes the average class recall for the given embeddings. Embeddings are split into "test" and "queried" subsets.
    For each embedding in the "test" set, the nearest `k` embeddings in the "queried" set are retrieved using the given
    distance metric, giving us a recall for each test embedding that corresponds to the percentage of the top `k`
    retrieved embeddings that correspond to the same class as the test image. The average class recall is thus the
    average recall over all test images.
    :param dataset_name: name of the registered dataset which will be embedded.
    :type: str
    :param embeddings_name: the name of the pickle file where the embeddings will be saved.
    :type: str
    :param test_split: the proportion of the embeddings that will be used for the queries needed to compute the average
    class recall
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
    embeddings = load_embedding_pickles(embeddings_name, device).to(device)
    # Split embeddings into "test" and "queried" subsets
    query_embeddings, queried_embeddings, query_indexes, queried_indexes = random_simple_split(embeddings, test_split)
    recalls = []
    for i, query_embedding in \
            tqdm(enumerate(query_embeddings), total=query_embeddings.shape[0], desc='Computing recall'):
        _, top_indices = get_top_k(query_embedding.unsqueeze(0), queried_embeddings, k, distance, device)
        recalls.append(sum([dataset[queried_indexes[j]][1] == dataset[query_indexes[i]][1] for j in top_indices]) / k)
    print('Average class recall: {0:.2f}%'.format(mean(recalls) * 100))
