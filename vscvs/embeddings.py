__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Utilities for handling image embeddings. """


import csv

from functools import reduce
from itertools import repeat
from statistics import mean

from tqdm import tqdm

import torch

from torch.multiprocessing import Pool
from torch.nn import CosineSimilarity
from torch.nn import PairwiseDistance
from torch.utils.data import DataLoader
from vscvs.datasets import get_dataset
from vscvs.decorators import log_time
from vscvs.decorators import torch_no_grad
from vscvs.utils import get_device
from vscvs.utils import get_map_location
from vscvs.utils import get_path
from vscvs.visualization import plot_image_retrieval


@log_time
@torch_no_grad
def create_embeddings(model, dataset_name, embeddings_name, batch_size, workers, n_gpu):
    """
    Creates embedding vectors for each element in the given DataSet, and saves a single Pytorch tensor of shape
    `len(dataset), embeddings_size` with all the embeddings in a file with the given name.
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
    model = model.eval().to(device)
    dataset = get_dataset(dataset_name)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    embedding_list = [
        model(data[0].to(device))  # model output sent to 'cpu' to prevent gpu memory overflow
        for data in tqdm(data_loader, total=len(data_loader), desc="Embedding data")
    ]
    torch.save(torch.cat(embedding_list).to("cpu"), open(get_path("embeddings", "{}.pt".format(embeddings_name)), "wb"))


@log_time
def create_metadata_tsv(dataset_name):
    """
    Creates a metadata TSV file that can be used with any embeddings of the given dataset to be displayed in the
    Tensorboard embedding projector.
    :param dataset_name: name of the registered DatasetFolder which will be embedded.
    :type: str
    """
    dataset = get_dataset(dataset_name)
    with open(get_path("embeddings", "{}.tsv".format(dataset_name)), "w") as f:
        writer = csv.DictWriter(f, ["class_idx", "class", "path"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                {"class_idx": idx, "class": dataset.classes[idx], "path": path}
                for path, idx in tqdm(dataset.samples, desc="Creating tsv")
            ]
        )


def load_embeddings(embeddings_name):
    """
    Loads an embedding directory composed of pickled Tensors with image embeddings for a batch.
    :param embeddings_name: the name of the pickle directory where the embeddings are saved.
    :type: str
    :return: a Pytorch tensor with all the embeddings found in the provided embedding directory. The later must
    contain pickled tensor objects with image embeddings.
    :type: torch.Tensor
    """
    return torch.load(
        open(get_path("embeddings", "{}.pt".format(embeddings_name)), "rb"), map_location=get_map_location()
    )


def get_top_k(query_embedding, queried_embeddings, k, distance):
    """
    Returns the distances and indices of the k nearest embeddings in the `queried_embeddings` tensor to the
    `query_embedding` tensor.
    :param query_embedding: tensor with the embedding of the query image.
    :type: torch.Tensor with shape [1, embedding_length]
    :param queried_embeddings: tensor with the stacked embeddings of the queried dataset.
    :type: torch.Tensor with shape [queried_dataset_size, embedding_length]
    :param k: the number of most similar images to be returned.
    :type: int
    :param distance: which distance function to be used for nearest neighbor computation. Either 'cosine' or 'pairwise'
    :type: str, either 'cosine' or 'pairwise'
    :return: the closest k embeddings in the `embeddings` tensor to the `query_embedding`. A 2-tuple with shape `[k]`
    tensor with their distances and indices are returned (respectively).
    :type: Tuple[torch.Tensor, torch.Tensor]
    """
    p_dist = PairwiseDistance(p=2) if distance == "pairwise" else CosineSimilarity()
    distances = p_dist(queried_embeddings, query_embedding)
    return torch.topk(distances, k)  # return the top k results


def get_top_k_indices(query_embedding, queried_embeddings, k, distance):
    """
    Simple wrapper over `get_top_k` that returns only the index list. This is needed to be able to do the parallel
    starmap in `average_class_recall_parallel`, as what is returned by the `get_top_k` function cannot be pickled.
    :param query_embedding: tensor with the embedding of the query image.
    :type: torch.Tensor with shape [1, embedding_length]
    :param queried_embeddings: tensor with the stacked embeddings of the queried dataset.
    :type: torch.Tensor with shape [queried_dataset_size, embedding_length]
    :param k: the number of most similar images to be returned.
    :type: int
    :param distance: which distance function to be used for nearest neighbor computation. Either 'cosine' or 'pairwise'
    :type: str, either 'cosine' or 'pairwise'
    :return: the indices of the closest k embeddings in the `embeddings` tensor to the `query_embedding`.
    :type: torch.Tensor of shape [k]
    """
    _, top_indices = get_top_k(query_embedding, queried_embeddings, k, distance)
    return top_indices


@log_time
def retrieve_top_k(
    model,
    query_image_filename,
    query_dataset_name,
    queried_dataset_name,
    queried_embeddings_name,
    k=16,
    distance="cosine",
    n_gpu=0,
):
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
    queried_embeddings = load_embeddings(queried_embeddings_name).to(device)
    # Get the query image and create the embedding for it
    image, image_class = query_dataset[query_image_filename]
    # Send elements to the specified device
    image, model = [var.to(device) for var in [image, model]]
    query_embedding = model(image.unsqueeze(0))  # unsqueeze to add the missing dimension expected by the model
    # Compute the distance to the query embedding for all images in the Dataset
    queried_embeddings, query_embedding = [var.to(device) for var in [queried_embeddings, query_embedding]]
    top_distances, top_indices = get_top_k(query_embedding, queried_embeddings, k, distance)
    plot_image_retrieval(image, image_class, query_dataset, queried_dataset, top_distances, top_indices)


@log_time
def average_class_recall(
    query_dataset, queried_dataset, query_embeddings, queried_embeddings, k, distance="cosine", n_gpu=0
):
    """
    Computes the average class recall for the given embeddings. Embeddings are split into "test" and "queried" subsets.
    For each embedding in the "test" set, the nearest `k` embeddings in the "queried" set are retrieved using the given
    distance metric, giving us a recall for each test embedding that corresponds to the percentage of the top `k`
    retrieved embeddings that correspond to the same class as the test image. The average class recall is thus the
    average recall over all test images.
    :param query_dataset: the dataset to which the query image belongs.
    :type: torch.utils.data.Dataset
    :param queried_dataset: the dataset to be queried. It must have the exact same classes as the query dataset.
    :type: torch.utils.data.Dataset
    :param query_embeddings: the embedding tensors that will be used to retrieve images in the queried dataset.
    :type: torch.Tensor of shape [query_dataset_length, embedding_length]
    :param queried_embeddings: the embedding tensors that will be retrieved to compute the average class recall.
    :type: torch.Tensor of shape [queried_dataset_length, embedding_length]
    :param k: the number of most similar images to be retrieved in order to compute the recall.
    :type: int
    :param distance: which distance function to be used for nearest neighbor computation. Either 'cosine' or 'pairwise'
    :type: str, either 'cosine' or 'pairwise'
    :param n_gpu: number of available GPUs. If zero, the CPU will be used.
    :type: int
    """
    device = get_device(n_gpu)
    query_embeddings, queried_embeddings = query_embeddings.to(device), queried_embeddings.to(device)
    recalls = []
    for i, query_embedding in tqdm(
        enumerate(query_embeddings), total=query_embeddings.shape[0], desc="Computing recall"
    ):
        top_indices = get_top_k_indices(query_embedding.unsqueeze(0), queried_embeddings, k, distance)
        recalls.append(sum([queried_dataset[j][1] == query_dataset[i][1] for j in top_indices]) / k)
    print("Average class recall: {0:.2f}%".format(mean(recalls) * 100))


def class_recall(queried_dataset, top_indices, query_image_class):
    """
    Computes the recall for the results of the given top distances query.
    :param queried_dataset: the dataset that was queried. It must have the exact same classes as the query dataset.
    It is used to retrieve the classes for the given `top_indices`.
    :type: torch.utils.data.Dataset
    :param top_indices: a list with the indices, in the `queried_dataset`, of the most similar images to the
    query image.
    :type: List[int]
    :param query_image_class: the class index (not name) of the query image.
    :type: int
    :return: the class recall, which corresponds to the percentage of the retrieved results (the given most similar
    images to the query image) have the same class as the query image
    :type: float
    """
    # noinspection PyTypeChecker
    return reduce(lambda part, i: part + queried_dataset[i][1] == query_image_class, top_indices, 0) / len(top_indices)


@log_time
def average_class_recall_parallel(
    query_dataset, queried_dataset, query_embeddings, queried_embeddings, k, distance="cosine", n_gpu=0, processes=None
):
    """
    Query the embeddings for a dataset with the given image. The image is embedded with the given model. Pairwise
    distances to the query image are computed for each embedding in the dataset, so the embeddings created by `model`
    must have the same length as the ones in the embedding directory. The `k` most similar images are displayed.
    Computed in parallel.
    :param query_dataset: the dataset to which the query image belongs.
    :type: torch.utils.data.Dataset
    :param queried_dataset: the dataset to be queried. It must have the exact same classes as the query dataset.
    :type: torch.utils.data.Dataset
    :param query_embeddings: the embedding tensors that will be used to retrieve images in the queried dataset.
    :type: torch.Tensor of shape [query_dataset_length, embedding_length]
    :param queried_embeddings: the embedding tensors that will be retrieved to compute the average class recall.
    :type: torch.Tensor of shape [queried_dataset_length, embedding_length]
    :param k: the number of most similar images to be retrieved in order to compute the recall.
    :type: int
    :param distance: which distance function to be used for nearest neighbor computation. Either 'cosine' or 'pairwise'
    :type: str, either 'cosine' or 'pairwise'
    :param n_gpu: number of available GPUs. If zero, the CPU will be used.
    :type: int
    :param processes: number of parallel workers to use. If `None`, then `os.cpu_count()` will be used.
    :type: int
    """
    device = get_device(n_gpu)
    query_embeddings, queried_embeddings = query_embeddings.to(device), queried_embeddings.to(device)
    with Pool(processes=processes) as pool:
        top_indices_per_query = pool.starmap(  # `multiprocessing` lib uses pickle as serializer backend, so the mapped
            get_top_k_indices,  # func. must be declared in global scope. We use `starmap` to pass multiple parameters to
            tqdm(  #  that func. `repeat` gives iterator that returns given param, so we can pass them without copying
                zip(query_embeddings.unsqueeze(1), *[repeat(param) for param in [queried_embeddings, k, distance]]),
                total=query_embeddings.shape[0],
                desc="Computing distances",
            ),
        )  # `total` required for `__len__` attribute
        recalls = pool.starmap(
            class_recall,
            tqdm(
                zip(repeat(queried_dataset), top_indices_per_query, query_dataset.targets),
                total=query_embeddings.shape[0],
                desc="Computing recall",
            ),
        )
        print("Average class recall: {0:.2f}%".format(mean(recalls) * 100))
