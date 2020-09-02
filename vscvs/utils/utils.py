__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" General utilities. All functions are imported in the `utils.__init__.py` for ease of access. """


import os
import re

from collections import OrderedDict
from datetime import datetime

import yaml

import torch

from settings import CHECKPOINT_NAME_FORMAT
from torch import nn
from torch.utils.data import DataLoader

from .path import get_checkpoint_path


def camel_to_snake_case(camel_cased_string):
    """
    Convert the format of the given string from CamelCase to snake_case.
    :param camel_cased_string: the string in CamelCase format.
    :type: str
    :return: the same string, but in snake_case format.
    :type: str
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_cased_string).lower()


def camel_to_snake_case_dict_keys(dictionary):
    """
    Convert the format of the given dictionary keys from CamelCase to snake_case.
    :param dictionary: the dictionary with keys in CamelCase format.
    :type: Dict
    :return: the same dictionary, but with its keys formatted in snake_case.
    :type: Dict
    """
    return {camel_to_snake_case(key): value for key, value in dictionary.items()}


def get_device(n_gpu):
    """
    Returns the name of the PyTorch device to be used, based on the number of available gpus.
    :param n_gpu: number of GPUs available. Use 0 for CPU mode.
    :type: int
    :return: the name of the device to be used by PyTorch.
    :type: str
    """
    return torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")


def get_map_location():
    """
    Returns a `torch.device` object or string containing a device tag that can be provided to
    the PyTorch serialization functions, based on cuda availability.
    :return: a map location consistent with the run environment.
    :type: Union[torch.device, str]
    """
    return "cpu" if not torch.cuda.is_available() else lambda storage, loc: storage.cuda()


def get_out_features_from_model(model):
    """
    Return the number of features from the last layer of a PyTorch model.
    :param model: the model.
    :type: torch.nn.Module
    :return: the number of features of the last layer of `model`.
    :type: int
    """
    return get_out_features_from_state_dict(model.state_dict())


def get_out_features_from_state_dict(state_dict):
    """
    Return the number of features from the last layer of a state_dict.
    :param state_dict: the state_dict of a PyTorch model.
    :type: OrderedDict
    :return: the number of features of the last layer of `state_dict`.
    :type: int
    """
    return next(reversed(state_dict.values())).shape[
        0
    ]  # OrderedDict guarantees last elem. in values list is last layer


def initialize_weights(
    model, conv_mean=0.2, conv_std=0.0, batch_norm_mean=0.2, batch_norm_std=1.0, batch_norm_bias=0.0
):
    """
    Custom weights initialization.
    The function takes an initialized model as input and re-initializes all convolutional,
    convolutional-transpose, and batch normalization layer weights randomly from a Normal distribution.
    The function should be applied to models immediately after initialization.
    Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    :param model: the model to be initialized
    :type: torch.nn.Module
    :param conv_mean: the mean for the Normal distribution for the convolutional layer weights
    :type: float
    :param conv_std: the standard deviation for the Normal distribution for the convolutional layer weights
    :type: float
    :param batch_norm_mean: the mean for the Normal distribution for the batch normalization layer weights
    :type: float
    :param batch_norm_std: the standard deviation for the Normal distribution for the batch normalization layer weights
    :type: float
    :param batch_norm_bias: constant initial value for the batch normalization bias layer weights
    :type: float
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, conv_mean, conv_std)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, batch_norm_mean, batch_norm_std)
        nn.init.constant_(model.bias.data, batch_norm_bias)


def load_state_dict_from_checkpoint(state_dict_file, checkpoint_name, date_string, *tags):
    """
    Load a state dictionary from its checkpoint file.
    :param state_dict_file: the name of the state_dict file.
    :type: str
    :param checkpoint_name: the name of the checkpoint directory.
    :type: str
    :param date_string: the checkpoint date in string format.
    :type: str
    :param tags: the checkpoint tags (subdirectories).
    :type: List[str]
    :return: the state dictionary.
    :type: OrderedDict
    """
    date = datetime.strptime(date_string, CHECKPOINT_NAME_FORMAT)
    checkpoint_directory = get_checkpoint_path(checkpoint_name, *tags, date=date)
    try:
        return torch.load(
            os.path.join(checkpoint_directory, "{}.pt".format(state_dict_file)), map_location=get_map_location()
        )
    except FileNotFoundError:
        return torch.load(
            os.path.join(checkpoint_directory, "{}.pth".format(state_dict_file)), map_location=get_map_location()
        )


def load_classification_model_from_checkpoint(model, state_dict_file, checkpoint_name, date_string, *tags):
    """
    Load a classification model from its state dictionary file.
    :param model: the model to be loaded.
    :type: torch.nn.Module
    :param state_dict_file: the name of the state_dict file.
    :type: str
    :param checkpoint_name: the name of the checkpoint directory.
    :type: str
    :param date_string: the checkpoint date in string format.
    :type: str
    :param tags: the checkpoint tags (subdirectories).
    :type: List[str]
    :return: the model, loaded with the state dictionary at the specified checkpoint.
    :type: torch.nn.Module
    """
    state_dict = load_state_dict_from_checkpoint(state_dict_file, checkpoint_name, date_string, *tags)
    out_features = get_out_features_from_state_dict(state_dict)
    model = model(out_features=out_features)
    model.load_state_dict(state_dict)
    return model


def load_siamese_model_from_checkpoint(model_0, model_1, state_dict_file, checkpoint_name, date_string, *tags):
    """
    Load a siamese model from its state dictionary file.
    :param model_0: the model of the first siamese branch.
    :type: torch.nn.Module
    :param model_1: the model of the second siamese branch.
    :type: torch.nn.Module
    :param state_dict_file: the name of the state_dict file.
    :type: str
    :param checkpoint_name: the name of the checkpoint directory.
    :type: str
    :param date_string: the checkpoint date in string format.
    :type: str
    :param tags: the checkpoint tags (subdirectories).
    :type: List[str]
    :return: the mode, loaded with the state dictionary at the specified checkpoint.
    :type: torch.nn.Module
    """
    from vscvs.models import SiameseNetwork

    state_dict = load_state_dict_from_checkpoint(state_dict_file, checkpoint_name, date_string, *tags)
    state_dict_0 = OrderedDict({key: value for key, value in state_dict.items() if "embedding_network_0" in key})
    state_dict_1 = OrderedDict({key: value for key, value in state_dict.items() if "embedding_network_1" in key})
    out_features_0 = get_out_features_from_state_dict(state_dict_0)
    out_features_1 = get_out_features_from_state_dict(state_dict_1)
    model = SiameseNetwork(
        embedding_network_0=model_0(out_features=out_features_0),
        embedding_network_1=model_1(out_features=out_features_1),
    )
    model.load_state_dict(state_dict)
    return model


def load_yaml(file_path):
    """
    Load a yaml file as a Python dictionary.
    :param file_path: the absolute or relative path to the yaml file.
    :type: str
    :return: the contents of the yaml file, as a Python dictionary.
    :type: Dict
    """
    return yaml.load(open(file_path, "r"), Loader=yaml.Loader)


def remove_last_layer(model):
    """
    Remove the last layer from a PyTorch model. This is useful for creating image embeddings from a classifier network.
    :param model: the PyTorch model to be modified.
    :type: torch.nn.module
    :return: the modified network, without the last layer.
    :type: torch.nn.module
    """
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def sprite_tensor(dataset_name, size=(64, 64)):
    """
    Create an image tensor that can be used to create a Tensorboard embedding projector sprite image for the dataset
    corresponding to `dataset_name`. This image tensor must have a shape of `(N, C, H, W)` where `N` is the length of
    the dataset, `C` is the number of color channels, and `H` and `W` are the height and width dimensions for the
    sprite components. Dataset elements must be reduced in size in order to fit in memory when using Tensorboard.
    :param dataset_name: the name of the dataset from which the sprite tensor will be created.
    :type: str
    :param size: the dimension of the resulting sprite image components. If `size` is a sequence like `(h, w)`, the
    output size will be matched to this. If size is an int, the smaller edge of the image will be matched to this
    number. i.e, if `height > width`, then image will be rescaled to `(size * height / width, size)`.
    :type: int
    :return: the sprite image tensor, with the dimensions specified above.
    :type: torch.Tensor
    """
    from vscvs.datasets import get_dataset

    dataset = get_dataset(dataset_name, normalize=False, size=size)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(data_loader))[0]


def str_to_bin_array(number, array_length=None):
    """
    Creates a binary array for the given number. If a length is specified, then the returned array will have the same
    add leading zeros to match `array_length`.
    :param number: the number to be represented as a binary array.
    :type: int
    :param array_length: (optional) the length of the binary array to be created.
    :type: int
    :return: the binary array representation of `number`.
    :type: List[int]
    """
    bin_str = "{0:b}".format(number)
    bin_str = bin_str.zfill(array_length) if array_length else bin_str
    return list(map(int, bin_str))
