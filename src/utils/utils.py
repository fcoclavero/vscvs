__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" General utilities. All functions are imported in the `utils.__init__.py` for ease of access. """


import os
import shutil
import torch
import yaml

from datetime import datetime
from torch import nn as nn

from settings import CHECKPOINT_NAME_FORMAT, ROOT_DIR


def get_device(n_gpu):
    """
    Returns the name of the PyTorch device to be used, based on the number of available gpus.
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :return: the name of the device to be used by PyTorch
    :type: str
    """
    return torch.device('cuda:0' if (torch.cuda.is_available() and n_gpu > 0) else 'cpu')


def get_checkpoint_directory(model_name, tag=None, date=datetime.now()):
    """
    Get the path where model checkpoints should be stored.
    :param model_name: the name of the model
    :type: str
    :param tag: optional tag for model checkpoint and tensorboard logs
    :type: str
    :param date: the date string of the model checkpoint. Defaults to the current date.
    :type: str
    :return: the model checkpoint path
    :type: str
    """
    return os.path.join(ROOT_DIR, 'data', 'checkpoints', model_name, tag or '', date.strftime(CHECKPOINT_NAME_FORMAT))


def get_log_directory(model_name, tag=None, date=datetime.now()):
    """
    Get the path where model checkpoints should be stored.
    :param model_name: the name of the model
    :type: str
    :param tag: optional tag for model checkpoint and tensorboard logs
    :type: str
    :param date: the date string of the model checkpoint. Defaults to the current date.
    :type: str
    :return: the model checkpoint path
    :type: str
    """
    return os.path.join(ROOT_DIR, 'data', 'logs', model_name, tag or '', date.strftime(CHECKPOINT_NAME_FORMAT))


def get_out_features_from_model(model):
    """
    Return the number of features from the last layer of a PyTorch model.
    :param model: the model
    :type: torch.nn.Module
    :return: the number of features of the last layer of `model`
    :type: int
    """
    return get_out_features_from_state_dict(model.state_dict())


def get_out_features_from_state_dict(state_dict):
    """
    Return the number of features from the last layer of a state_dict.
    :param state_dict: the state_dict of a PyTorch model
    :type: OrderedDict
    :return: the number of features of the last layer of `state_dict`
    :type: int
    """
    return next(reversed(state_dict.values())).shape[0] # OrderedDict guarantees last elem. in values list is last layer


def get_subdirectories(path):
    """
    Get a list of all the child directories of the given path.
    :param path: the path who's child directories are to be returned
    :type: str
    :return: the paths of the child directories, relative to the given path
    :type: list<str>
    """
    return next(os.walk(path))[1]


def initialize_weights(model, conv_mean=0.2, conv_std=0.0, batch_norm_mean=0.2,
                       batch_norm_std=1.0, batch_norm_bias=0.0):
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
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, conv_mean, conv_std)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, batch_norm_mean, batch_norm_std)
        nn.init.constant_(model.bias.data, batch_norm_bias)


def load_yaml(file_path):
    """
    Load a yaml file as a Python dictionary.
    :param file_path: the absolute or relative path to the yaml file.
    :type: str
    :return: the contents of the yaml file, as a Python dictionary.
    :type: dict
    """
    return yaml.load(open(file_path, 'r'), Loader=yaml.Loader)


def recreate_directory(directory_path):
    """
    Delete and recreate the directory at the given path to ensure an empty dir
    :param directory_path: the path to the directory to be recreated
    :type: str
    """
    shutil.rmtree(directory_path, ignore_errors=True)
    os.makedirs(directory_path)


def remove_last_layer(model):
    """
    Remove the last layer from a PyTorch model. This is useful for creating image embeddings from a classifier network.
    :param model: the PyTorch model to be modified
    :type: pytorch.nn.module
    :return: the modified network, without the last layer
    :type: pytorch.nn.module
    """
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def str_to_bin_array(number, array_length=None):
    """
    Creates a binary array for the given number. If a length is specified, then the returned array will have the same
    add leading zeros to match `array_length`.
    :param number: the number to be represented as a binary array
    :type: int
    :param array_length: (optional) the length of the binary array to be created.
    :type: int
    :return: the binary array representation of `number`
    :type: list<int either 0 or 1>
    """
    bin_str = '{0:b}'.format(number)
    bin_str = bin_str.zfill(array_length) if array_length else bin_str
    return list(map(int, bin_str))
