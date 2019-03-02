import torch.nn as nn


def initialize_weights(model, conv_mean=0.2, conv_std=0.0, batch_norm_mean=0.2, batch_norm_std=1.0, batch_norm_bias=0.0):
    """
    Custom weights initialization.
    The function takes an initialized model as input and re-initializes all convolutional, convolutional-transpose,
    and batch normalization layer weights randomly from a Normal distribution. The function should be applied to
    models immediately after initialization.
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