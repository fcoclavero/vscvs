__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Generator modules for GAN modules. """


import torch.nn as nn


class MultimodalEncoder(nn.Module):
    """
    Pytorch module for multimodal encoder that aggregates the encoder modules for different data modes.
    The input for each encoder module must be passed in a tuple: input index `i` is passed on to encoder index `i`.
    Encoder outputs are returned as a list in the same order.
    """
    def __init__(self, *mode_embedding_networks):
        """
        Model constructor.
        :param mode_embedding_networks: the embedding networks for each mode.
        :type: list<torch.nn.Module>
        """
        super().__init__()
        self.mode_embedding_networks = mode_embedding_networks

    def forward(self, *mode_inputs):
        """
        Perform a forward pass on the network which computes the embeddings for each mode by performing a forward pass
        on every embedding network with the corresponding input.
        :param mode_inputs: list with the input for each of the mode embedding networks. The inputs of each mode must be
        provided in the same mode order as the mode embedding networks were declared.
        :type: list<torch.Tensor> with tensors of sizes compatible with each corresponding mode embedding network.
        :return: the embeddings for every input
        :type: list<torch.Tensor>
        """
        return [embedding_network(i) for embedding_network, i in zip(self.mode_embedding_networks, mode_inputs)]


class MultimodalEncoderShared(nn.Module):
    """
    Pytorch module for multimodal encoder that encodes the inputs using the same network, sharing weights.
    An input tuple is passed to the network. All inputs are encoded using the same network and weights All input tuple
    elements must have a shape compatible with the embedding network.
    """
    def __init__(self, mode_embedding_network):
        """
        Model constructor.
        :param mode_embedding_network: the embedding network for all modes.
        :type: torch.nn.Module
        """
        super().__init__()
        self.embedding_network = mode_embedding_network

    def forward(self, *mode_inputs):
        """
        Perform a forward pass on the network which computes the embeddings for each mode by performing a forward pass
        on every embedding network with the corresponding input.
        :param mode_inputs: list with the input for each of the mode embedding networks. The inputs of each mode must be
        provided in the same mode order as the mode embedding networks were declared.
        :type: list<torch.Tensor> with tensors of sizes compatible with each corresponding mode embedding network.
        :return: the embeddings for every input
        :type: list<torch.Tensor>
        """
        return [self.embedding_network(i) for i in mode_inputs]


class MultimodalEncoderCombined(nn.Module):
    """
    Pytorch module for multimodal encoder that encodes inputs using a first layer of individual mode encoding networks,
    followed by a shared wights encoder network.
    """
    def __init__(self, shared_embedding_network, *individual_embedding_networks):
        """
        Model constructor.
        :param shared_embedding_network: the embedding network that will be shared across modes. It will receive the
        outputs of each individual embedding network and produce a second embedding for each input.
        :type: torch.nn.Module
        :param individual_embedding_networks: the embedding networks for each individual modes. They receive the input
        in the format of each node and produce an output embedding compatible with the input of the shared embedding
        network.
        :type: list<torch.nn.Module>
        """
        super().__init__()
        self.individual_embedding_network = MultimodalEncoder(*individual_embedding_networks)
        self.shared_embedding_network = MultimodalEncoderShared(shared_embedding_network)

    def forward(self, *mode_inputs):
        """
        Perform a forward pass on the network which computes the embeddings for each mode by performing a forward pass
        on every embedding network with the corresponding input.
        :param mode_inputs: list with the input for each of the mode embedding networks. The inputs of each mode must be
        provided in the same mode order as the mode embedding networks were declared.
        :type: list<torch.Tensor> with tensors of sizes compatible with each corresponding mode embedding network.
        :return: the embeddings for every input
        :type: list<torch.Tensor>
        """
        return self.shared_embedding_network(self.individual_embedding_network(*mode_inputs))
