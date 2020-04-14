__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" GAN model definition. """


import torch.nn as nn


class GAN(nn.Module):
    """
    Pytorch module for a GAN.
    A GAN produces similar embeddings for similar inputs by training using a contrastive loss function that
    has penalizes similar embeddings for two images of different classes, as well as dissimilar embeddings for two
    images of the same class.
    """
    def __init__(self, discriminator_network, generator_network):
        """
        Model constructor.
        :param discriminator_network: the network that will encode the first element of each sample pair.
        :type: torch.nn.module
        :param generator_network: the network that will encode the first element of each sample pair.
        :type: torch.nn.module
        """
        super().__init__()
        self.discriminator_network = discriminator_network
        self.generator_network = generator_network

    def forward(self, input):
        """
        Perform a forward pass on the network, computing the embeddings for both inputs.
        :param input: the generator_network network input
        :type: torch.Tensor with a size compatible with `generator_network`
        :return: the embeddings for both inputs
        :type: torch.Tensor, torch.Tensor
        """
        return self.discriminator_network(self.generator_network(input))


class CVSGAN(GAN):
    """
    Pytorch module for a CVS GAN.

    This network uses an adversarial approach for generating a common vector space between two different data modes, for
    example photos and sketches, or images and text.


    """
    def __init__(self, discriminator_network, *mode_generator_networks):
        """
        Model constructor.
        :param embedding_network: the network to be used in the siamese training. It must accept network inputs and
        produce network outputs.
        :type: torch.nn.module
        """
        super().__init__()
        self.discriminator_network = discriminator_network

    def forward(self, *mode_inputs):
        """
        Perform a forward pass on the network, computing the embeddings for both inputs.
        :param mode_inputs: list with the input for each of the mode embedding networks. The inputs of each mode must be
        provided in the same mode order as the mode embedding networks were declared.
        :type: list<torch.Tensor> with tensors of sizes compatible with each corresponding mode embedding network.
        :return: the embeddings for both inputs
        :type: torch.Tensor, torch.Tensor
        """
        embedding_0 = self.embedding_network(input_0)
        embedding_1 = self.embedding_network(input_1)
        return embedding_0, embedding_1
