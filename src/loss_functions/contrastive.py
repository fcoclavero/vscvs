__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Contrastive loss module for siamese networks. """


import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function for siamese networks.
    """
    def __init__(self, margin=.2):
        """
        Loss constructor
        :param margin: defines an acceptable threshold for two embeddings to be considered as dissimilar.
        :type: float
        """
        self.margin = margin
        super().__init__()

    def forward(self, x_0, x_1, y):
        """
        Compute the Contrastive Loss between two embeddings, given the label indicating whether the two embeddings
        belong to the same class. The Contrastive Loss is defined as:

        $L_c(x_0, x_1, y) = \frac{1}{2}(1-y)D^2(x_0,x_1) + \frac{1}{2}y\{m-D^2(x_0,x_1)\}_{+}$

        where $x_0$ and $x_1$ are the two embeddings, $y$ is the training pair label, such that

        $y=\left\{\begin{array}{ll}0  & \text{if } (x_0,x_1) \text{ are similar} \\ 1 & \text{if } (x_0,x_1)
        \text{ are dissimilar} \end{array} \right.$

        $\{.\}$ is the hinge loss function, and $m$ is the parameter defining an acceptable threshold for $x_0$ and
        $x_1$ to be considered as dissimilar.

        As seen in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0097849317302194)
        :param x_0: the first embedding.
        :type: torch.Tensor
        :param x_1: the second embedding.
        :type: torch.Tensor
        :param y: the training pair label indicating the similarity between `x_0` and `x_1`.
        :return: the Contrastive Loss between `x_0` and `x_1`.
        :type: float
        """
        euclidean_distance = torch.nn.functional.pairwise_distance(x_0, x_1) # cross-domain euclidian distance
        return torch.mean(
            0.5 * (1 - y) * torch.pow(euclidean_distance, 2) +
            0.5 * y * torch.clamp(self.margin -  torch.pow(euclidean_distance, 2), min=0.0)
        )
