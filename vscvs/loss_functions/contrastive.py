__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Contrastive loss module for siamese networks. """


import torch

from .mixins import ReductionMixin


class ContrastiveLoss(ReductionMixin, torch.nn.Module):
    """
    Contrastive loss function for siamese networks.
    """
    def __init__(self, *args, margin=.2, **kwargs):
        """
        Loss constructor.
        :param args: mixin arguments
        :type: list
        :param margin: defines an acceptable threshold for two embeddings to be considered as dissimilar.
        :type: float
        :param kwargs: mixin keyword arguments
        :type: dict
        """
        self.margin = margin
        super().__init__(*args, **kwargs)

    def forward(self, x_0, x_1, y):
        """
        Compute the Contrastive Loss between two embeddings, given the label indicating whether the two embeddings
        belong to the same class. The Contrastive Loss is defined as:

        $L_c(x_0, x_1, y) = \frac{1}{2}(1 - y)D^2(x_0, x_1) + \frac{1}{2}y\{m - D^2(x_0, x_1)\}_{+}$

        where $x_0$ and $x_1$ are the two embeddings, $y$ is the training pair label, such that

        $y=\left\{\begin{array}{ll}0  & \text{if } (x_0, x_1) \text{ are similar} \\ 1 & \text{if } (x_0, x_1)
        \text{ are dissimilar} \end{array} \right.$

        $\{.\}$ is the hinge loss function, and $m$ is the parameter defining an acceptable threshold for $x_0$ and
        $x_1$ to be considered as dissimilar.

        Thus, the objective of the loss function is that the embeddings of positive pairs are close together in the
        embedding space, while the embeddings of negative pairs are at least as far as the margin.

        As seen in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0097849317302194)
        :param x_0: the first embeddings.
        :type: torch.Tensor
        :param x_1: the second embeddings.
        :type: torch.Tensor
        :param y: the training pair label indicating the similarity between `x_0` and `x_1`.
        :return: the Contrastive Loss between `x_0` and `x_1`.
        :type: float
        """
        euclidean_distances_squared = torch.nn.functional.pairwise_distance(x_0, x_1).pow(2) # cross-domain
        losses =  0.5 * ((1 - y) * euclidean_distances_squared +
                        y * torch.clamp(self.margin -  euclidean_distances_squared, min=0.0))
        return self.reduce(losses)
