__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Triplet loss module for triplet networks. """


import torch

from .mixins import ReductionMixin


class TripletLoss(ReductionMixin, torch.nn.Module):
    """
    Triplet loss function for triplet networks.
    """
    def __init__(self, *args, margin=.2, **kwargs):
        """
        :param args: mixin arguments
        :type: list
        :param margin: parameter defining the minimum acceptable difference between the distance from the anchor element
        to the negative, and the distance from the anchor to the negative.
        :type: float
        :param kwargs: mixin keyword arguments
        :type: dict
        """
        self.margin = margin
        super().__init__(*args, **kwargs)

    def forward(self, anchor, positive, negative):
        """
        Compute the Triplet Loss between a batch of triplets. The Triplet Loss is defined as:

        $L_t(x_a, x_p, x_n) = \frac{1}{2}\{m + D^2(x_a, x_p) - D^2(x_a, x_n)\}_{+}$

        where $x_a$, $x_p$ and $x_n$ are the anchor, positive and negative embeddings, respectively. The margin $m$ is
        the parameter defining the minimum acceptable difference between the distance from the anchor element to the
        negative, and the distance from the anchor to the negative.

        The objective of the loss function is that the distance between the representations of the anchor and the
        negative is greater (and bigger than the margin) than the distance between the anchor and the positive.

        Three kinds of triplets can be distinguished:

        1. Easy triplets: $D^2(x_a, x_n) > D^2(x_a, x_p) + m $, where the negative sample is already sufficiently
        distant to the anchor sample respect to the positive sample in the embedding space. The loss is 0 and the
        network parameters are not updated.
        2. Hard triplets: $D^2(x_a, x_n) < D^2(x_a, x_p)$, where the negative sample is closer to the anchor than the
        positive. The loss is positive (and greater than $m$).
        3. Semi-hard triplets: $D^2(x_a, x_p) < D^2(x_a, x_n) < D^2(x_a, x_p) + m $, where the negative sample is more
        distant to the anchor than the positive, but the distance is not greater than the margin, so the loss is still
        positive (and smaller than $m$).

        Good triplet selection (sufficient hard triplets, and few easy triplets) is essential for better training
        and model performance. [See.](https://gombru.github.io/2019/04/03/ranking_loss/)

        As seen in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0097849317302194)
        :param anchor: the anchor embeddings which will be compared with the positive and negative embeddings.
        :type: torch.Tensor
        :param positive: the positive embeddings.
        :type: torch.Tensor
        :param negative: the negative embeddings
        :type: torch.Tensor
        """
        distance_to_positive = torch.nn.functional.pairwise_distance(anchor, positive).pow(2)
        distance_to_negative = torch.nn.functional.pairwise_distance(anchor, negative).pow(2)
        losses = 0.5 * torch.clamp(self.margin + distance_to_positive - distance_to_negative, min=0.0)
        return self.reduce(losses)
