__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite metrics for triplet networks. """


import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Loss, Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from overrides import overrides

from vscvs.metrics.mulitmodal import AbstractAverageDistances


class AccuracyTriplets(Metric):
    """
    Computes the average accuracy for a triplet network, defined as the proportion of triplets in which the positive
    embedding is closer to the anchor than the negative embedding (this is the desired behaviour).
    """
    def __init__(self, *args, **kwargs):
        """
        :param args: Metric arguments
        :type: tuple
        :param kwargs: Metric keyword arguments
        :type: dict
        """
        self._num_correct = 0
        self._num_examples = 0
        super().__init__(*args, **kwargs)

    @sync_all_reduce('_num_correct', '_num_examples')
    @overrides
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed.')
        return self._num_correct / self._num_examples

    @reinit__is_reduced
    @overrides
    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using the passed triplet batch output.
        :param output: the output of the engine's process function, using the triplet format: 3-tuple with the
        triplet elements' embeddings.
        :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
        :raises ValueError: when loss function cannot be computed
        """
        anchor_embeddings, positive_embeddings, negative_embeddings = output
        batch_size = anchor_embeddings.shape[0]
        # Compute distances between the anchors and the positives and negatives, which should be at a greater distance.
        positive_distances = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings).pow(2)
        negative_distances = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings).pow(2)
        # Find the triplets where the desirable condition is met, that `positive_distances[i] < negative_distances[i]`
        condition = positive_distances < negative_distances
        # Update metric fields
        self._num_correct += condition.sum()
        self._num_examples += batch_size


class LossTriplets(Loss):
    """
    Computes the average loss for a triplet network.
    """
    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using the passed triplet batch output.
        :param output: the output of the engine's process function, using the triplet format: 3-tuple with the
        triplet elements' embeddings.
        :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
        :raises ValueError: when loss function cannot be computed
        """
        if len(output) == 3:
            kwargs = {}
            anchor_embeddings, positive_embeddings, negative_embeddings = output
        else:
            anchor_embeddings, positive_embeddings, negative_embeddings, kwargs = output

        average_loss = self._loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        batch_size = self._batch_size(anchor_embeddings)
        self._sum += average_loss.item() * batch_size
        self._num_examples += batch_size


class AverageDistancesTriplets(AbstractAverageDistances):
    """
    Computes the average distances from the anchor to the positive and negative elements of each triplet.
    """
    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using the passed triplet batch output.
        :param output: the output of the engine's process function, using the triplet format: 3-tuple with triplet
        elements' embeddings.
        :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
        """
        anchor_embeddings, positive_embeddings, negative_embeddings = output
        batch_size = self._batch_size(anchor_embeddings)
        self._sum_positive_distances = \
            torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings).pow(2).sum()
        self._sum_negative_distances = \
            torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings).pow(2).sum()
        self._num_examples_negative += batch_size
        self._num_examples_positive += batch_size
