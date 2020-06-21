__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Custom Ignite metrics for siamese networks. """


import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Loss
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.metric import sync_all_reduce
from overrides import overrides

from .mulitmodal import AbstractAverageDistances


class AccuracySiamesePairs(Metric):
    """
    Computes the average accuracy for a siamese network, defined as the accuracy of the best distance decision
    threshold for classifying batch elements into their targets (similar/dissimilar).
    """

    def __init__(self, *args, **kwargs):
        """
        :param args: Metric arguments
        :type: Tuple
        :param kwargs: Metric keyword arguments
        :type: Dict
        """
        self._num_correct = 0
        self._num_examples = 0
        super().__init__(*args, **kwargs)

    @sync_all_reduce("_num_correct", "_num_examples")
    @overrides
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
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
        :override: updates the metric's state using the passed siamese batch output.
        :param output: the output of the engine's process function, using the siamese format: 3-tuple with the
        first pair elements' embeddings, the second pair elements' embeddings, and the targets.
        :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises ValueError: when loss function cannot be computed
        """
        embeddings_0, embeddings_1, target = output
        batch_size = target.shape[0]
        # Compute distances between paired embeddings. Similar elements should be at a smaller distance.
        distances = torch.nn.functional.pairwise_distance(embeddings_0, embeddings_1).pow(2)
        # Get the indices of the sorted `distances` array to be able to sort the target tensor in that order.
        sorted_indices = distances.argsort(dim=-1)
        # We will now compare the actual similar/dissimilar labels of the sorted target tensor ...
        sorted_target = torch.tensor([target[i] for i in sorted_indices])
        # ... with a threshold-based classification at every possible threshold (using the existing distances). The
        # lowest [largest] possible threshold (a distance smaller [larger] than the smallest [largest] pairwise
        distance_threshold_decisions = torch.cat(
            [  # distance) should classify all elements as dissimilar [similar].
                torch.zeros([1, batch_size]),  # minimum decision threshold, all zeros
                torch.tril(torch.ones(batch_size, batch_size)),
            ]
        )  # progressively increase threshold
        # Repeat sorted target tensor to do the `==` operation in parallel
        sorted_target_repeat = sorted_target.repeat(batch_size + 1).view(batch_size + 1, batch_size)
        # Do `==` to find where classes match and reduce to obtain the matches for each threshold
        matching_classes = (sorted_target_repeat == distance_threshold_decisions).sum(dim=0)
        # Find the threshold with the best classification accuracy (the threshold with most matches) and update fields
        self._num_correct += int(matching_classes[torch.argmax(matching_classes)])
        self._num_examples += batch_size


# noinspection DuplicatedCode
class LossSiamesePairs(Loss):
    """
    Computes the average loss for a siamese network.
    """

    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using the passed siamese batch output.
        :param output: the output of the engine's process function, using the siamese format: 3-tuple with the
        first pair elements' embeddings, the second pair elements' embeddings, and the targets.
        :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises ValueError: when loss function cannot be computed
        """
        if len(output) == 3:
            kwargs = {}
            embeddings_0, embeddings_1, target = output
        else:
            embeddings_0, embeddings_1, target, kwargs = output

        average_loss = self._loss_fn(embeddings_0, embeddings_1, target, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        batch_size = self._batch_size(target)
        self._sum += average_loss.item() * batch_size
        self._num_examples += batch_size


class AverageDistancesSiamesePairs(AbstractAverageDistances):
    """
    Computes the average distances for positive and negative pairs in a siamese network.
    """

    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using the passed siamese batch output.
        :param output: the output of the engine's process function, using the siamese format: 3-tuple with the
        first pair elements' embeddings, the second pair elements' embeddings, and the targets.
        :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        embeddings_0, embeddings_1, target = output
        batch_size = self._batch_size(target)
        euclidean_distances_squared = torch.nn.functional.pairwise_distance(embeddings_0, embeddings_1).pow(2)
        batch_sum_negative = (euclidean_distances_squared * target).sum()  # target 1 means negative pair
        self._sum_negative_distances += batch_sum_negative
        self._sum_positive_distances += euclidean_distances_squared.sum() - batch_sum_negative
        self._num_examples_negative += target.sum()
        self._num_examples_positive += batch_size - target.sum()
