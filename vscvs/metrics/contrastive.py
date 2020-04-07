__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite metrics for siamese networks. """


import torch

from abc import ABC
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class SiameseMetric(Metric, ABC):
    """
    Base class for ignite metrics for siamese networks. They receive batch pair embeddings and their target tensor,
    indicating whether each pair is similar (0) or dissimilar (1): either `(embeddings_0, embeddings_1, target)` or
    `{'embeddings_0': embeddings_0, 'embeddings_1': embeddings_1, 'target': target}`. This makes more sense in this
    setting than the default `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}` input (useful in supervised training).
    """
    def __init__(self, output_transform=lambda x: x, device=None):
        """
        Class constructor.
        :param output_transform: a callable that is used to transform the output of the `process_function` of an
        `ignite.engine.Engine` output into the form expected by the metric. In the case of siamese metrics, this is a
        3-tuple with the first pair elements' embeddings, the second pair elements' embeddings, and the targets.
        :type: Callable<args: `output`, ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
        :param device: device type specification.
        :type: str of torch.device (optional) (default: None)
        """
        super().__init__(output_transform=output_transform, device=device)


class SiameseAccuracy(SiameseMetric):
    """
    Computes the average accuracy for a siamese network, defined as the accuracy of the best distance decision
    threshold for classifying batch elements into their targets (similar/dissimilar).
    """
    def __init__(self, *args, **kwargs):
        """
        :param args: SiameseMetric arguments
        :type: tuple
        :param kwargs: SiameseMetric keyword arguments
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self._num_correct = 0
        self._num_examples = 0

    @sync_all_reduce('_num_correct', '_num_examples')
    def compute(self):
        """
        Computes the average accuracy based on it's accumulated state. This is called at the end of each epoch.
        :return: the actual average accuracy
        :type: float
        :raises NotComputableError: when the metric cannot be computed
        """
        if self._num_examples == 0:
            raise NotComputableError('SiameseAccuracy must have at least one example before it can be computed.')

        return self._num_correct / self._num_examples

    @reinit__is_reduced
    def reset(self):
        """
        Resets the metric to it's initial state. This is called at the start of each epoch.
        """
        self._num_correct = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        """
        Updates the metric's state using the passed batch output. This is called once for each batch.
        :param output: the output of the engine's process function, using the siamese format: 3-tuple with the
        first pair elements' embeddings, the second pair elements' embeddings, and the targets.
        :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
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
        distance_threshold_decisions = torch.cat([ # distance) should classify all elements as dissimilar [similar].
            torch.zeros([1, batch_size]), # minimum decision threshold, all zeros
            torch.tril(torch.ones(batch_size, batch_size))]) # progressively increase threshold
        # Repeat sorted target tensor to do the `==` operation in parallel
        sorted_target_repeat = sorted_target.repeat(batch_size + 1).view(batch_size + 1, batch_size)
        # Do `==` to find where classes match and reduce to obtain the matches for each threshold
        matching_classes = (sorted_target_repeat == distance_threshold_decisions).sum(dim=0)
        # Find the threshold with the best classification accuracy (the threshold with most matches) and update fields
        self._num_correct += int(matching_classes[torch.argmax(matching_classes)])
        self._num_examples += batch_size


class SiameseLoss(SiameseMetric):
    """
    Computes the average loss for a Siamese network.
    """
    def __init__(self, loss_fn, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: SiameseMetric arguments
        :type: tuple
        :param loss_fn: callable taking image pair embeddings and their target tensor, optionally other arguments, and
        returns the average loss over all observations in the batch.
        :type: Callable<args: `embeddings_0`, `embeddings_1`, `target`, ret: float>
        :param batch_size: callable taking a target tensor returns the first dimension size (usually the batch size).
        :type: Callable<args: 'tensor', ret: int>
        :param kwargs: SiameseMetric keyword arguments
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self._loss_fn = loss_fn
        self._batch_size = batch_size
        self._sum = 0
        self._num_examples = 0

    @sync_all_reduce('_sum', '_num_examples')
    def compute(self):
        """
        Computes the average loss based on it's accumulated state. This is called at the end of each epoch.
        :return: the actual average loss
        :type: float
        :raises NotComputableError: when the metric cannot be computed
        """
        if self._num_examples == 0:
            raise NotComputableError('SiameseLoss must have at least one example before it can be computed.')

        return self._sum / self._num_examples

    @reinit__is_reduced
    def reset(self):
        """
        Resets the metric to it's initial state. This is called at the start of each epoch.
        """
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        """
        Updates the metric's state using the passed batch output. This is called once for each batch.
        :param output: the output of the engine's process function, using the siamese format: 3-tuple with the
        first pair elements' embeddings, the second pair elements' embeddings, and the targets.
        :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
        :raises ValueError: when loss function cannot be computed
        """
        if len(output) == 3:
            kwargs = {}
            embeddings_0, embeddings_1, target = output
        else:
            embeddings_0, embeddings_1, target, kwargs = output

        average_loss = self._loss_fn(embeddings_0, embeddings_1, target, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        batch_size = self._batch_size(target)
        self._sum += average_loss.item() * batch_size
        self._num_examples += batch_size


class SiameseAverageDistances(SiameseMetric):
    """
    Computes the average distances for positive and negative pairs in a Siamese network.
    This is a utility class to define the `SiameseAveragePositiveDistance` and `SiameseAverageNegativeDistance` metrics.
    As this class returns a tuple, it will cause a logging error if used directly on a Trainer.
    """
    def __init__(self, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: SiameseMetric arguments
        :type: tuple
        :param batch_size: callable taking a target tensor returns the first dimension size (usually the batch size).
        :type: Callable<args: 'tensor', ret: int>
        :param kwargs: SiameseMetric keyword arguments
        :type: dict
        """
        super().__init__(*args, **kwargs)
        self._batch_size = batch_size
        self._sum_positive = 0
        self._sum_negative = 0
        self._num_examples_positive = 0
        self._num_examples_negative = 0

    @sync_all_reduce('_sum_positive', '_num_examples_positive', '_sum_negative', '_num_examples_negative')
    def compute(self):
        """
        Computes the average distance between positive and negative pairs, separately, based on the accumulated state.
        This is called at the end of each epoch.
        :return: the actual average distances
        :type: tuple<float>
        :raises NotComputableError: when the metric cannot be computed
        """
        if self._num_examples_positive == 0 or self._num_examples_negative == 0:
            raise NotComputableError('SiameseAverageDistances needs at least one example per target to be computed.')

        return self._sum_positive / self._num_examples_positive, self._sum_negative / self._num_examples_negative

    @reinit__is_reduced
    def reset(self):
        """
        Resets the metric to it's initial state. This is called at the start of each epoch.
        """
        self._sum_positive = 0
        self._sum_negative = 0
        self._num_examples_positive = 0
        self._num_examples_negative = 0

    @reinit__is_reduced
    def update(self, output):
        """
        Updates the metric's state using the passed batch output. This is called once for each batch.
        :param output: the output of the engine's process function, using the siamese format: 3-tuple with the
        first pair elements' embeddings, the second pair elements' embeddings, and the targets.
        :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
        """
        embeddings_0, embeddings_1, target = output
        batch_size = self._batch_size(target)
        euclidean_distances_squared = torch.nn.functional.pairwise_distance(embeddings_0, embeddings_1).pow(2)
        batch_sum_negative = (euclidean_distances_squared * target).sum() # target 1 means negative pair
        self._sum_negative += batch_sum_negative
        self._sum_positive += euclidean_distances_squared.sum() - batch_sum_negative
        self._num_examples_negative += target.sum()
        self._num_examples_positive += batch_size - target.sum()
