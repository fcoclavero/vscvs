__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite metrics for triplet networks. """


import torch

from abc import ABC
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class TripletMetric(Metric, ABC):
    """
    Base class for ignite metrics for triplet networks. They receive batch triplet embeddings: either
    `(anchor_embeddings, positive_embeddings, negative_embeddings)` or `{'anchor_embeddings': anchor_embeddings,
    'positive_embeddings': positive_embeddings, 'negative_embeddings': negative_embeddings}`. This makes more sense in
    this setting than the default `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}` input (useful in supervised training).
    """
    def __init__(self, output_transform=lambda x: x, device=None):
        """
        Class constructor.
        :param output_transform: a callable that is used to transform the output of the `process_function` of an
        `ignite.engine.Engine` output into the form expected by the metric. In the case of triplet metrics, this is a
        3-tuple with the triplet embeddings
        :type: Callable<args: `output`, ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>> (optional)
        :param device: device type specification.
        :type: str of torch.device (optional) (default: None)
        """
        super().__init__(output_transform=output_transform, device=device)


class Accuracy(TripletMetric):
    """
    Computes the average accuracy for a triplet network, defined as the proportion of triplets in which the positive
    embedding is closer to the anchor than the negative embedding (this is the desired behaviour).
    """
    def __init__(self, *args, **kwargs):
        """
        :param args: TripletMetric arguments
        :type: tuple
        :param kwargs: TripletMetric keyword arguments
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
            raise NotComputableError('TripletAccuracy must have at least one example before it can be computed.')

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


class Loss(TripletMetric):
    """
    Computes the average loss for a triplet network.
    """
    def __init__(self, loss_fn, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: TripletMetric arguments
        :type: tuple
        :param loss_fn: callable taking triplet embeddings and optionally other arguments, and returns the average loss
        over all observations in the batch.
        :type: Callable<args: `anchor_embeddings`, `positive_embeddings`, `negative_embeddings`, ret: float>
        :param batch_size: callable taking a target tensor returns the first dimension size (usually the batch size).
        :type: Callable<args: 'tensor', ret: int>
        :param kwargs: TripletMetric keyword arguments
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


class AverageDistances(TripletMetric):
    """
    Computes the average distances from the anchor to the positive and negative elements of each triplet.
    This is a utility class to define the `TripletAveragePositiveDistance` and `TripletAverageNegativeDistance` metrics.
    As this class returns a tuple, it will cause a logging error if used directly on a Trainer.
    """
    def __init__(self, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: TripletMetric arguments
        :type: tuple
        :param batch_size: callable taking a target tensor returns the first dimension size (usually the batch size).
        :type: Callable<args: 'tensor', ret: int>
        :param kwargs: TripletMetric keyword arguments
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
        Computes the average distance  from the anchor to the positive and negative elements of each triplet, based on
        the accumulated state. This is called at the end of each epoch.
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
        :param output: the output of the engine's process function, using the triplet format: 3-tuple with triplet
        elements' embeddings.
        :type: tuple<torch.Tensor, torch.Tensor, torch.Tensor>
        """
        anchor_embeddings, positive_embeddings, negative_embeddings = output
        batch_size = self._batch_size(anchor_embeddings)
        self._sum_positive = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings).pow(2)
        self._sum_negative = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings).pow(2)
        self._num_examples_negative += batch_size
        self._num_examples_positive += batch_size
