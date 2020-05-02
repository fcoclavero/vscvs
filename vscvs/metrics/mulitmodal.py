__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite metrics for multimodal networks. """


from abc import ABC
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from overrides import overrides


class AbstractAverageDistances(Metric, ABC):
    """
    Computes the average distances for embeddings of the same (positive) and different (negative) classes.
    """
    def __init__(self, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: Metric arguments
        :type: tuple
        :param batch_size: callable taking a target tensor returns the first dimension size (usually the batch size).
        :type: Callable<args: 'tensor', ret: int>
        :param kwargs: Metric keyword arguments
        :type: dict
        """
        self._batch_size = batch_size
        self._sum_positive_distances = 0
        self._sum_negative_distances = 0
        self._num_examples_positive = 0
        self._num_examples_negative = 0
        super().__init__(*args, **kwargs)

    @overrides
    @sync_all_reduce('_sum_positive', '_num_examples_positive', '_sum_negative', '_num_examples_negative')
    def compute(self):
        if self._num_examples_positive == 0 or self._num_examples_negative == 0:
            raise NotComputableError('AverageDistances needs at least one example per target to be computed.')
        return self._sum_positive_distances / self._num_examples_positive, \
               self._sum_negative_distances / self._num_examples_negative

    @overrides
    @reinit__is_reduced
    def reset(self):
        self._sum_positive_distances = 0
        self._sum_negative_distances = 0
        self._num_examples_positive = 0
        self._num_examples_negative = 0
