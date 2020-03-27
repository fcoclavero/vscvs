__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom ignite metrics. """


from abc import ABC
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

from src.utils.data import output_transform_siamese


class SiameseMetric(Metric, ABC):
    """
    Base class for ignite metrics for siamese networks. They receive batch pair embeddings and their target tensor,
    indicating whether each pair is similar (0) or dissimilar (1): either `(embeddings_0, embeddings_1, target)` or
    `{'embeddings_0': embeddings_0, 'embeddings_1': embeddings_1, 'target': target}`. This makes more sense in this
    setting than the default `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}` input (useful in supervised training).
    """
    def __init__(self, output_transform=output_transform_siamese, device=None):
        """
        Class constructor.
        :param output_transform: function that receives `embeddings_0` (first elements of each siamese pair in the
        batch), `embeddings_1` (second elements of each siamese pair in the batch), and `target` (target tensor,
        indicating whether each pair is similar or dissimilar) and the returns value to be assigned to the engine's
        state.output after each iteration. Default is returning either `(embeddings_0, embeddings_1, target)` or
        `{'embeddings_0': embeddings_0, 'embeddings_1': embeddings_1, 'target': target}`.
        :type: Callable<args: `batch`, `device`, `non_blocking`, ret: tuple<torch.Tensor, torch.Tensor, torch.Tensor>>
        (optional) (default: `output_transform_siamese`)
        :param device: device type specification.
        :type: str of torch.device (optional) (default: None)
        """
        super().__init__(output_transform=output_transform, device=device)


class SiameseLoss(SiameseMetric):
    """
    Computes the average loss for a Siamese network.
    """
    def __init__(self, *args, loss_fn, batch_size=lambda x: len(x), **kwargs):
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

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        """
        Computes the average loss based on it's accumulated state. This is called at the end of each epoch.
        :return: the actual average loss
        :type: float
        :raises NotComputableError: when the metric cannot be computed
        """
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')
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
        embeddings_0, embeddings_1, target, kwargs = output
        kwargs = kwargs if kwargs else {}
        average_loss = self._loss_fn(embeddings_0, embeddings_1, target, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        batch_size = self._batch_size(target)
        self._sum += average_loss.item() * batch_size
        self._num_examples += batch_size
