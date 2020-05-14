__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite metrics for GANs. """


from abc import ABC
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from overrides import overrides

from .siamese import AverageDistancesSiamesePairs


class AbstractLossGAN(Metric, ABC):
    """
    Computes the average loss for a GAN.
    """
    def __init__(self, loss_fn, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: Metric arguments
        :type: tuple
        :param loss_fn: callable that takes a GAN output and returns the average losses over all batch elements.
        :type: Callable[[Tuple], float]
        :param batch_size: callable that takes a target tensor and returns the first dimension size (`batch_size`).
        :type: Callable[[torch.Tensor] int]
        :param kwargs: Metric keyword arguments
        :type: dict
        """
        self._loss_fn = loss_fn
        self._batch_size = batch_size
        self._sum_generator_loss = 0
        self._sum_discriminator_loss = 0
        self._num_examples = 0
        super().__init__(*args, **kwargs)

    @sync_all_reduce('_sum_generator_loss', '_sum_discriminator_loss', '_num_examples')
    @overrides
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')
        return self._sum_generator_loss / self._num_examples, self._sum_discriminator_loss / self._num_examples

    @reinit__is_reduced
    @overrides
    def reset(self):
        self._sum_generator_loss = 0
        self._sum_discriminator_loss = 0
        self._num_examples = 0


class LossMultimodalGAN(AbstractLossGAN):
    """
    Computes the average loss for a multimodal GAN.
    """
    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using the passed GAN batch output.
        :param output: the output of the engine's process function, using the GAN format, which by default is a tuple
        containing embeddings, mode_predictions, mode_labels, generator_labels, classes
        :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        :raises ValueError: when loss function cannot be computed
        """
        if len(output) == 5:
            kwargs = {}
            _, mode_predictions, mode_labels, generator_labels, _ = output
        else:
            _, mode_predictions, mode_labels, generator_labels, _, kwargs = output

        generator_loss = self._loss_fn(mode_predictions, generator_labels, **kwargs)
        discriminator_loss = self._loss_fn(mode_predictions, mode_labels, **kwargs)

        if len(generator_loss.shape) != 0 or len(discriminator_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        batch_size = self._batch_size(mode_predictions)
        self._sum_generator_loss += generator_loss.item() * batch_size
        self._sum_discriminator_loss += discriminator_loss.item() * batch_size
        self._num_examples += batch_size


class AverageDistancesMultimodalSiamesePairs(AverageDistancesSiamesePairs):
    """
    Computes the average distances for positive and negative pairs in a multimodal siamese network.
    """
    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using a multimodal siamese GAN batch output.
        :param output: the output of the engine's process function, using the multimodal siamese format: 6-tuple with
        the first pair elements' embeddings, the second pair elements' embeddings, the siamese targets, mode
        predictions, mode labels, and generator labels.
        :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        embeddings_0, embeddings_1, siamese_target, *_ = output
        super().update(embeddings_0, embeddings_1, siamese_target,)
