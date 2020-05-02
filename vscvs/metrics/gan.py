__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite metrics for GANs. """


from abc import ABC
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from overrides import overrides


class AbstractGANLoss(Metric, ABC):
    """
    Computes the average loss for a GAN.
    """
    def __init__(self, loss_fn, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: Metric arguments
        :type: tuple
        :param loss_fn: callable taking a GAN output and returns the average losses over all observations in the batch.
        :type: Callable<args: ``, `positive_embeddings`, `negative_embeddings`, ret: tuple<float>>
        :param batch_size: callable taking a target tensor returns the first dimension size (usually the batch size).
        :type: Callable<args: 'tensor', ret: int>
        :param kwargs: Metric keyword arguments
        :type: dict
        """
        self._loss_fn = loss_fn
        self._batch_size = batch_size
        self._sum_generator_loss = 0
        self._sum_discriminator_loss = 0
        self._num_examples = 0
        super().__init__(*args, **kwargs)

    @overrides
    @sync_all_reduce('_sum_generator_loss', '_sum_discriminator_loss', '_num_examples')
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Loss must have at least one example before it can be computed.')
        return self._sum_generator_loss / self._num_examples, self._sum_discriminator_loss / self._num_examples

    @overrides
    @reinit__is_reduced
    def reset(self):
        self._sum_generator_loss = 0
        self._sum_discriminator_loss = 0
        self._num_examples = 0


class MultimodalGANLoss(AbstractGANLoss):
    """
    Computes the average loss for a multimodal GAN.
    """
    @overrides
    @reinit__is_reduced
    def update(self, output):
        """
        :override: updates the metric's state using the passed GAN batch output.
        :param output: the output of the engine's process function, using the GAN format, which contains the element
        embeddings in the first element by default.
        :type: tuple<torch.Tensor, ...>
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
