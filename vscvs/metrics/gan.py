__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Custom Ignite metrics for GANs. """


import torch

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
    def __init__(self, *args, batch_size=lambda x: len(x), **kwargs):
        """
        :param args: `Metric` arguments.
        :type: Tuple
        :param batch_size: callable that takes a target tensor and returns the first dimension size (`batch_size`).
        :type: Callable[[torch.Tensor] int]
        :param kwargs: `Metric` keyword arguments.
        :type: Dict
        """
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

    def update_metric_state(self, generator_loss, discriminator_loss, batch_size):
        """
        Utility method used to update the metrics internal state based on the computed generator and discriminator loss.
        :param generator_loss: the value of the generator loss for the engine output passed to the `update` function.
        :type: float
        :param discriminator_loss: the value of the discriminator loss for the output passed to the `update` function.
        :type: float
        :param batch_size: the batch size of the output passed to the `update` function.
        :type: int
        """
        self._sum_generator_loss += generator_loss * batch_size
        self._sum_discriminator_loss += discriminator_loss * batch_size
        self._num_examples += batch_size


class LossMultimodalGAN(AbstractLossGAN):
    """
    Computes the average loss for a multimodal GAN.
    """
    def __init__(self, loss_fn, *args, **kwargs):
        """
        :param loss_fn: callable that takes a multimodal GAN output and returns the average losses over batch elements.
        :type: Callable[[Tuple], float]
        :param args: `AbstractLossGAN` arguments.
        :type: List
        :param kwargs: `AbstractLossGAN` keyword arguments.
        :type: Dict
        """
        super().__init__(*args, **kwargs)
        self.generator_loss_fn = self.discriminator_loss_fn = loss_fn

    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using the passed GAN batch output.
        :param output: the output of the engine's process function, using the GAN format, which by default is a tuple
        containing embeddings, mode_predictions, mode_labels, and generator_labels.
        :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        :raises ValueError: when loss function cannot be computed
        """
        if len(output) == 4:
            kwargs = {}
            _, mode_predictions, mode_labels, generator_labels = output
        else:
            _, mode_predictions, mode_labels, generator_labels, kwargs = output

        generator_loss = self.generator_loss_fn(mode_predictions, generator_labels, **kwargs).item()
        discriminator_loss = self.discriminator_loss_fn(mode_predictions, mode_labels, **kwargs).item()
        batch_size = self._batch_size(mode_predictions)
        self.update_metric_state(generator_loss, discriminator_loss, batch_size)


class LossBimodalSiamesePairs(AbstractLossGAN, ABC):
    """
    Computes the loss for positive and negative pairs in a bimodal siamese network.
    """
    def __init__(self, loss_fn, *args, **kwargs):
        """
        :param loss_fn: 2-tuple of loss functions, the first corresponding to the bimodal loss and the second to the
        siamese contrastive loss. The bimodal loss function is a callable that takes a bimodal GAN output and returns
        the reduced mode classification loss over batch elements. The siamese loss function is a callable that takes
        two batches of embeddings (where elements with the same index correspond to siamese pairs) and the siamese
        target tensor (which indicates whether the pair at each index is positive - same class - or negative) and
        returns the reduced contrastive loss over batch elements.
        :type: Tuple[Callable[[Tuple], float], Callable[[Tuple], float]]
        :param args: `AbstractLossGAN` arguments.
        :type: List
        :param kwargs: `AbstractLossGAN` keyword arguments.
        :type: Dict
        """
        super().__init__(*args, **kwargs)
        self.bimodal_loss_fn, self._contrastive_loss_fn = loss_fn

    @reinit__is_reduced
    @overrides
    def update(self, output):
        """
        :override: updates the metric's state using a multimodal siamese GAN batch output.
        :param output: the output of the engine's process function, using the multimodal siamese format: 6-tuple with
        the first pair elements' embeddings, the second pair elements' embeddings, the siamese target, mode
        predictions, mode labels, and generator labels.
        :type: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        embeddings_0, embeddings_1, siamese_target, mode_predictions, mode_labels, generator_labels = output
        generator_loss = self.bimodal_loss_fn(mode_predictions, generator_labels).item() + \
                         self._contrastive_loss_fn(embeddings_0, embeddings_1, siamese_target).item()
        discriminator_loss = self.bimodal_loss_fn(mode_predictions, mode_labels).item()
        batch_size = self._batch_size(mode_predictions)
        self.update_metric_state(generator_loss, discriminator_loss, batch_size)


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
        super().update((embeddings_0, embeddings_1, siamese_target))
