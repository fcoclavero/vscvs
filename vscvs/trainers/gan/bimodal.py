__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a Bimodal GAN architecture. """


from abc import ABC
from overrides import overrides
from torch.nn import BCEWithLogitsLoss
from typing import Callable

from .gan import AbstractGANTrainer
from ..engines.gan import create_multimodal_gan_evaluator, create_multimodal_gan_trainer, \
    prepare_bimodal_batch_variables
from vscvs.metrics import LossMultimodalGAN
from vscvs.models import ResNextNormalized, InterModalDiscriminator, MultimodalEncoder
from vscvs.decorators import kwargs_parameter_dict


class AbstractBiModalGANTrainer(AbstractGANTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a bi-modal GAN architecture.
    """
    def __init__(self, *args, mode_embedding_networks=None, loss_weight=None, **kwargs):
        """
        :param args: AbstractGANTrainer arguments
        :type: tuple
        :param mode_embedding_networks: the embedding networks for each mode. They will be used as generators for the
        generative adversarial formulation.
        :type: list<torch.nn.Module>
        :param loss_weight: manual rescaling weight given to the loss of each batch element. If given, has to be a
        Tensor of size `batch_size`.
        :type: torch.Tensor
        :param kwargs: AbstractGANTrainer keyword arguments
        :type: dict
        """
        self.loss_weight = loss_weight
        super().__init__(*args, generator_network=MultimodalEncoder(*mode_embedding_networks), **kwargs)

    @property
    @overrides
    def loss(self):
        return BCEWithLogitsLoss(reduction=self.loss_reduction, weight=self.loss_weight)

    @overrides
    def _create_evaluator_engine(self):
        loss = LossMultimodalGAN(self.loss)
        return create_multimodal_gan_evaluator(
            *self.model, device=self.device, metrics={'generator_loss': loss[0], 'discriminator_loss': loss[1]},
            prepare_batch_variables=prepare_bimodal_batch_variables)

    @overrides
    def _create_trainer_engine(self):
        return create_multimodal_gan_trainer(*self.model, *self.optimizer, self.loss, device=self.device,
                                             prepare_batch_variables=prepare_bimodal_batch_variables)


@kwargs_parameter_dict
def train_gan_bimodal(*args, optimizer_mixin=None, **kwargs):
    """
    Train a bimodal GAN.
    :param args: BiModalGANTrainer arguments
    :type: tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: BiModalGANTrainer keyword arguments
    :type: dict
    """
    class BiModalGANTrainer(optimizer_mixin, AbstractBiModalGANTrainer):
        _optimizer: Callable # type hinting: `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm
    trainer = BiModalGANTrainer(*args, discriminator_network=InterModalDiscriminator(input_dimension=250),
                                mode_embedding_networks=[ResNextNormalized(out_features=250, pretrained=True),
                                                         ResNextNormalized(out_features=250, pretrained=True)],
                                **kwargs)
    trainer.run()
