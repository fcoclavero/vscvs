__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a Multimodal GAN architecture. """


from abc import ABC
from overrides import overrides
from torch.nn import MSELoss
from typing import Callable

from vscvs.metrics import LossMultimodalGAN
from vscvs.models import ResNextNormalized, InterModalDiscriminatorSoftmax, MultimodalEncoder
from vscvs.trainers.gan import AbstractGANTrainer
from vscvs.trainers.engines.gan import create_multimodal_gan_evaluator, create_multimodal_gan_trainer
from vscvs.decorators import kwargs_parameter_dict


class AbstractMultiModalGANTrainer(AbstractGANTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a multi-modal GAN architecture.
    """
    def __init__(self, *args, mode_embedding_networks=None, **kwargs):
        """
        :param args: AbstractGANTrainer arguments
        :type: Tuple
        :param mode_embedding_networks: the embedding networks for each mode. They will be used as generators for the
        generative adversarial formulation.
        :type: List[torch.nn.Module]
        :param kwargs: AbstractGANTrainer keyword arguments
        :type: Dict
        """
        super().__init__(*args, generator_network=MultimodalEncoder(*mode_embedding_networks), **kwargs)

    @property
    @overrides
    def loss(self):
        return MSELoss(reduction=self.loss_reduction)

    @overrides
    def _create_evaluator_engine(self):
        loss = LossMultimodalGAN(self.loss)
        return create_multimodal_gan_evaluator(*self.model, device=self.device,
                                               metrics={'Loss/generator': loss[0], 'Loss/discriminator': loss[1]})

    @overrides
    def _create_trainer_engine(self):
        return create_multimodal_gan_trainer(*self.model, *self.optimizer, self.loss, device=self.device)


@kwargs_parameter_dict
def train_gan_multimodal(*args, optimizer_mixin=None, **kwargs):
    """
    Train a multimodal GAN.
    :param args: MultiModalGANTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: MultiModalGANTrainer keyword arguments
    :type: Dict
    """
    class MultiModalGANTrainer(optimizer_mixin, AbstractMultiModalGANTrainer):
        _optimizer: Callable # type hinting: `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm
    trainer = MultiModalGANTrainer(*args, discriminator_network=InterModalDiscriminatorSoftmax(input_dimension=250),
                                   mode_embedding_networks=[ResNextNormalized(out_features=250, pretrained=True),
                                                            ResNextNormalized(out_features=250, pretrained=True)],
                                   **kwargs)
    trainer.run()
