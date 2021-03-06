__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Ignite trainer for a Bimodal GAN architecture. """


from abc import ABC
from typing import Callable

from overrides import overrides
from torch.nn import BCEWithLogitsLoss

from vscvs.decorators import kwargs_parameter_dict
from vscvs.loss_functions import ContrastiveLoss
from vscvs.metrics import AverageDistancesMultimodalSiamesePairs
from vscvs.metrics import LossBimodalSiamesePairs
from vscvs.metrics import LossMultimodalGAN
from vscvs.models import InterModalDiscriminator
from vscvs.models import MultimodalEncoder
from vscvs.models import ResNextNormalized

from ..engines.gan import create_multimodal_gan_evaluator
from ..engines.gan import create_multimodal_gan_siamese_evaluator
from ..engines.gan import create_multimodal_gan_siamese_trainer
from ..engines.gan import create_multimodal_gan_trainer
from ..engines.gan import prepare_bimodal_batch_variables
from .gan import AbstractGANTrainer


class AbstractBiModalGANTrainer(AbstractGANTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a bi-modal GAN architecture.
    """

    def __init__(self, *args, mode_embedding_networks=None, loss_weight=None, **kwargs):
        """
        :param args: AbstractGANTrainer arguments
        :type: Tuple
        :param mode_embedding_networks: the embedding networks for each mode. They will be used as generators for the
        generative adversarial formulation.
        :type: List[torch.nn.Module]
        :param loss_weight: manual rescaling weight given to the loss of each batch element. If given, has to be a
        Tensor of size `batch_size`.
        :type: torch.Tensor
        :param kwargs: AbstractGANTrainer keyword arguments
        :type: Dict
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
            *self.model,
            device=self.device,
            metrics={"generator_loss": loss[0], "discriminator_loss": loss[1]},
            prepare_batch_variables=prepare_bimodal_batch_variables
        )

    @overrides
    def _create_trainer_engine(self):
        return create_multimodal_gan_trainer(
            *self.model,
            *self.optimizer,
            self.loss,
            device=self.device,
            prepare_batch_variables=prepare_bimodal_batch_variables
        )


class AbstractBiModalGANSiameseTrainer(AbstractBiModalGANTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a bi-modal GAN architecture with
    the addition of a contrastive term in the loss functions.
    """

    def __init__(self, *args, margin=0.2, **kwargs):
        """
        :param args: AbstractBiModalGANTrainer arguments
        :type: Tuple
        :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the
        embeddings of two examples as dissimilar. Dissimilar image pairs will be pushed apart unless their distance
        is already greater than the margin. Similar sketch–image pairs will be pulled together in the feature space.
        :type: float
        :param kwargs: AbstractBiModalGANTrainer keyword arguments
        :type: Dict
        """
        self.margin = margin
        super().__init__(*args, **kwargs)

    @property
    @overrides
    def loss(self):
        return (
            BCEWithLogitsLoss(reduction=self.loss_reduction, weight=self.loss_weight),
            ContrastiveLoss(margin=self.margin, reduction=self.loss_reduction),
        )

    @overrides
    def _create_evaluator_engine(self):
        average_distances = AverageDistancesMultimodalSiamesePairs()
        loss = LossBimodalSiamesePairs(self.loss)
        return create_multimodal_gan_siamese_evaluator(
            *self.model,
            device=self.device,
            prepare_batch_variables=prepare_bimodal_batch_variables,
            metrics={
                "Average Distance/positive": average_distances[0],
                "Average Distance/negative": average_distances[1],
                "Loss/generator": loss[0],
                "Loss/discriminator": loss[1],
            }
        )

    @overrides
    def _create_trainer_engine(self):
        return create_multimodal_gan_siamese_trainer(
            *self.model,
            *self.optimizer,
            *self.loss,
            device=self.device,
            prepare_batch_variables=prepare_bimodal_batch_variables
        )


@kwargs_parameter_dict
def train_gan_bimodal(*args, optimizer_mixin=None, **kwargs):
    """
    Train a bimodal GAN.
    :param args: BiModalGANTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: BiModalGANTrainer keyword arguments
    :type: Dict
    """

    class BiModalGANTrainer(optimizer_mixin, AbstractBiModalGANTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm

    trainer = BiModalGANTrainer(
        *args,
        discriminator_network=InterModalDiscriminator(input_dimension=250),
        mode_embedding_networks=[
            ResNextNormalized(out_features=250, pretrained=True),
            ResNextNormalized(out_features=250, pretrained=True),
        ],
        **kwargs
    )
    trainer.run()


@kwargs_parameter_dict
def train_gan_bimodal_siamese(*args, optimizer_mixin=None, **kwargs):
    """
    Train a bimodal GAN.
    :param args: BiModalGANSiameseTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: BiModalGANSiameseTrainer keyword arguments
    :type: Dict
    """

    class BiModalGANSiameseTrainer(optimizer_mixin, AbstractBiModalGANSiameseTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm

    trainer = BiModalGANSiameseTrainer(
        *args,
        discriminator_network=InterModalDiscriminator(input_dimension=250),
        mode_embedding_networks=[
            ResNextNormalized(out_features=250, pretrained=True),
            ResNextNormalized(out_features=250, pretrained=True),
        ],
        **kwargs
    )
    trainer.run()
