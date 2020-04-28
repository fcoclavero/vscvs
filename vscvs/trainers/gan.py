__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GAN architecture. """


import os
import torch

from abc import ABC
from ignite.engine import Events
from torch.nn import BCELoss

from vscvs.models import ResNextNormalized
from vscvs.models.gan import InterModalDiscriminatorSoftmax, MultimodalEncoder
from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.engines.gan import create_gan_evaluator, create_gan_trainer
from vscvs.trainers.mixins import GANOptimizerMixin
from vscvs.utils import initialize_weights
from vscvs.decorators import kwargs_parameter_dict


class AbstractGANTrainer(GANOptimizerMixin, AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a GAN architecture.
    """
    def __init__(self, *args, discriminator_network=None, generator_network=None, loss_reduction='mean',
                 loss_weight=None, **kwargs):
        """
        Trainer constructor.
        :param args: Trainer arguments
        :type: tuple
        :param discriminator_network: the discriminator model that classifies generator outputs.
        :type: torch.nn.Module
        :param generator_network: the generator model.
        :type: torch.nn.Module
        :param loss_reduction: reduction to apply to batch element loss values to obtain the loss for the whole batch.
`       Must correspond to a valid reduction for the `ContrastiveLoss`.
        :type: str
        :param loss_weight: manual rescaling weight given to the loss of each batch element. If given, has to be a
        Tensor of size `batch_size`.
        :type: torch.Tensor
        :param kwargs: Trainer keyword arguments
        :type: dict
        """
        self.discriminator = discriminator_network
        self.generator = generator_network
        self.loss_weight = loss_weight
        self.loss_reduction = loss_reduction
        super().__init__(*args, **kwargs)

    @property
    def initial_model(self):
        self.generator.apply(initialize_weights)
        self.discriminator.apply(initialize_weights)
        return self.generator, self.discriminator

    @property
    def trainer_id(self):
        return 'GAN{}{}'.format(self.generator.__class__.__name__, self.discriminator.__class__.__name__)

    def _add_model_checkpoint_savers(self):
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.checkpoint_saver_best, {
            'generator': self.generator, 'discriminator': self.discriminator})
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.checkpoint_saver_periodic, {
            'generator': self.generator, 'discriminator': self.discriminator})

    def _create_evaluator_engine(self):
        return create_gan_evaluator(*self.model, device=self.device, metrics={})

    def _create_trainer_engine(self):
        return create_gan_trainer(*self.model, *self.optimizer, self.loss, device=self.device)

    def _load_model_checkpoint(self, previous_checkpoint_directory):
        """
        Load the model state_dict saved in the model checkpoint file into the already initialized model field.
        :param previous_checkpoint_directory: directory containing the checkpoint to me loaded.
        :type: str
        """
        state_dicts = torch.load(os.path.join(previous_checkpoint_directory, '{}.pth'.format(self.resume_checkpoint)))
        self.generator.load_state_dict(state_dicts['generator'])
        self.discriminator.load_state_dict(state_dicts['discriminator'])


class AbstractMultiModalGANTrainer(AbstractGANTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a multi-modal GAN architecture.
    """
    def __init__(self, *args, mode_embedding_networks=None, **kwargs):
        """
        Trainer constructor.
        :param args: AbstractGANTrainer arguments
        :type: tuple
        :param mode_embedding_networks: the embedding networks for each mode. They will be used as generators for the
        generative adversarial formulation.
        :type: list<torch.nn.Module>
        :param kwargs: AbstractGANTrainer keyword arguments
        :type: dict
        """
        super().__init__(*args, generator_network=MultimodalEncoder(*mode_embedding_networks), **kwargs)

    @property
    def loss(self):
        return BCELoss(reduction=self.loss_reduction, weight=self.loss_weight)


@kwargs_parameter_dict
def train_gan_multimodal(*args, optimizer_mixin=None, **kwargs):
    """
    Train a multimodal GAN.
    :param args: MultiModalGANTrainer arguments
    :type: tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: MultiModalGANTrainer keyword arguments
    :type: dict
    """
    class MultiModalGANTrainer(optimizer_mixin, AbstractMultiModalGANTrainer):
        pass
    trainer = MultiModalGANTrainer(*args, discriminator_network=InterModalDiscriminatorSoftmax(input_dimension=250),
                                   mode_embedding_networks=[ResNextNormalized(out_features=250, pretrained=True),
                                                            ResNextNormalized(out_features=250, pretrained=True)],
                                   **kwargs)
    trainer.run()
