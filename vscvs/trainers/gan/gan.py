__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GAN architectures. """


import os
import torch

from abc import ABC
from ignite.engine import Events
from overrides import overrides
from tqdm import tqdm

from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.mixins import GANOptimizerMixin
from vscvs.utils import initialize_weights


class AbstractGANTrainer(GANOptimizerMixin, AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a GAN architecture.
    """
    def __init__(self, *args, discriminator_network=None, generator_network=None, loss_reduction='mean', **kwargs):
        """
        :param args: Trainer arguments
        :type: Tuple
        :param discriminator_network: the discriminator model that classifies generator outputs.
        :type: torch.nn.Module
        :param generator_network: the generator model.
        :type: torch.nn.Module
        :param loss_reduction: reduction to apply to batch element loss values to obtain the loss for the whole batch.
        :type: str
        :param kwargs: Trainer keyword arguments
        :type: Dict
        """
        self.discriminator = discriminator_network
        self.generator = generator_network
        self.loss_reduction = loss_reduction
        super().__init__(*args, **kwargs)

    @property
    @overrides
    def initial_model(self):
        self.generator.apply(initialize_weights)
        self.discriminator.apply(initialize_weights)
        return self.generator, self.discriminator

    @property
    @overrides
    def progressbar_description(self):
        """
        The format string to be displayed beside the `tqdm` progressbar.
        :return: the progressbar description string
        :type: str
        """
        return 'TRAINING epoch {}/{} => generator loss: {:.5f} discriminator loss: {:.5f}'

    @property
    @overrides
    def trainer_id(self):
        return 'GAN{}{}'.format(self.generator.__class__.__name__, self.discriminator.__class__.__name__)

    @overrides
    def _add_model_checkpoint_savers(self):
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.checkpoint_saver_best, {
            'generator': self.generator, 'discriminator': self.discriminator})
        self.trainer_engine.add_event_handler(Events.EPOCH_COMPLETED, self.checkpoint_saver_periodic, {
            'generator': self.generator, 'discriminator': self.discriminator})

    @overrides
    def _event_log_training_output(self, trainer):
        """
        Write the trainer state output to the progressbar and tensorboard log writer.
        :param trainer: the ignite trainer engine this event was bound to.
        :type: ignite.engine.Engine
        """
        self.writer.add_scalar('Trainer Output/generator', trainer.state.output[0], self.step)
        self.writer.add_scalar('Trainer Output/discriminator', trainer.state.output[1], self.step)
        self.progressbar.desc = self.progressbar_description.format(self.epoch, self.last_epoch, *trainer.state.output)

    @overrides
    def _load_model_checkpoint(self, previous_checkpoint_directory):
        """
        Load the model state_dict saved in the model checkpoint file into the already initialized model field.
        :param previous_checkpoint_directory: directory containing the checkpoint to me loaded.
        :type: str
        """
        state_dicts = torch.load(os.path.join(previous_checkpoint_directory, '{}.pt'.format(self.resume_checkpoint)))
        self.generator.load_state_dict(state_dicts['generator'])
        self.discriminator.load_state_dict(state_dicts['discriminator'])

    @property
    @overrides
    def _progressbar(self):
        """
        TQDM progressbar to display trainer progress.
        :return: the trainer progressbar.
        :type: tqdm.tqdm
        """
        return tqdm(initial=0, leave=False, total=len(self.train_loader),
                    desc=self.progressbar_description.format(self.epoch, self.last_epoch, 0.0, 0.0))
