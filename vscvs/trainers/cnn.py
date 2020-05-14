__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainers for a CNN classification networks. """


from abc import ABC
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from overrides import overrides
from torch.nn import CrossEntropyLoss
from typing import Callable

from .abstract_trainer import AbstractTrainer
from .mixins import EarlyStoppingMixin
from vscvs.models import CNNLogSoftmax
from vscvs.utils import prepare_batch
from vscvs.decorators import kwargs_parameter_dict


class AbstractCNNTrainer(EarlyStoppingMixin, AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a CNN model.
    """
    def __init__(self, *args, out_features=125, **kwargs):
        """
        :param args: Trainer arguments
        :type: tuple
        :param out_features: number of output features. If `None`, defaults to 1000.
        :type: int
        :param kwargs: Trainer keyword arguments
        :type: dict
        """
        self.out_features = out_features
        super().__init__(*args, **kwargs)

    @property
    @overrides
    def initial_model(self):
        return CNNLogSoftmax(out_features=self.out_features)

    @property
    @overrides
    def loss(self):
        return CrossEntropyLoss()

    @staticmethod
    @overrides
    def _score_function(engine):
        validation_loss = engine.state.metrics['Loss']
        return -validation_loss

    @property
    @overrides
    def trainer_id(self):
        return 'CNN'

    @overrides
    def _create_evaluator_engine(self):
        return create_supervised_evaluator(
            self.model, metrics={'Accuracy': Accuracy(), 'Loss': Loss(self.loss)}, device=self.device)

    @overrides
    def _create_trainer_engine(self):
        return create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)


@kwargs_parameter_dict
def train_cnn(*args, optimizer_mixin=None, **kwargs):
    """
    Train a classification Convolutional Neural Network for image classes.
    :param args: CNNTrainer arguments
    :type: tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: CNNTrainer keyword arguments
    :type: dict
    """
    class CNNTrainer(optimizer_mixin, AbstractCNNTrainer):
        _optimizer: Callable # type hinting: `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm
    trainer = CNNTrainer(*args, **kwargs)
    trainer.run()
