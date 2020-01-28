__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainers for ResNet classification networks. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss

from src.models.convolutional.resnet import ResNet
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.mixins import EarlyStoppingMixin
from src.utils.data import prepare_batch
from src.utils.decorators import kwargs_parameter_dict


def resnet(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a ResNet model.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a ResNet model
    :type: `cls.__class__`
    """
    class ResNetTrainer(cls):
        """
        Trainer for a ResNext image classifier.
        """
        def __init__(self, *args, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractSGDOptimizerTrainer and EarlyStoppingMixin arguments
            :type: tuple
            :param kwargs: AbstractSGDOptimizerTrainer and EarlyStoppingMixin keyword arguments
            :type: dict
            """
            super().__init__(*args, **kwargs)

        @property
        def initial_model(self):
            return ResNet(out_features=125)

        @property
        def loss(self):
            return CrossEntropyLoss()

        @property
        def serialized_checkpoint(self):
            return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'momentum': self.momentum}

        @property
        def trainer_id(self):
            return 'resnet'

        @staticmethod
        def _score_function(engine):
            validation_loss = engine.state.metrics['loss']
            return -validation_loss

        def _create_evaluator_engine(self):
            return create_supervised_evaluator(
                self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss)}, device=self.device)

        def _create_trainer_engine(self):
            return create_supervised_trainer(
                self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)

    return ResNetTrainer


@kwargs_parameter_dict
def train_resnet(*args, optimizer_decorator=None, **kwargs):
    """
    Train a ResNet image classifier.
    :param args: ResNetTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: ResNetTrainer keyword arguments
    :type: dict
    """
    @resnet
    @optimizer_decorator
    class SGDResNetTrainer(AbstractTrainer, EarlyStoppingMixin):
        pass
    trainer = SGDResNetTrainer(*args, **kwargs)
    trainer.run()
