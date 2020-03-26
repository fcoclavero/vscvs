__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainers for ResNet classification networks. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss

from src.models import ResNetLogSoftmax
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.mixins import EarlyStoppingMixin
from src.utils.data import prepare_batch
from src.decorators import kwargs_parameter_dict


def resnet(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a ResNet model.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a ResNet model
    :type: `cls.__class__`
    """
    class Trainer(cls):
        """
        Trainer for a ResNext image classifier.
        """
        def __init__(self, *args, out_features=125, pretrained=False, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractSGDOptimizerTrainer and EarlyStoppingMixin arguments
            :type: tuple
            :param out_features: number of output features. If `None`, defaults to 1000.
            :type: int or None
            :param pretrained: if True, uses a model pre-trained on ImageNet.
            :type: boolean
            :param kwargs: AbstractSGDOptimizerTrainer and EarlyStoppingMixin keyword arguments
            :type: dict
            """
            self.out_features = out_features
            self.pretrained = pretrained
            super().__init__(*args, **kwargs)

        @property
        def initial_model(self):
            return ResNetLogSoftmax(out_features=self.out_features, pretrained=self.pretrained)

        @property
        def loss(self):
            return CrossEntropyLoss()

        @property
        def trainer_id(self):
            return 'ResNet'

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

    return Trainer


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
    class ResNetTrainer(AbstractTrainer, EarlyStoppingMixin):
        pass
    trainer = ResNetTrainer(*args, **kwargs)
    trainer.run()
