__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a ResNet classification network. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss

from src.models.convolutional.resnet import ResNet
from src.trainers.abstract_trainers import AbstractSGDOptimizerTrainer
from src.trainers.mixins import EarlyStoppingMixin
from src.utils.data import prepare_batch
from src.utils.decorators import kwargs_parameter_dict


class ResNetTrainer(AbstractSGDOptimizerTrainer, EarlyStoppingMixin):
    """
    Trainer for a ResNext image classifier.
    """
    def __init__(self, *args, **kwargs):
        """
        Trainer constructor.
        :param args: AbstractTrainer and EarlyStoppingMixin arguments
        :type: tuple
        :param kwargs: AbstractTrainer and EarlyStoppingMixin keyword arguments
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


@kwargs_parameter_dict
def train_resnet(*args, **kwargs):
    """
    Train a ResNet image classifier.
    :param args: ResNetTrainer arguments
    :type: tuple
    :param kwargs: ResNetTrainer keyword arguments
    :type: dict
    """
    trainer = ResNetTrainer(*args, **kwargs)
    trainer.run()
