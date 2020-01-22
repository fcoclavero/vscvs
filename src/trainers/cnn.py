__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a CNN classification network. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from src.models import ClassificationConvolutionalNetwork
from src.trainers.abstract_trainer import AbstractTrainer
from src.utils.data import prepare_batch


class CNNTrainer(AbstractTrainer):
    """
    Trainer for a simple class classification CNN.
    """
    def __init__(self, *args, learning_rate=.01, momentum=.8, **kwargs):
        """
        Trainer constructor.
        :param learning_rate: learning rate for optimizers
        :type: float
        :param momentum: momentum parameter for SGD optimizer
        :type: float
        :param args: AbstractTrainer arguments
        :type: tuple
        :param kwargs: AbstractTrainer keyword arguments
        :type: dict
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    @property
    def initial_model(self):
        return ClassificationConvolutionalNetwork()

    @property
    def loss(self):
        return CrossEntropyLoss()

    @property
    def optimizer(self):
        return SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'momentum': self.momentum}

    @property
    def trainer_id(self):
        return 'cnn_sk'

    def _create_evaluator_engine(self):
        return create_supervised_evaluator(
            self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss)}, device=self.device)

    def _create_trainer_engine(self):
        return create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)


def train_cnn(*args, **kwargs):
    """
    Train a classification Convolutional Neural Network for image classes.
    :param args: ResNetTrainer arguments
    :type: tuple
    :param kwargs: ResNetTrainer keyword arguments
    :type: dict
    """
    trainer = CNNTrainer(*args, **kwargs)
    trainer.run()
