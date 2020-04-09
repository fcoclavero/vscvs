__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainers for a CNN classification networks. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss

from vscvs.models import CNNLogSoftmax
from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.mixins import EarlyStoppingMixin
from vscvs.utils.data import prepare_batch
from vscvs.decorators import kwargs_parameter_dict


def cnn(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a CNN model.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a CNN model
    :type: `cls.__class__`
    """
    class CNNTrainer(cls):
        """
        Trainer for a simple class classification CNN.
        """
        def __init__(self, *args, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractTrainer arguments
            :type: tuple
            :param kwargs: AbstractTrainer keyword arguments
            :type: dict
            """
            super().__init__(*args, **kwargs)

        @property
        def initial_model(self):
            return CNNLogSoftmax(out_features=250)

        @property
        def loss(self):
            return CrossEntropyLoss()

        @property
        def trainer_id(self):
            return 'CNN'

        def _create_evaluator_engine(self):
            return create_supervised_evaluator(
                self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss)}, device=self.device)

        def _create_trainer_engine(self):
            return create_supervised_trainer(
                self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)

    return CNNTrainer


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
    @cnn
    class CNNTrainer(optimizer_mixin, AbstractTrainer, EarlyStoppingMixin):
        pass
    trainer = CNNTrainer(*args, **kwargs)
    trainer.run()
