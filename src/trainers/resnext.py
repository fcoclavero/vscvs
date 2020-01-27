__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a ResNext classification network. """


from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, TopKCategoricalAccuracy, Precision
from torch import round
from torch.nn import CrossEntropyLoss

from src.models.convolutional.resnext import ResNext
from src.trainers.abstract_trainers import AbstractAdamOptimizerTrainer
from src.trainers.mixins import EarlyStoppingMixin
from src.utils.data import prepare_batch
from src.utils.decorators import kwargs_parameter_dict


class ResNextTrainer(AbstractAdamOptimizerTrainer, EarlyStoppingMixin):
    """
    Trainer for a ResNext image classifier.
    """
    def __init__(self, *args, **kwargs):
        """
        Trainer constructor.
        :param args: AbstractAdamOptimizerTrainer and EarlyStoppingMixin arguments
        :type: tuple
        :param kwargs: AbstractAdamOptimizerTrainer and EarlyStoppingMixin keyword arguments
        :type: dict
        """
        super().__init__(*args, **kwargs)

    @property
    def initial_model(self):
        return ResNext(out_features=125)

    @property
    def loss(self):
        return CrossEntropyLoss()

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'momentum': self.momentum}

    @property
    def trainer_id(self):
        return 'resnext'

    @staticmethod
    def _score_function(engine):
        precision = engine.state.metrics['precision']
        return precision

    def _create_evaluator_engine(self):
        return create_supervised_evaluator(
            self.model, device=self.device,
            metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss), 'recall': Recall(average=True),
                     'top_k_categorical_accuracy': TopKCategoricalAccuracy(k=10), 'precision': Precision(average=True)})

    def _create_trainer_engine(self):
        return create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)

    @staticmethod
    def _output_transform(output):
        y_pred, y = output
        y_pred = round(y_pred)
        return y_pred, y


@kwargs_parameter_dict
def train_resnext(*args, **kwargs):
    """
    Train a ResNext image classifier.
    :param args: ResNextTrainer arguments
    :type: tuple
    :param kwargs: ResNextTrainer keyword arguments
    :type: dict
    """
    trainer = ResNextTrainer(*args, **kwargs)
    trainer.run()
