__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainers for ResNext classification networks. """


from abc import ABC
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, TopKCategoricalAccuracy, Precision
from overrides import overrides
from torch.nn import CrossEntropyLoss
from typing import Callable

from .abstract_trainer import AbstractTrainer
from .mixins import EarlyStoppingMixin
from vscvs.models import ResNextLogSoftmax
from vscvs.utils import prepare_batch
from vscvs.decorators import kwargs_parameter_dict


class AbstractResNextTrainer(EarlyStoppingMixin, AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a ResNext model.
    """
    def __init__(self, *args, out_features=125, pretrained=False, **kwargs):
        """
        :param args: Trainer arguments
        :type: Tuple
        :param out_features: number of output features. If `None`, defaults to 1000.
        :type: int
        :param pretrained: if True, uses a model pre-trained on ImageNet.
        :type: bool
        :param kwargs: Trainer keyword arguments
        :type: Dict
        """
        self.out_features = out_features
        self.pretrained = pretrained
        super().__init__(*args, **kwargs)

    @property
    @overrides
    def initial_model(self):
        return ResNextLogSoftmax(out_features=self.out_features, pretrained=self.pretrained)

    @property
    @overrides
    def loss(self):
        return CrossEntropyLoss()

    @property
    @overrides
    def trainer_id(self):
        return 'ResNext'

    @staticmethod
    @overrides
    def _score_function(engine):
        precision = engine.state.metrics['Precision']
        return precision

    @overrides
    def _create_evaluator_engine(self):
        return create_supervised_evaluator(
            self.model, device=self.device,
            metrics={'Accuracy': Accuracy(), 'Loss': Loss(self.loss), 'Recall': Recall(average=True),
                     'Top K Categorical Accuracy': TopKCategoricalAccuracy(k=10),
                     'Precision': Precision(average=True)})

    @overrides
    def _create_trainer_engine(self):
        return create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device, prepare_batch=prepare_batch)


@kwargs_parameter_dict
def train_resnext(*args, optimizer_mixin=None, **kwargs):
    """
    Train a ResNext image classifier.
    :param args: ResNextTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: ResNextTrainer keyword arguments
    :type: Dict
    """
    class ResNextTrainer(optimizer_mixin, AbstractResNextTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm
    trainer = ResNextTrainer(*args, **kwargs)
    trainer.run()
