__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Ignite trainer for a GCN image label classifier, using binary or one-hot encodings as image feature vectors. """


from abc import ABC
from ignite.metrics import Accuracy, Loss, Recall, TopKCategoricalAccuracy
from overrides import overrides
from torch.nn import CrossEntropyLoss
from typing import Callable

from .abstract_trainer import AbstractTrainer
from .engines.classification_gcn import create_classification_gcn_evaluator, create_classification_gcn_trainer
from vscvs.datasets import get_dataset
from vscvs.models import GCNClassification
from vscvs.decorators import kwargs_parameter_dict


class AbstractClassificationGCNTrainer(AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a classification GCN.
    """

    def __init__(self, *args, dataset_name=None, processes=None, **kwargs):
        """
        :param args: Trainer arguments
        :type: Tuple
        :param dataset_name: the name of the Dataset to be used for training
        :type: str
        :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then
        `os.cpu_count()` will be used.
        :type: int
        :param kwargs: Trainer keyword arguments
        :type: Dict
        """
        self.dataset_name = dataset_name
        self.processes = processes
        super().__init__(*args, dataset_name=self.dataset_name, **kwargs)

    @property
    @overrides
    def initial_model(self):
        dataset = get_dataset(self.dataset_name)
        return GCNClassification(len(dataset.classes), 11)

    @property
    @overrides
    def loss(self):
        return CrossEntropyLoss()

    @property
    @overrides
    def trainer_id(self):
        return "ClassificationGCN"

    @overrides
    def _create_evaluator_engine(self):
        return create_classification_gcn_evaluator(
            self.model,
            self.dataset.classes_dataframe,
            device=self.device,
            processes=self.processes,
            metrics={
                "Accuracy": Accuracy(),
                "Loss": Loss(self.loss),
                "Recall": Recall(average=True),
                "Top K Categorical Accuracy": TopKCategoricalAccuracy(k=10),
            },
        )

    @overrides
    def _create_trainer_engine(self):
        return create_classification_gcn_trainer(
            self.model,
            self.optimizer,
            self.loss,
            self.dataset.classes_dataframe,
            device=self.device,
            processes=self.processes,
        )


@kwargs_parameter_dict
def train_classification_gcn(*args, optimizer_mixin=None, **kwargs):
    """
    Train a ClassificationGCN image classifier.
    :param args: ClassificationGCNTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: ClassificationGCNTrainer keyword arguments
    :type: Dict
    """

    class ClassificationGCNTrainer(optimizer_mixin, AbstractClassificationGCNTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm

    trainer = ClassificationGCNTrainer(*args, **kwargs)
    trainer.run()
