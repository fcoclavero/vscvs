__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GCN image label classifier, using binary or one-hot encodings as image feature vectors. """


from abc import ABC
from ignite.metrics import Accuracy, Loss, Recall, TopKCategoricalAccuracy
from torch.nn import CrossEntropyLoss

from vscvs.datasets import get_dataset
from vscvs.models import GCNClassification
from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.engines.classification_gcn import create_classification_gcn_evaluator, \
    create_classification_gcn_trainer
from vscvs.utils.data import prepare_batch_graph
from vscvs.decorators import kwargs_parameter_dict


class AbstractClassificationGCNTrainer(AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a classification GCN.
    """
    def __init__(self, *args, dataset_name=None, processes=None, **kwargs):
        """
        :param args: Trainer arguments
        :type: tuple
        :param dataset_name: the name of the Dataset to be used for training
        :type: str
        :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then
        `os.cpu_count()` will be used.
        :type: int or None
        :param kwargs: Trainer keyword arguments
        :type: dict
        """
        self.dataset_name = dataset_name
        self.processes = processes
        super().__init__(*args, dataset_name=self.dataset_name, **kwargs)

    @property
    def initial_model(self):
        dataset = get_dataset(self.dataset_name)
        return GCNClassification(len(dataset.classes), 11)

    @property
    def loss(self):
        return CrossEntropyLoss()

    @property
    def trainer_id(self):
        return 'ClassificationGCN'

    def _create_evaluator_engine(self):
        return create_classification_gcn_evaluator(
            prepare_batch_graph, self.model, self.dataset.classes_dataframe, device=self.device,
            processes=self.processes, metrics={
                'accuracy': Accuracy(), 'loss': Loss(self.loss),
                'recall': Recall(average=True), 'top_k_categorical_accuracy': TopKCategoricalAccuracy(k=10)})

    def _create_trainer_engine(self):
        return create_classification_gcn_trainer(
            prepare_batch_graph, self.model, self.dataset.classes_dataframe,
            self.optimizer, self.loss, device=self.device, processes=self.processes)


@kwargs_parameter_dict
def train_classification_gcn(*args, optimizer_mixin=None, **kwargs):
    """
    Train a ClassificationGCN image classifier.
    :param args: ClassificationGCNTrainer arguments
    :type: tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: ClassificationGCNTrainer keyword arguments
    :type: dict
    """
    class ClassificationGCNTrainer(optimizer_mixin, AbstractClassificationGCNTrainer):
        pass
    trainer = ClassificationGCNTrainer(*args, **kwargs)
    trainer.run()
