__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GCN image label classifier, using binary or one-hot encodings as image feature vectors. """


from ignite.metrics import Accuracy, Loss, Recall, TopKCategoricalAccuracy
from torch.nn import CrossEntropyLoss

from vscvs.datasets import get_dataset
from vscvs.models import GCNClassification
from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.engines.classification_gcn import create_classification_gcn_evaluator, \
    create_classification_gcn_trainer
from vscvs.utils.data import prepare_batch_graph
from vscvs.decorators import kwargs_parameter_dict


def classification_gcn(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a classification GCN.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a classification GCN
    :type: `cls.__class__`
    """
    class Trainer(cls):
        """
        Trainer for a class classification GCN that uses only image classes and batch clique graphs where vertex weights
        correspond to word vector distances between image class labels.
        """
        def __init__(self, *args, dataset_name=None, processes=None, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractTrainer and EarlyStoppingMixin arguments
            :type: tuple
            :param dataset_name: the name of the Dataset to be used for training
            :type: str
            :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then
            `os.cpu_count()` will be used.
            :type: int or None
            :param kwargs: AbstractTrainer and EarlyStoppingMixin keyword arguments
            :type: dict
            """
            self.dataset_name = dataset_name
            self.processes = processes
            super().__init__(*args, dataset_name = self.dataset_name, **kwargs)

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

    return Trainer


@kwargs_parameter_dict
def train_classification_gcn(*args, optimizer_decorator=None, **kwargs):
    """
    Train a ClassificationGCN image classifier.
    :param args: ClassificationGCNTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: ClassificationGCNTrainer keyword arguments
    :type: dict
    """
    @classification_gcn
    @optimizer_decorator
    class ClassificationGCNTrainer(AbstractTrainer):
        pass
    trainer = ClassificationGCNTrainer(*args, **kwargs)
    trainer.run()
