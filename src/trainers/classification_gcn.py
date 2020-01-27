__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GCN image label classifier, using binary or one-hot encodings as image feature vectors. """


from ignite.metrics import Accuracy, Loss, Recall, TopKCategoricalAccuracy
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.datasets import get_dataset
from src.models import ClassificationGCN
from src.trainers.abstract_trainers import AbstractTrainer
from src.trainers.engines.classification_gcn import create_classification_gcn_evaluator, \
    create_classification_gcn_trainer
from src.utils.data import prepare_batch_graph
from src.utils.decorators import kwargs_parameter_dict


class ClassificationGCNTrainer(AbstractTrainer):
    """
    Trainer for a class classification GCN that uses only image classes and batch clique graphs where vertex weights
    correspond to word vector distances between image class labels.
    """
    def __init__(self, *args, dataset_name=None, learning_rate=.01, weight_decay=5e-4, processes=None, **kwargs):
        """
        Trainer constructor.
        :param args: AbstractTrainer and EarlyStoppingMixin arguments
        :type: tuple
        :param dataset_name: the name of the Dataset to be used for training
        :type: str
        :param learning_rate: learning rate for Adam optimizer
        :type: float
        :param weight_decay: weight_decay parameter for Adam optimizer
        :type: float
        :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then
        `os.cpu_count()` will be used.
        :type: int or None
        :param kwargs: AbstractTrainer and EarlyStoppingMixin keyword arguments
        :type: dict
        """
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.processes = processes
        super().__init__(*args, dataset_name = self.dataset_name, **kwargs)

    @property
    def initial_model(self):
        dataset = get_dataset(self.dataset_name)
        return ClassificationGCN(len(dataset.classes), 11)

    @property
    def loss(self):
        return CrossEntropyLoss()

    @property
    def optimizer(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay}

    @property
    def trainer_id(self):
        return 'classification_gcn'

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
def train_classification_gcn(*args, **kwargs):
    """
    Train a ClassificationGCN image classifier.
    :param args: ClassificationGCNTrainer arguments
    :type: tuple
    :param kwargs: ClassificationGCNTrainer keyword arguments
    :type: dict
    """
    trainer = ClassificationGCNTrainer(*args, **kwargs)
    trainer.run()
