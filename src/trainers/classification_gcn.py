__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GCN image label classifier. """


from ignite._utils import convert_tensor
from ignite.metrics import Accuracy, Loss
from torch.nn import NLLLoss
from torch.optim import Adam

from src.datasets import get_dataset, get_dataset_classes_dataframe
from src.models.graph_convolutional_network import ClassificationGCN
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.engines.classification_gcn import *
from src.utils.data import batch_clique_graph


class ClassificationGCNTrainer(AbstractTrainer):
    """
    Trainer for a class classification GCN that uses only image classes and batch clique graphs where vertex weights
    correspond to word vector distances between image class labels.
    """
    def __init__(self, dataset_name, resume_date=None, train_validation_split=.8, batch_size=16, epochs=2, workers=6,
                 n_gpu=0, tag=None, learning_rate=.01, weight_decay=5e-4, processes=None, drop_last=False):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.processes = processes
        self.classes_dataframe = get_dataset_classes_dataframe(dataset_name)
        super().__init__(dataset_name, resume_date, train_validation_split, batch_size, epochs=epochs, workers=workers,
                         n_gpu=n_gpu, tag=tag, drop_last=drop_last)

    @property
    def initial_model(self):
        dataset = get_dataset(self.dataset_name)
        return ClassificationGCN(11, len(dataset.classes))

    @property
    def loss(self):
        return NLLLoss()

    @property
    def optimizer(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay}

    @property
    def trainer_id(self):
        return 'classification_gcn'

    def _prepare_batch(self, batch, device=None, non_blocking=False):
        """
        Prepare batch for training: pass to a device with options. Assumes data and labels are the first
        two parameters of each sample.
        :param batch: data to be sent to device.
        :type: list
        :param device: device type specification
        :type: str (optional) (default: None)
        :param non_blocking: if True and the copy is between CPU and GPU, the copy may run asynchronously
        :type: bool (optional)
        """
        graph = batch_clique_graph(batch, self.classes_dataframe, self.processes)
        graph.apply(
            lambda attr: convert_tensor(attr.float(), device=device, non_blocking=non_blocking), 'x', 'edge_attr')
        return graph

    def _create_evaluator_engine(self):
        return create_classification_gcn_evaluator(
            self._prepare_batch, self.model, device=self.device,
                metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss)})

    def _create_trainer_engine(self):
        return create_classification_gcn_trainer(
            self._prepare_batch, self.model, self.optimizer, self.loss, device=self.device)


def train_classification_gcn(dataset_name, resume_date=None, train_validation_split=.8, batch_size=16, epochs=2,
                             workers=4, n_gpu=0, tag=None, learning_rate=.01, weight_decay=5e-4, processes=None):
    """
    Trains a GCN to predict image labels using a GCN over batch clique graphs where nodes correspond to batch images and
    vertex weights corresponds to image label word vector distances.
    :param dataset_name: the name of the Dataset to be used for training
    :type: str
    :param resume_date: date of the trainer state to be resumed. Dates must have the following
    format: `%y-%m-%dT%H-%M`
    :type: str
    :param train_validation_split: proportion of the training set that will be used for actual
    training. The remaining data will be used as the validation set.
    :type: float
    :param batch_size: batch size during training
    :type: int
    :param epochs: the number of epochs used for training
    :type: int
    :param workers: number of workers for data_loader
    :type: int
    :param n_gpu: number of GPUs available. Use 0 for CPU mode
    :type: int
    :param tag: optional tag for model checkpoint and tensorboard logs
    :type: str
    :param learning_rate: learning rate for optimizers
    :type: float
    :param weight_decay: weight_decay parameter for Adam optimizer
    :type: float
    :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then `os.cpu_count()`
    will be used.
    :type: int or None
    """
    trainer = ClassificationGCNTrainer(dataset_name, resume_date, train_validation_split, batch_size, epochs,
                                       workers, n_gpu, tag, learning_rate, weight_decay, processes, drop_last=True)
    trainer.run()
