__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a GCN image label classifier, using HOG feature vectors for images. """


from ignite.metrics import Accuracy, Loss, Recall, TopKCategoricalAccuracy
from torch.nn import CrossEntropyLoss

from src.datasets import get_dataset
from src.models import HOGGCN
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.engines.hog_gcn import create_hog_gcn_evaluator, create_hog_gcn_trainer
from src.utils.data import prepare_batch
from src.utils.decorators import kwargs_parameter_dict


def hog_gcn(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a HOG GCN model.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a HOG GCN model
    :type: `cls.__class__`
    """
    class Trainer(cls):
        """
        Trainer for an image label classifier using a GCN over batch clique graphs where nodes correspond to batch
        images, node feature vectors correspond to batch image HOG descriptors, and vertex weights corresponds to image label
        word vector distances.
        """
        def __init__(self, *args, dataset_name=None, in_channels=3, cell_size=8, bins=9, signed_gradients=False,
                     processes=None, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractTrainer arguments
            :type: tuple
            :param dataset_name: the name of the Dataset to be used for training
            :type: str
            :param in_channels: the number of channels for inputs.
            :type: int
            :param cell_size: the image will be divided into cells of the specified size, and the histogram of gradients is
            calculated in each one. Received as a tuple indicating the x and y dimensions of the cell, measured in pixels.
            :type: int
            :param bins: number of bins for the histogram of each cell.
            :type: int
            :param signed_gradients: gradients are represented using its angle and magnitude. Angles can be expressed
            using values between 0 and 360 degrees or between 0 and 180 degrees. If the latter are used, we call the
            gradient “unsigned” because a gradient and it’s negative are represented by the same numbers. Empirically it has
            been shown that unsigned gradients work better than signed gradients for tasks such as pedestrian detection.
            :type: boolean
            :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then
            `os.cpu_count()` will be used.
            :type: int or None
            :param kwargs: AbstractTrainer keyword arguments
            :type: dict
            """
            self.dataset_name = dataset_name
            self.in_channels = in_channels
            self.cell_size = cell_size
            self.bins = bins
            self.signed_gradients = signed_gradients
            self.processes = processes
            super().__init__(*args, dataset_name = self.dataset_name, **kwargs)

        @property
        def initial_model(self):
            dataset = get_dataset(self.dataset_name)
            image_dimension = dataset[0][0].shape[1]
            return HOGGCN(self.dataset.classes_dataframe, image_dimension, self.in_channels, self.cell_size, self.bins,
                          self.signed_gradients, self.processes)

        @property
        def loss(self):
            return CrossEntropyLoss()

        @property
        def serialized_checkpoint(self):
            return {**super().serialized_checkpoint, 'in_channels': self.in_channels, 'cell_size': self.cell_size,
                    'bins': self.bins, 'signed_gradients': self.signed_gradients, 'processes': self.processes}

        @property
        def trainer_id(self):
            return 'HOGGCN'

        def _create_evaluator_engine(self):
            return create_hog_gcn_evaluator(
                prepare_batch, self.model, self.dataset.classes_dataframe, device=self.device, processes=self.processes,
                metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss), 'recall': Recall(average=True),
                         'top_k_categorical_accuracy': TopKCategoricalAccuracy(k=10)})

        def _create_trainer_engine(self):
            return create_hog_gcn_trainer(prepare_batch, self.model, self.dataset.classes_dataframe, self.optimizer,
                                          self.loss, device=self.device, processes=self.processes)

    return Trainer


@kwargs_parameter_dict
def train_hog_gcn(*args, optimizer_decorator=None, **kwargs):
    """
    Train a HOGGCN image classifier.
    :param args: HOGGCNTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: HOGGCNTrainer keyword arguments
    :type: dict
    """
    @hog_gcn
    @optimizer_decorator
    class HOGGCNTrainer(AbstractTrainer):
        pass
    trainer = HOGGCNTrainer(*args, **kwargs)
    trainer.run()
