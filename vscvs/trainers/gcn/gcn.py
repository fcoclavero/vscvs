__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Ignite trainer for a GCN semi-supervised classifier. """


from abc import ABC
from typing import Callable

from ignite.metrics import Accuracy
from ignite.metrics import Loss
from ignite.metrics import Recall
from ignite.metrics import TopKCategoricalAccuracy
from overrides import overrides
from torch.nn import CrossEntropyLoss

from vscvs.datasets import get_dataset
from vscvs.decorators import kwargs_parameter_dict
from vscvs.models import GCNClassification

from vscvs.trainers.abstract_trainer import AbstractTrainer
from vscvs.trainers.gcn.engines.hog import create_hog_gcn_evaluator
from vscvs.trainers.gcn.engines.hog import create_hog_gcn_trainer


class AbstractGCNTrainer(AbstractTrainer, ABC):
    """
    Abstract Trainer for a semi-supervised node classifier using a GCN over batch clique graphs where nodes correspond to batch
    images, node feature vectors correspond to batch image HOG descriptors, and vertex weights corresponds to image
    label word vector distances.
    """

    def __init__(
        self,
        *args,
        dataset_name=None,
        in_channels=3,
        cell_size=8,
        bins=9,
        signed_gradients=False,
        processes=None,
        **kwargs
    ):
        """
        :param args: Trainer arguments
        :type: Tuple
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
        :type: bool
        :param processes: number of parallel workers to be used for creating batch graphs. If `None`, then
        `os.cpu_count()` will be used.
        :type: int
        :param kwargs: Trainer keyword arguments
        :type: Dict
        """
        self.dataset_name = dataset_name
        super().__init__(*args, dataset_name=self.dataset_name, **kwargs)

    @property
    @overrides
    def initial_model(self):
        dataset = get_dataset(self.dataset_name)
        image_dimension = dataset[0][0].shape[1]
        return GCNClassification(
            num_classes=,
            in_channels=,
        )

    @property
    @overrides
    def loss(self):
        return CrossEntropyLoss()

    @property
    @overrides
    def trainer_id(self):
        return "GCN"

    @overrides
    def _create_evaluator_engine(self):
        return create_hog_gcn_evaluator(
            self.model,
            device=self.device,
            metrics={
                "Accuracy": Accuracy(),
                "Loss": Loss(self.loss),
                "Recall": Recall(average=True),
                "Top K Categorical Accuracy": TopKCategoricalAccuracy(k=10),
            },
        )

    @overrides
    def _create_trainer_engine(self):
        return create_hog_gcn_trainer(self.model, self.optimizer, self.loss, device=self.device)


@kwargs_parameter_dict
def train_hog_gcn(*args, optimizer_mixin=None, **kwargs):
    """
    Train a HOGGCN image classifier.
    :param args: HOGGCNTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: HOGGCNTrainer keyword arguments
    :type: Dict
    """

    class HOGGCNTrainer(optimizer_mixin, AbstractHOGGCNTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm

    trainer = HOGGCNTrainer(*args, **kwargs)
    trainer.run()
