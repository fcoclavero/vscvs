__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"

from torch.nn import TripletMarginLoss

""" Ignite trainer for a Triplet Network architecture. """


from abc import ABC
from typing import Callable

from overrides import overrides

from vscvs.decorators import kwargs_parameter_dict
from vscvs.loss_functions import TripletLoss
from vscvs.metrics import AccuracyTriplets
from vscvs.metrics import AverageDistancesTriplets
from vscvs.metrics import LossTriplets
from vscvs.models import CNNNormalized
from vscvs.models import ResNetNormalized
from vscvs.models import ResNextNormalized
from vscvs.models import TripletSharedPositiveNegative

from .abstract_trainer import AbstractTrainer
from .engines.triplet import create_triplet_evaluator
from .engines.triplet import create_triplet_trainer


class AbstractTripletTrainer(AbstractTrainer, ABC):
    """
    Abstract class for creating Trainer classes with the common options needed for a triplet architecture.
    """

    def __init__(
        self, *args, anchor_network=None, positive_negative_network=None, loss_reduction="mean", margin=0.2, **kwargs
    ):
        """
        :param args: Trainer arguments
        :type: Tuple
        :param anchor_network: the model to be used for computing anchor image embeddings.
        :type: torch.nn.Module
        :param positive_negative_network: the model to be used for computing image embeddings for the positive
        and negative elements in each triplet.
        :type: torch.nn.Module
        :param loss_reduction: reduction to apply to batch element loss values to obtain the loss for the whole batch.
`       Must correspond to a valid reduction for the `TripletLoss`.
        :type: str
        :param margin: parameter for the triplet loss, defining the minimum acceptable difference between the
        distance from the anchor element to the negative, and the distance from the anchor to the negative.
        :type: float
        :param kwargs: Trainer keyword arguments
        :type: Dict
        """
        self.anchor_network = anchor_network
        self.positive_negative_network = positive_negative_network
        self.loss_reduction = loss_reduction
        self.margin = margin
        super().__init__(*args, **kwargs)

    @property
    @overrides
    def initial_model(self):
        return TripletSharedPositiveNegative(self.anchor_network, self.positive_negative_network)

    @property
    @overrides
    def loss(self):
        # return TripletLoss(margin=self.margin, reduction=self.loss_reduction)
        return TripletMarginLoss(margin=self.margin, p=2.0)

    @property
    @overrides
    def trainer_id(self):
        return "Triplet{}".format(self.anchor_network.__class__.__name__)

    @overrides
    def _create_evaluator_engine(self):
        average_distances = AverageDistancesTriplets()
        return create_triplet_evaluator(
            self.model,
            device=self.device,
            metrics={
                "Accuracy": AccuracyTriplets(),
                "Average Distance/positive": average_distances[0],
                "Average Distance/negative": average_distances[1],
                "Loss": LossTriplets(self.loss),
            },
        )

    @overrides
    def _create_trainer_engine(self):
        return create_triplet_trainer(self.model, self.optimizer, self.loss, device=self.device)


@kwargs_parameter_dict
def train_triplet_cnn(*args, optimizer_mixin=None, **kwargs):
    """
    Train a Triplet CNN architecture.
    :param args: TripletTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: TripletTrainer keyword arguments
    :type: Dict
    """

    class TripletTrainer(optimizer_mixin, AbstractTripletTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm

    trainer = TripletTrainer(
        *args,
        anchor_network=CNNNormalized(out_features=250),
        positive_negative_network=CNNNormalized(out_features=250),
        **kwargs
    )
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnet(*args, optimizer_mixin=None, **kwargs):
    """
    Train a Triplet ResNet architecture.
    :param args: TripletTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: TripletTrainer keyword arguments
    :type: Dict
    """

    class TripletTrainer(optimizer_mixin, AbstractTripletTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm

    trainer = TripletTrainer(
        *args,
        anchor_network=ResNetNormalized(out_features=250, pretrained=True),
        positive_negative_network=ResNetNormalized(out_features=250),
        **kwargs
    )
    trainer.run()


@kwargs_parameter_dict
def train_triplet_resnext(*args, optimizer_mixin=None, **kwargs):
    """
    Train a Triplet ResNext architecture.
    :param args: TripletTrainer arguments
    :type: Tuple
    :param optimizer_mixin: Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: vscvs.trainers.mixins.OptimizerMixin
    :param kwargs: TripletTrainer keyword arguments
    :type: Dict
    """

    class TripletTrainer(optimizer_mixin, AbstractTripletTrainer):
        _optimizer: Callable  # type hinting `_optimizer` defined in `optimizer_mixin`, but is not recognized by PyCharm

    trainer = TripletTrainer(
        *args,
        anchor_network=ResNextNormalized(out_features=250, pretrained=True),
        positive_negative_network=ResNextNormalized(out_features=250),
        **kwargs
    )
    trainer.run()
