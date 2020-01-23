__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a siamese network. """


from ignite.metrics import Accuracy, Loss
from torch.optim import SGD
from torchvision.models import resnet50, resnext50_32x4d

from src.loss_functions import ContrastiveLoss
from src.models import ClassificationConvolutionalNetwork, SiameseNetwork
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.engines.siamese import create_siamese_evaluator, create_siamese_trainer
from src.utils.decorators import kwargs_parameter_dict


class SiameseTrainer(AbstractTrainer):
    """
    Trainer for a siamese network.
    """
    def __init__(self, *args, architecture_model=None, learning_rate=.01, margin=1.0, momentum=.8, **kwargs):
        """
        Trainer constructor.
        :param args: AbstractTrainer arguments
        :type: tuple
        :param architecture_model: the model to be used for each branch of the siamese architecture. The same
        architecture will be used for embedding each image pair, and weights will be shared.
        :type: torch.nn.Module
        :param learning_rate: learning rate for SGD optimizer
        :type: float
        :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the embeddings
        of two examples as dissimilar.
        :type: float
        :param momentum: momentum parameter for SGD optimizer
        :type: float
        :param kwargs: AbstractTrainer keyword arguments
        :type: dict
        """
        self.architecture_model = architecture_model
        self.margin = margin
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    @property
    def initial_model(self):
        return SiameseNetwork(self.architecture_model)

    @property
    def loss(self):
        return ContrastiveLoss(margin=self.margin)

    @property
    def optimizer(self):
        return SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    @property
    def serialized_checkpoint(self):
        return {**super().serialized_checkpoint, 'learning_rate': self.learning_rate, 'momentum': self.momentum}

    @property
    def trainer_id(self):
        return 'siamese {}'.format(self.architecture_model.__class__.__name__)

    def _create_evaluator_engine(self):
        return create_siamese_evaluator(self.model, metrics={'loss': Loss(self.loss)}, device=self.device)

    def _create_trainer_engine(self):
        return create_siamese_trainer(self.model, self.optimizer, self.loss, device=self.device)


@kwargs_parameter_dict
def train_siamese_cnn(*args, **kwargs):
    """
    Train a Siamese CNN architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    trainer = SiameseTrainer(*args, architecture_model = ClassificationConvolutionalNetwork(), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnet(*args, **kwargs):
    """
    Train a Siamese ResNet architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    trainer = SiameseTrainer(*args, architecture_model = resnet50(), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnext(*args, **kwargs):
    """
    Train a Siamese ResNext architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    trainer = SiameseTrainer(*args, architecture_model = resnext50_32x4d(), **kwargs)
    trainer.run()
