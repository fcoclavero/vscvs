__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Ignite trainer for a siamese network. """


from ignite.metrics import Loss
from torchvision.models import resnet50, resnext50_32x4d

from src.loss_functions import ContrastiveLoss
from src.models import ConvolutionalNetwork, SiameseNetwork
from src.trainers.abstract_trainer import AbstractTrainer
from src.trainers.engines.siamese import create_siamese_evaluator, create_siamese_trainer
from src.utils.decorators import kwargs_parameter_dict


def siamese(cls):
    """
    Class decorator for creating Trainer classes with the common options needed for a siamese architecture.
    :param cls: a Trainer class
    :type: AbstractTrainer subclass
    :return: `cls`, but implementing the common options for training a siamese architecture
    :type: `cls.__class__`
    """
    class Trainer(cls):
        """
        Trainer for a siamese network.
        """
        def __init__(self, *args, embedding_network=None, margin=1.0, **kwargs):
            """
            Trainer constructor.
            :param args: AbstractTrainer arguments
            :type: tuple
            :param embedding_network: the model to be used for each branch of the siamese architecture. The same
            architecture will be used for embedding each image pair, and weights will be shared.
            :type: torch.nn.Module
            :param margin: parameter for the contrastive loss, defining the acceptable threshold for considering the
            embeddings of two examples as dissimilar.
            :type: float
            :param kwargs: AbstractTrainer keyword arguments
            :type: dict
            """
            self.embedding_network = embedding_network
            self.margin = margin
            super().__init__(*args, **kwargs)

        @property
        def initial_model(self):
            return SiameseNetwork(self.embedding_network, self.embedding_network)

        @property
        def loss(self):
            return ContrastiveLoss(margin=self.margin)

        @property
        def trainer_id(self):
            return 'Siamese{}'.format(self.embedding_network.__class__.__name__)

        def _create_evaluator_engine(self):
            return create_siamese_evaluator(self.model, metrics={'loss': Loss(self.loss)}, device=self.device)

        def _create_trainer_engine(self):
            return create_siamese_trainer(self.model, self.optimizer, self.loss, device=self.device)

    return Trainer


@kwargs_parameter_dict
def train_siamese_cnn(*args, optimizer_decorator=None, **kwargs):
    """
    Train a Siamese CNN architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    @optimizer_decorator
    class SiameseTrainer(AbstractTrainer):
        pass
    trainer = SiameseTrainer(*args, embedding_network=ConvolutionalNetwork(), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnet(*args, optimizer_decorator=None, **kwargs):
    """
    Train a Siamese ResNet architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    @optimizer_decorator
    class SiameseTrainer(AbstractTrainer):
        pass
    trainer = SiameseTrainer(*args, embedding_network=resnet50(), **kwargs)
    trainer.run()


@kwargs_parameter_dict
def train_siamese_resnext(*args, optimizer_decorator=None, **kwargs):
    """
    Train a Siamese ResNext architecture.
    :param args: SiameseTrainer arguments
    :type: tuple
    :param optimizer_decorator: class decorator for creating Trainer classes that override the `AbstractTrainer`'s
    `optimizer` property with a specific optimizer.
    :type: function
    :param kwargs: SiameseTrainer keyword arguments
    :type: dict
    """
    @siamese
    @optimizer_decorator
    class SiameseTrainer(AbstractTrainer):
        pass
    trainer = SiameseTrainer(*args, embedding_network=resnext50_32x4d(), **kwargs)
    trainer.run()
