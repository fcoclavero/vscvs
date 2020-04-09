__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Trainer class decorators with the implementation of common features, such as common optimizers. """


from torch.nn import Module
from torch.optim import Adam, SGD


class ModuleMixin:
    """
    Utility class that type hints `AbstractTrainer` methods that will be available to the mixins in this package, as
    they are meant to be used in multiple inheritance with `vscvs.trainers.abstract_trainer.AbstractTrainer`.
    """
    model: Module


class AdamOptimizerMixin(ModuleMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) optimizer.
    """
    def __init__(self, *args, learning_rate=0.001, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0,
                 amsgrad=False, **kwargs):
        """
        Trainer constructor that receives the optimizer parameters.
        :param args: arguments for additional mixins
        :type: tuple
        :param learning_rate: learning rate for optimizers
        :type: float
        :param betas: coefficients used for computing running averages of gradient and its square
        :type: Tuple<float, float>
        :param epsilon: term added to the denominator to improve numerical stability
        :type: float
        :param weight_decay: weight decay for L2 penalty
        :type: float
        :param amsgrad: whether to use the AMSGrad variant of this algorithm from the paper [On the Convergence of
        Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        :type: boolean
        :param kwargs: keyword arguments for additional mixins
        :type: dict
        """
        self.learning_rate = learning_rate
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        super().__init__(*args, **kwargs)

    @property
    def optimizer(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas, eps=self.epsilon,
                    weight_decay=self.weight_decay, amsgrad=self.amsgrad)


class SGDOptimizerMixin(ModuleMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with a
    [Stochastic Gradient Descent (SGD)](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) optimizer.
    """
    def __init__(self, *args, learning_rate=.01, momentum=.8, **kwargs):
        """
        Trainer constructor that receives the optimizer parameters.
        :param args: arguments for additional mixins
        :type: tuple
        :param learning_rate: learning rate for optimizers
        :type: float
        :param momentum: momentum parameter for SGD optimizer
        :type: float
        :param kwargs: keyword arguments for additional mixins
        :type: dict
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    @property
    def optimizer(self):
        return SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
