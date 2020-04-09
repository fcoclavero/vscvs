__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Trainer class decorators with the implementation of common features, such as common optimizers. """


from torch.nn import Module
from torch.optim import Adam, AdamW, RMSprop, SGD


class OptimizerMixin:
    """
    Base class for optimizers. It also type hints `AbstractTrainer` methods that will be available to the mixins in this
    package, as they are to be used in multiple inheritance with `vscvs.trainers.abstract_trainer.AbstractTrainer`.
    """
    model: Module

    def __init__(self, *args, learning_rate=.01, **kwargs):
        """
        Trainer constructor that receives the optimizer parameters.
        :param args: arguments for additional mixins
        :type: tuple
        :param learning_rate: learning rate for optimizers
        :type: float
        :param kwargs: keyword arguments for additional mixins
        :type: dict
        """
        self.learning_rate = learning_rate
        super().__init__(*args, **kwargs)


class AdamOptimizerMixin(OptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) optimizer.
    """
    def __init__(self, *args, betas=(.9, .999), epsilon=1e-08, weight_decay=0, amsgrad=False, **kwargs):
        """
        Trainer constructor that receives the optimizer parameters.
        :param args: arguments for additional mixins
        :type: tuple
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
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        super().__init__(*args, **kwargs)

    @property
    def optimizer(self):
        return Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas, eps=self.epsilon,
                    weight_decay=self.weight_decay, amsgrad=self.amsgrad)


class AdamWOptimizerMixin(AdamOptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [AdamW](https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW) optimizer.
    """
    @property
    def optimizer(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate, betas=self.betas, eps=self.epsilon,
                     weight_decay=self.weight_decay, amsgrad=self.amsgrad)


class RMSpropOptimizerMixin(OptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [RMSprop](https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop) optimizer.
    """
    def __init__(self, *args, alpha=.99, epsilon=1e-08, weight_decay=0, momentum=0, centered=False, **kwargs):
        """
        Trainer constructor that receives the optimizer parameters.
        :param args: arguments for additional mixins
        :type: tuple
        :param alpha: the smoothing constant.
        :type: float
        :param epsilon: term added to the denominator to improve numerical stability
        :type: float
        :param weight_decay: weight decay for L2 penalty
        :type: float
        :param momentum: momentum factor.
        :type: float
        :param centered: whether to compute the centered RMSProp (gradient normalized by an estimation of its variance).
        :type: boolean
        :param kwargs: keyword arguments for additional mixins
        :type: dict
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        super().__init__(*args, **kwargs)

    @property
    def optimizer(self):
        return RMSprop(self.model.parameters(), lr=self.learning_rate, alpha=self.alpha, eps=self.epsilon,
                       weight_decay=self.weight_decay, momentum=self.momentum, centered=self.centered)


class SGDOptimizerMixin(OptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with a
    [Stochastic Gradient Descent (SGD)](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) optimizer.
    """
    def __init__(self, *args, momentum=.8, **kwargs):
        """
        Trainer constructor that receives the optimizer parameters.
        :param args: arguments for additional mixins
        :type: tuple
        :param momentum: momentum parameter for SGD optimizer
        :type: float
        :param kwargs: keyword arguments for additional mixins
        :type: dict
        """
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    @property
    def optimizer(self):
        return SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
