__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


""" Trainer class decorators with the implementation of common features, such as common optimizers. """


from typing import Callable

from adabound import AdaBound
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import RMSprop


class OptimizerMixin:
    """
    Base class for optimizers.
    """

    def __init__(self, *args, learning_rate=0.01, **kwargs):
        """
        :param args: arguments for additional mixins
        :type: Tuple
        :param learning_rate: learning rate for optimizers
        :type: float
        :param kwargs: keyword arguments for additional mixins
        :type: Dict
        """
        self.learning_rate = learning_rate
        super().__init__(*args, **kwargs)


class GANOptimizerMixin:
    """
    Base class for the optimizers of a GAN Trainer. These Trainers require two optimizers, one for the generator and
    another for the discriminator. This Mixin is meant to be used with a normal OptimizerMixin.
    """

    _optimizer: Callable
    discriminator: Module
    generator: Module

    @property
    def optimizer(self):
        """
        :override: return the optimizers for the two adversarial models.
        :return: the optimizers for the generator and discriminator model modules.
        :type: Tuple[torch.optim.Optimizer, torch.optim.Optimizer]
        """
        return self._optimizer(self.generator.parameters()), self._optimizer(self.discriminator.parameters())


class AdaBoundOptimizerMixin(OptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [AdaBound](https://github.com/Luolc/AdaBound) optimizer.
    """

    def __init__(
        self,
        *args,
        betas=(0.9, 0.999),
        final_learning_rate=0.1,
        gamma=1e-3,
        epsilon=1e-08,
        weight_decay=0,
        amsbound=False,
        **kwargs
    ):
        """
        :param args: arguments for additional mixins
        :param betas: coefficients used for computing running averages of gradient and its square
        :type: Tuple<float, float>
        :type: Tuple
        :param final_learning_rate: final (SGD) learning rate.
        :type: float
        :param gamma: convergence speed of the bound functions.
        :type: float
        :param epsilon: term added to the denominator to improve numerical stability
        :type: float
        :param weight_decay: weight decay for L2 penalty
        :type: float
        :param amsbound: whether to use the AMSGrad variant of this algorithm from the paper [Adaptive Gradient Methods
        with Dynamic Bound of Learning Rate]( https://openreview.net/forum?id=Bkg3g2R9FX)
        :type: bool
        :param kwargs: keyword arguments for additional mixins
        :type: Dict
        """
        self.betas = betas
        self.final_learning_rate = final_learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsbound = amsbound
        super().__init__(*args, **kwargs)

    def _optimizer(self, parameters):
        return AdaBound(
            parameters,
            lr=self.learning_rate,
            betas=self.betas,
            final_lr=self.final_learning_rate,
            gamma=self.gamma,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            amsbound=self.amsbound,
        )


class AdamOptimizerMixin(OptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) optimizer.
    """

    def __init__(self, *args, betas=(0.9, 0.999), epsilon=1e-08, weight_decay=0, amsgrad=False, **kwargs):
        """
        :param args: arguments for additional mixins
        :type: Tuple
        :param betas: coefficients used for computing running averages of gradient and its square
        :type: Tuple<float, float>
        :param epsilon: term added to the denominator to improve numerical stability
        :type: float
        :param weight_decay: weight decay for L2 penalty
        :type: float
        :param amsgrad: whether to use the AMSGrad variant of this algorithm from the paper [On the Convergence of
        Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        :type: bool
        :param kwargs: keyword arguments for additional mixins
        :type: Dict
        """
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        super().__init__(*args, **kwargs)

    def _optimizer(self, parameters):
        return Adam(
            parameters,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class AdamWOptimizerMixin(AdamOptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [AdamW](https://pytorch.org/docs/stable/optim.html#torch.optim.AdamW) optimizer.
    """

    def _optimizer(self, parameters):
        return AdamW(
            parameters,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class RMSpropOptimizerMixin(OptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with an
    [RMSprop](https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop) optimizer.
    """

    def __init__(self, *args, alpha=0.99, epsilon=1e-08, weight_decay=0, momentum=0, centered=False, **kwargs):
        """
        :param args: arguments for additional mixins
        :type: Tuple
        :param alpha: the smoothing constant.
        :type: float
        :param epsilon: term added to the denominator to improve numerical stability
        :type: float
        :param weight_decay: weight decay for L2 penalty
        :type: float
        :param momentum: momentum factor.
        :type: float
        :param centered: whether to compute the centered RMSProp (gradient normalized by an estimation of its variance).
        :type: bool
        :param kwargs: keyword arguments for additional mixins
        :type: Dict
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        super().__init__(*args, **kwargs)

    def _optimizer(self, parameters):
        return RMSprop(
            parameters,
            lr=self.learning_rate,
            alpha=self.alpha,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )


class SGDOptimizerMixin(OptimizerMixin):
    """
    Trainer mixin for creating Trainer classes that override the `AbstractTrainer`'s `optimizer` property with a
    [Stochastic Gradient Descent (SGD)](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) optimizer.
    """

    def __init__(self, *args, momentum=0.8, **kwargs):
        """
        :param args: arguments for additional mixins
        :type: Tuple
        :param momentum: momentum parameter for SGD optimizer
        :type: float
        :param kwargs: keyword arguments for additional mixins
        :type: Dict
        """
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    def _optimizer(self, parameters):
        return SGD(parameters, lr=self.learning_rate, momentum=self.momentum)
