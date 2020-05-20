__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Loss function module mixins. """


class ReductionMixinMeta(type):
    """
    Custom metaclass for accessing the reduction definitions without needing to instance a ReductionMixin object.
    """
    def __init__(cls, *args, **kwargs):
        """
        :param args: default metaclass constructor arguments.
        :type: List
        :param kwargs: default metaclass constructor keyword arguments.
        :type: Dict
        """
        super().__init__(*args, **kwargs)
        cls._reductions = {
            'mean': lambda batch_losses: batch_losses.mean(),
            'none': lambda batch_losses: batch_losses,
            'sum': lambda batch_losses: batch_losses.sum()}

    @property
    def reductions(cls):
        """
        Dictionary containing all the supported reductions and the reduction function that will be applied on the
        output tensor of a loss function containing the loss for each batch element. Implemented as a property to allow
        child mixins to extend the choices easily.
        :return: the dictionary of reduction choices. The reduction functions take a tensor of batch losses and return
        their reduction, which can be either a float or another tensor.
        :type: Dict[str, Callable[[torch.Tensor], Union[float, torch.Tensor]]]
        """
        return cls._reductions

    @property
    def reduction_choices(cls):
        """
        List of the available reductions. These are the valid values for the `reduction` parameter in the constructor.
        :return: a list of the valid reductions.
        :type: List[str]
        """
        return list(cls.reductions.keys())


class ReductionMixin(metaclass=ReductionMixinMeta):
    """
    Mixin for adding loss reduction options to a loss function module. A `reduction` constructor parameter is provided
    to select a reduction type from a list of choices. The reduction becomes available in the `reduce` method.
    """
    def __init__(self, *args, reduction='mean', **kwargs):
        """
        :param args: mixin arguments
        :type: List
        :param reduction: specifies the reduction to apply on the output. Must correspond to a key in the
        `reduction_choices` property.
        :type: str
        :param kwargs: mixin keyword arguments
        :type: Dict
        :raises ValueError: if `reduction` is not one of the valid choices
        """
        self.reduction = reduction
        if self.reduction not in self.reduction_choices:
            raise ValueError('reduction must be one of the following choices: {}'.format(self.reductions))
        super().__init__(*args, **kwargs)

    def reduce(self, output):
        """
        Modify the `forward` method of the loss module, applying the selected reduction on the output of the loss
        module being extended.
        :param output: original loss module output
        :type: object
        :return: the reduced output of the loss module
        :type: torch.Tensor
        """
        # noinspection PyUnresolvedReferences
        return self.reductions[self.reduction](output) # `reduction` property is a dict, so it does have `__getitem__`

    """ Make reduction class properties available to class instances. """

    @property
    def reductions(self):
        return type(self).reductions

    @property
    def reduction_choices(self):
        return type(self).reduction_choices
