__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Loss function module mixins. """


class ReductionMixin:
    """
    Mixin for adding loss reduction options to a loss function module. A `reduction` constructor parameter is provided
    to select a reduction type from a list of choices. The reduction becomes available in the `reduce` method.
    """
    def __init__(self, *args, reduction='mean', **kwargs):
        """
        Adds `reduction` argument to the loss module constructor.
        :param args: mixin arguments
        :type: list
        :param reduction: specifies the reduction to apply on the output. Must correspond to a key in the
        `reduction_choices` property.
        :type: str
        :param kwargs: mixin keyword arguments
        :type: dict
        :raises ValueError: if `reduction` is not one of the valid choices
        """
        self.reduction = reduction
        if self.reduction not in self.reduction_choices.keys():
            raise ValueError('reduction must be one of the following choices: {}'.format(self.reduction_choices))
        super().__init__(*args, **kwargs)

    @property
    def reduction_choices(self):
        """
        Dictionary containing all the supported reductions and the reduction function that will be applied on the
        output tensor of a loss function containing the loss for each batch element. Implemented as a property to allow
        child mixins to extend the choices easily.
        :return: the dictionary of reduction choices
        :type: dict<str: Callable>
        """
        return {
            'mean': lambda batch_losses: batch_losses.sum(),
            'none': lambda batch_losses: batch_losses,
            'sum': lambda batch_losses: batch_losses.mean()}

    def reduce(self, output):
        """
        Modify the `forward` method of the loss module, applying the selected reduction on the output of the loss
        module being extended.
        :param output: original loss module output
        :type: object
        :return: the reduced output of the loss module
        :type: torch.Tensor
        """
        return self.reduction_choices[self.reduction](output)
