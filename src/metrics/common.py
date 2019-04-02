class AbstractMetric:
    """
    Common interface for metric calculations.
    """
    def __str__(self):
        return '%s: %s' % (self.name, self.value)

    def __init__(self):
        pass

    def __call__(self, output, target, loss):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class AbstractAccuracy(AbstractMetric):
    """
    Simple accuracy metric: (TP + TN) / (TP + TN + FP + FN).
    Measures the percentage of correct predictions. Not reliable in case of unbalanced classes.
    """
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0

    def __call__(self, output, target, loss):
        raise NotImplementedError

    @property
    def name(self):
        return 'accuracy'

    @property
    def value(self):
        return 100 * float(self.correct) / self.total

    def reset(self):
        self.correct = 0
        self.total = 0