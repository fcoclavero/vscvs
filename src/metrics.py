class Metric:
    """
    Common interface for metric calculations.
    """
    def __str__(self):
        raise NotImplementedError

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


class Accuracy(Metric):
    """
    Simple accuracy metric: (TP + TN) / (TP + TN + FP + FN).
    """
    def __str__(self):
        return '%s: %s' % (self.name, self.value)

    def __init__(self):
        super(Accuracy, self).__init__()
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        prediction = outputs.argmax(1)
        self.correct += (prediction == target).sum().float()
        self.total += len(target)
        return self.value

    @property
    def name(self):
        return 'accuracy'

    @property
    def value(self):
        return 100 * float(self.correct) / self.total

    def reset(self):
        self.correct = 0
        self.total = 0