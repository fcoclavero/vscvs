__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


""" Utility classes for computing different metrics for binary classification on the go. """


from sklearn.metrics import confusion_matrix

from src.metrics.common import AbstractAccuracy, AbstractMetric


class Accuracy(AbstractAccuracy):
    """
    Accuracy for binary classification.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, output, target, loss):
        prediction = (output > 0.5).float()
        self.correct += (prediction == target).sum().float()
        self.total += len(target)
        return self.value


class Precision(AbstractMetric):
    """
    Simple precision metric: TP / (TP + FP).
    Measures the percentage of positives predictions that are correct.
    """
    def __init__(self):
        super().__init__()
        self.true_positive = 0
        self.false_positive = 0

    def __call__(self, output, target, loss):
        prediction = output > 0.5
        tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()
        self.true_positive += tp
        self.false_positive +=  fp
        return self.value

    @property
    def name(self):
        return 'precision'

    @property
    def value(self):
        return 100 * float(self.true_positive) / (self.true_positive + self.false_positive)

    def reset(self):
        self.true_positive = 0
        self.false_positive = 0


class Recall(AbstractMetric):
    """
    Simple recall metric: TP / (TP + FN).
    Measures how well we find all the positives.
    """
    def __init__(self):
        super().__init__()
        self.true_positive = 0
        self.false_negative = 0

    def __call__(self, output, target, loss):
        prediction = output > 0.5
        tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()
        self.true_positive += tp
        self.false_negative += fn
        return self.value

    @property
    def name(self):
        return 'recall'

    @property
    def value(self):
        return 100 * float(self.true_positive) / (self.true_positive + self.false_negative)

    def reset(self):
        self.true_positive = 0
        self.false_negative = 0


class F1(AbstractMetric):
    """
    F1 metric: 2 * (precision * recall) / (precision + recall).
    """
    def __init__(self):
        super().__init__()
        self.precision = Precision()
        self.recall = Recall()

    def __call__(self, output, target, loss):
        self.precision(output, target, loss)
        self.recall(output, target, loss)
        return self.value

    @property
    def name(self):
        return 'f1'

    @property
    def value(self):
        precision = self.precision.value
        recall = self.recall.value
        return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    def reset(self):
        self.precision = Precision()
        self.recall = Recall()