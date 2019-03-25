import numpy as np

from sklearn.metrics import confusion_matrix

from src.metrics.common import AbstractMetric, AbstractAccuracy


class Accuracy(AbstractAccuracy):
    """
    Accuracy for multi-class classification.
    """
    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, output, target, loss):
        prediction = output.argmax(1)
        self.correct += (prediction == target).sum().float()
        self.total += len(target)
        return self.value


class MeanAveragePrecision(AbstractMetric):
    """
    Mean precision over all classes: mean(TP_i / (TP_i + FP_i)) for i in classes.
    """
    def __init__(self):
        super(MeanAveragePrecision, self).__init__()
        self.average_precision = 0
        self.total = 0

    def __call__(self, output, target, loss):
        prediction = output.argmax(1)
        cm = confusion_matrix(target, prediction)
        self.average_precision += np.mean(np.diag(cm) / np.sum(cm, axis=0)) # TODO: watch out for zero divide when a class is not predicted
        self.total += 1
        return self.value

    @property
    def name(self):
        return 'mAP'

    @property
    def value(self):
        return self.average_precision / self.total

    def reset(self):
        self.average_precision = 0
        self.total = 0