import numpy as np

from sklearn.metrics import confusion_matrix

from src.metrics.common import AbstractMetric, AbstractAccuracy


class Accuracy(AbstractAccuracy):
    """
    Accuracy for multi-class classification.
    """
    def __call__(self, output, target, loss):
        prediction = output.argmax(1)
        self.correct += (prediction == target).sum().float()
        self.total += len(target)
        return self.value


class AveragePrecision(AbstractMetric):
    """
    Average precision over all classes: mean(TP_i / (TP_i + FP_i)) for i in classes.
    """
    def __init__(self):
        super(AveragePrecision, self).__init__()
        self.average_precision = 0

    def __call__(self, output, target, loss):
        prediction = output.argmax(1)
        # cm[i][i] are the tp for class i, sum of column i are all the positives for class i
        cm = confusion_matrix(target, prediction)
        # Map over true positives and positives and compute precision as tp / p.
        precisions = map(lambda tp, p: 0.0 if p == 0 else tp / p, np.diag(cm), np.sum(cm, axis=0))
        self.average_precision = np.mean(list(precisions)) # Get the mean over all classes
        return self.value

    @property
    def name(self):
        return 'AP'

    @property
    def value(self):
        return self.average_precision

    def reset(self):
        self.average_precision = 0


class MeanAveragePrecision(AveragePrecision):
    """
    Mean, over all experiments, of the average precision of all classes: mean(TP_i / (TP_i + FP_i)) for i in classes.
    """
    def __init__(self):
        super(MeanAveragePrecision, self).__init__()
        self.sum = 0.0
        self.total = 0.0

    def __call__(self, output, target, loss):
        super(MeanAveragePrecision, self).__call__(output, target, loss)
        self.sum += self.average_precision
        self.total += 1
        return self.value

    @property
    def name(self):
        return 'mAP'

    @property
    def value(self):
        return 0.0 if self.total == 0 else self.sum / self.total

    def reset(self):
        super(MeanAveragePrecision, self).reset()
        self.sum = 0
        self.total = 0