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
        super().__init__()
        self.average_precision = 0

    def __call__(self, output, target, loss):
        prediction = output.argmax(1)
        # cm[i][i] is the number of tp for class i, the sum of column i (axis=0) is the total of class i predictions
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
    Mean, over all experiments, of the average precision of all classes.
    """
    def __init__(self):
        super().__init__()
        self.sum = 0.0
        self.total = 0.0

    def __call__(self, output, target, loss):
        super().__call__(output, target, loss)
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
        super().reset()
        self.sum = 0
        self.total = 0


class AverageRecall(AbstractMetric):
    """
    Average recall over all classes: mean(TP_i / (TP_i + FN_i)) for i in classes.
    """
    def __init__(self):
        super().__init__()
        self.average_recall = 0

    def __call__(self, output, target, loss):
        prediction = output.argmax(1)
        # cm[i][i] is the number of tp for class i, the sum of row i (axis=1) is the total of class i labeled targets
        cm = confusion_matrix(target, prediction)
        # Map over true positives and target positives and compute recall as tp / (tp + fn).
        recalls = map(lambda tp, p: 0.0 if p == 0 else tp / p, np.diag(cm), np.sum(cm, axis=1))
        self.average_recall = np.mean(list(recalls)) # Get the mean over all classes
        return self.value

    @property
    def name(self):
        return 'AR'

    @property
    def value(self):
        return self.average_recall

    def reset(self):
        self.average_recall = 0


class MeanAverageRecall(AverageRecall):
    """
    Mean, over all experiments, of the average recall of all classes.
    """
    def __init__(self):
        super().__init__()
        self.sum = 0.0
        self.total = 0.0

    def __call__(self, output, target, loss):
        super().__call__(output, target, loss)
        self.sum += self.average_recall
        self.total += 1
        return self.value

    @property
    def name(self):
        return 'mAR'

    @property
    def value(self):
        return 0.0 if self.total == 0 else self.sum / self.total

    def reset(self):
        super().reset()
        self.sum = 0
        self.total = 0


class AverageF1(AbstractMetric):
    """
    Average F1 metric over all classes: mean(2 * (precision_i * recall_i) / (precision_i + recall_i)) for i in classes.
    """
    def __init__(self):
        super().__init__()
        self.precision = AveragePrecision()
        self.recall = AverageRecall()

    def __call__(self, output, target, loss):
        self.precision(output, target, loss)
        self.recall(output, target, loss)
        return self.value

    @property
    def name(self):
        return 'F1'

    @property
    def value(self):
        precision = self.precision.value
        recall = self.recall.value
        return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    def reset(self):
        self.precision = AveragePrecision()
        self.recall = AverageRecall()


class MeanAverageF1(AbstractMetric):
    """
    Mean, over all experiments, of the average F1 of all classes.
    """
    def __init__(self):
        super().__init__()
        self.precision = MeanAveragePrecision()
        self.recall = MeanAverageRecall()

    def __call__(self, output, target, loss):
        self.precision(output, target, loss)
        self.recall(output, target, loss)
        return self.value

    @property
    def name(self):
        return 'aF1'

    @property
    def value(self):
        precision = self.precision.value
        recall = self.recall.value
        return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    def reset(self):
        self.precision = MeanAveragePrecision()
        self.recall = MeanAverageRecall()