import numpy as np

from sklearn.metrics import confusion_matrix


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
        super(AbstractAccuracy, self).__init__()
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


class AccuracyBinary(AbstractAccuracy):
    """
    Accuracy for binary classification.
    """
    def __init__(self):
        super(AccuracyBinary, self).__init__()

    def __call__(self, output, target, loss):
        prediction = output > 0.5
        self.correct += (prediction == target).sum().float()
        self.total += len(target)
        return self.value


class Precision(AbstractMetric):
    """
    Simple precision metric: TP / (TP + FP).
    Measures the percentage of positives predictions that are correct.
    """
    def __init__(self):
        super(Precision, self).__init__()
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
    Simple precision metric: TP / (TP + FP).
    Measures the percentage of positives predictions that are correct.
    """
    def __init__(self):
        super(Recall, self).__init__()
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
    Simple precision metric: TP / (TP + FP).
    Measures the percentage of positives predictions that are correct.
    """
    def __init__(self):
        super(F1, self).__init__()
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
        return 2 * precision * recall / (precision + recall)

    def reset(self):
        self.precision = None
        self.recall = None


class MeanAveragePrecision(AbstractMetric):
    """
    Mean precision over all classes: mean(TP_i / (TP_i + FP_i)) for i in classes.
    """
    def __init__(self):
        super(MeanAveragePrecision, self).__init__()
        self.average_precision = 0 #
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