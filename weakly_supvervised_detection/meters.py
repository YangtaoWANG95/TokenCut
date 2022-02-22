import numpy as np

class AverageEpochMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def compute(self):
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

#class MultiAverageEpochMeter(object):
#    def __init__(self, name, fmt=':f'):
#        self.name = name
#        self.fmt = fmt
#        self.thresholds = np.arange(0., 1., 0.001)
#        self.reset()
#
#    def reset(self):
#        self.avg = 0
#        self.sum = np.zeros(len(self.thresholds))
#        self.count = 0
#
#    def update(self, val, n=1):
#        self.sum += val * n
#        self.count += n
#
#    def compute(self):
#        self.avg = self.sum / self.count
#        return self.avg.max()
#
#    def __str__(self):
#        fmtstr = '{name}: {avg' + self.fmt + '}'
#        return fmtstr.format(**self.__dict__)
#
#
#class AUCEpochMeter(object):
#
#    def __init__(self, name, fmt=':f'):
#        self.name = name
#        self.fmt = fmt
#        self.thresholds = np.append(np.arange(0., 1., 0.001), [1., 2., 3.])
#        self.num_bins = len(self.thresholds) - 1
#        self.reset()
#
#    def reset(self):
#        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float)
#        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float)
#
#    def update(self, gt_true_hist, gt_false_hist):
#        self.gt_true_score_hist += gt_true_hist
#        self.gt_false_score_hist += gt_false_hist
#
#    def compute(self):
#        num_gt_true = self.gt_true_score_hist.sum()
#        tp = self.gt_true_score_hist[::-1].cumsum()
#        fn = num_gt_true - tp
#
#        num_gt_false = self.gt_false_score_hist.sum()
#        fp = self.gt_false_score_hist[::-1].cumsum()
#        tn = num_gt_false - fp
#
#        if ((tp + fn) <= 0).all():
#            raise RuntimeError("No positive ground truth in the eval set.")
#        if ((tp + fp) <= 0).all():
#            raise RuntimeError("No positive prediction in the eval set.")
#
#        non_zero_indices = (tp + fp) != 0
#
#        precision = tp / (tp + fp)
#        recall = tp / (tp + fn)
#
#        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
#        return auc
#
#    def __str__(self):
#        fmtstr = '{name}: {avg' + self.fmt + '}'
#        return fmtstr.format(**self.__dict__)
