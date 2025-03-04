import numpy as np
import torch
from sklearn.metrics import roc_curve
from torchmetrics import Metric, MetricCollection


class EqualErrorRate(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds, target):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model (similarity scores)
            target: Ground truth labels (1 for same identity, 0 for different)
        """
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        """
        Computes the Equal Error Rate.
        """
        preds = torch.cat(self.preds, dim=0).cpu().numpy()
        target = torch.cat(self.target, dim=0).cpu().numpy()

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(target, preds)

        # Find the threshold where FAR = FRR (i.e., FPR = 1 - TPR)
        fnr = 1 - tpr
        abs_diff = np.abs(fpr - fnr)
        idx = np.nanargmin(abs_diff)

        # Return the EER
        eer = (fpr[idx] + fnr[idx]) / 2.0
        return torch.tensor(eer)


def get_metrics(**kwargs) -> MetricCollection:
    return MetricCollection({"eer": EqualErrorRate()})
