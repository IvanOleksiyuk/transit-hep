import numpy as np
from mltools.plotting import PredictionSummary
from sklearn.metrics import auc, roc_curve

from ..data.mnist import MNISTDataset


class ROC(PredictionSummary):
    def __init__(
        self,
        *,
        truth_ref: MNISTDataset,
        pred_ref: dict[str, dict[str, np.ndarray]],
        digit: int,
    ):
        super().__init__()
        truth_labels_one_hot = np.eye(10)[truth_ref.labels]

        assert (
            truth_labels_one_hot.shape
            == pred_ref[list(pred_ref.keys())[0]]["labels"].shape
        )
        self.truth_ref = truth_labels_one_hot
        self.pred_ref = pred_ref

        assert digit in range(10)
        self.digit = digit

    def compute(self, source: str):
        prob_truth = self.truth_ref[:, self.digit]
        prob_pred = self.pred_ref[source]["labels"][:, self.digit]

        false_positive_rate, true_positive_rate, _ = roc_curve(prob_truth, prob_pred)
        return (
            false_positive_rate,
            true_positive_rate,
            auc(false_positive_rate, true_positive_rate),
        )

    def sources(self):
        return self.pred_ref.keys()
