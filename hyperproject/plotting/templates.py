import matplotlib.pyplot as plt
from mltools.plotting import PlotTemplate

from .summaries import ROC


class RocCurveTemplate(PlotTemplate):
    def __init__(
        self, *, xlabel: str = "False Positive Rate", ylabel: str = "True Positive Rate"
    ):
        super().__init__()
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(
        self,
        fig: plt.Figure,
        summary: ROC,
    ) -> None:
        ax = fig.subplots(1, 1)
        for source_id in summary.sources():
            false_positive_rate, true_positive_rate, auc = summary.compute(source_id)
            ax.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{source_id} (AUC = {auc:.2f})",
            )

        ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(f"ROC for {summary.digit}")
        ax.legend(loc="lower right", frameon=False)
