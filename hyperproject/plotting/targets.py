import matplotlib as mpl
import scipy.constants as cs
from mltools.plotting import PlotTarget


class Monitor(PlotTarget):
    APPROX_STANDARD_MONITOR_WIDTH_MM = 500

    def __init__(self) -> None:
        reasonable_plot_width = self.APPROX_STANDARD_MONITOR_WIDTH_MM / 2
        reasonable_font_size = self.APPROX_STANDARD_MONITOR_WIDTH_MM / 50
        super().__init__(
            columnwidth_mm=reasonable_plot_width,
            use_latex=False,
            fontsize_plots=reasonable_font_size,
            fontfamily_plots="sans-serif",
        )

    def get_layout(self) -> mpl.layout_engine.LayoutEngine:
        return mpl.layout_engine.ConstrainedLayoutEngine(
            h_pad=0.1, w_pad=0.1, hspace=0.05, wspace=0.05
        )


class Paper(PlotTarget):
    def __init__(
        self,
        *,
        columnwidth_pt: float,
        latex_documentclass: str,
        latex_default_fontsize: str,
        latex_imports: str,
    ) -> None:
        super().__init__(
            columnwidth_mm=columnwidth_pt * cs.pt / cs.milli,
            latex_documentclass=latex_documentclass,
            latex_default_fontsize=latex_default_fontsize,
            latex_imports=latex_imports,
            fontsize_plots="small",
            fontfamily_plots="sans-serif",
        )
