from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget



class MetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Probabilities")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ddd; font-family: 'Open Sans', sans-serif;")
        layout.addWidget(title)

        self._figure = Figure(facecolor="#0a141d", tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(200)
        layout.addWidget(self._canvas)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("font-size: 12px; color: #538AC1; font-family: 'Open Sans', sans-serif;")
        layout.addWidget(self._summary_label)

    def update_metrics(self, metrics: MetricsResult):
        chart_data = metrics.charts.get("probabilities", [])
        labels = [d["label"] for d in chart_data]
        values = [d["value"] for d in chart_data]

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#0d1820")
        bars = ax.barh(labels, values, color="#538AC1")
        ax.set_xlim(0, 1)
        ax.tick_params(colors="#538AC1")
        for spine in ax.spines.values():
            spine.set_edgecolor("#152331")
        ax.bar_label(bars, fmt="%.2f", padding=4, color="#00B3DB", fontsize=9)
        self._canvas.draw()

        summary_text = "  |  ".join(
            f"{k}: {v:.4f}" for k, v in metrics.summary.items()
        )
        self._summary_label.setText(summary_text)

    def clear(self):
        self._figure.clear()
        self._canvas.draw()
        self._summary_label.setText("")
