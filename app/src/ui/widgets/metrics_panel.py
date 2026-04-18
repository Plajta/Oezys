from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from matplotlib.ticker import FuncFormatter

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

    def update_metrics(self, metrics):
        probs = getattr(metrics, "Probabilities", None) or getattr(metrics, "probabilities", []) or []
        labels = list(getattr(metrics, "labels", []))

        if not probs or not labels:
            self._figure.clear()
            self._canvas.draw()
            self._summary_label.setText("No data")
            return

        # Ensure consistent length
        n = min(len(probs), len(labels))
        probs = probs[:n]
        labels = labels[:n]

        values = [p / 100 for p in probs]

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#0d1820")

        y_pos = range(n)
        ax.barh(y_pos, values, color="#538AC1")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color="#ddd", fontsize=9)

        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x * 100)}%"))

        ax.tick_params(axis="x", colors="#538AC1", labelsize=9)
        ax.tick_params(axis="y", colors="#ddd", labelsize=9)

        for spine in ax.spines.values():
            spine.set_edgecolor("#152331")

        # Clamp text position so it doesn't overflow
        for i, (val, prob) in enumerate(zip(values, probs)):
            x_pos = min(val + 0.01, 0.98)
            ax.text(x_pos, i, f"{prob}%", va="center", color="#00B3DB", fontsize=9)

        self._canvas.draw()

        # Use same sliced labels
        best_idx = max(range(n), key=lambda i: probs[i])
        self._summary_label.setText(f"Prediction: {labels[best_idx]}")
    def clear(self):
        self._figure.clear()
        self._canvas.draw()
        self._summary_label.setText("")
