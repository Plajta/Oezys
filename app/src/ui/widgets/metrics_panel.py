from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

_MODEL_COLORS = ["#538AC1", "#00B3DB", "#E8A838"]

_BTN_ACTIVE = (
    "background-color: #538AC1; color: #fff; border: 1px solid #538AC1;"
    " border-radius: 4px; padding: 3px 12px; font-size: 11px;"
    " font-family: 'Open Sans', sans-serif;"
)
_BTN_INACTIVE = (
    "background-color: #0d1820; color: #538AC1; border: 1px solid #538AC1;"
    " border-radius: 4px; padding: 3px 12px; font-size: 11px;"
    " font-family: 'Open Sans', sans-serif;"
)


class MetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._metrics = None
        self._view = "single"  # "single" | "all"
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header row: title + toggle buttons
        header = QHBoxLayout()
        title = QLabel("Probabilities")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ddd; font-family: 'Open Sans', sans-serif;")
        header.addWidget(title)
        header.addStretch()

        self._btn_single = QPushButton("Results")
        self._btn_all = QPushButton("Models probabilities")
        self._btn_single.setFixedHeight(26)
        self._btn_all.setFixedHeight(26)
        self._btn_single.clicked.connect(lambda: self._switch_view("single"))
        self._btn_all.clicked.connect(lambda: self._switch_view("all"))
        header.addWidget(self._btn_single)
        header.addWidget(self._btn_all)
        layout.addLayout(header)

        self._figure = Figure(facecolor="#0a141d", tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(240)
        layout.addWidget(self._canvas)

        # Coloured model-name pills shown only in "all models" view
        self._legend_row = QHBoxLayout()
        self._legend_row.setSpacing(14)
        self._legend_widgets: list[QLabel] = []
        layout.addLayout(self._legend_row)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("font-size: 12px; color: #538AC1; font-family: 'Open Sans', sans-serif;")
        layout.addWidget(self._summary_label)

        self._refresh_btn_styles()

    def _switch_view(self, view: str):
        self._view = view
        self._refresh_btn_styles()
        if self._metrics is not None:
            self.update_metrics(self._metrics)

    def _refresh_btn_styles(self):
        self._btn_single.setStyleSheet(_BTN_ACTIVE if self._view == "single" else _BTN_INACTIVE)
        self._btn_all.setStyleSheet(_BTN_ACTIVE if self._view == "all" else _BTN_INACTIVE)

    def update_metrics(self, metrics):
        self._metrics = metrics

        all_probs = getattr(metrics, "AllProbabilities", None) or []
        model_names = list(getattr(metrics, "ModelNames", None) or [])
        labels = list(getattr(metrics, "labels", []))

        # Fallback for history records that only carry a single model
        if not all_probs:
            probs = getattr(metrics, "Probabilities", None) or []
            if probs:
                all_probs = [probs]
                model_names = ["Model"]

        if not all_probs or not labels:
            self._figure.clear()
            self._canvas.draw()
            self._summary_label.setText("No data")
            return

        if self._view == "single":
            self._draw_single(all_probs[0], labels, model_names[0] if model_names else "Primary model")
        else:
            self._draw_all(all_probs, model_names, labels)

    def _set_legend(self, model_names: list[str]):
        for w in self._legend_widgets:
            self._legend_row.removeWidget(w)
            w.deleteLater()
        self._legend_widgets.clear()
        for name, color in zip(model_names, _MODEL_COLORS):
            lbl = QLabel(f"● {name}")
            lbl.setStyleSheet(
                f"font-size: 10px; color: {color}; font-family: 'Open Sans', sans-serif;"
            )
            self._legend_row.addWidget(lbl)
            self._legend_widgets.append(lbl)
        # push items left
        if self._legend_widgets:
            self._legend_row.addStretch()

    def _clear_legend(self):
        for w in self._legend_widgets:
            self._legend_row.removeWidget(w)
            w.deleteLater()
        self._legend_widgets.clear()

    def _draw_single(self, probs: list[float], labels: list[str], model_name: str):
        self._clear_legend()
        self._canvas.setMinimumHeight(240)
        self._canvas.setMaximumHeight(16777215)

        n = min(len(probs), len(labels))
        values = [probs[i] / 100 for i in range(n)]

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#0d1820")

        y_pos = list(range(n))
        ax.barh(y_pos, values, color="#538AC1")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[:n], color="#ddd", fontsize=9)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
        ax.tick_params(axis="x", colors="#538AC1", labelsize=9)
        ax.tick_params(axis="y", colors="#ddd", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#152331")
        for i, (val, prob) in enumerate(zip(values, probs)):
            ax.text(min(val + 0.01, 0.98), i, f"{prob:.1f}%", va="center", color="#00B3DB", fontsize=9)

        self._canvas.draw()
        best = max(range(n), key=lambda i: probs[i])
        self._summary_label.setText(f"Prediction ({model_name}): {labels[best]}")

    def _draw_all(self, all_probs: list[list[float]], model_names: list[str], labels: list[str]):
        self._set_legend(model_names)
        self._canvas.setMinimumHeight(240)
        self._canvas.setMaximumHeight(16777215)

        n_models = len(all_probs)
        n_classes = min(len(labels), min(len(p) for p in all_probs))

        bar_h = 0.22
        group_gap = 0.12
        group_size = n_models * bar_h + group_gap
        y_centers = [i * group_size + (n_models * bar_h) / 2 for i in range(n_classes)]

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_facecolor("#0d1820")

        for mi, (model_probs, name) in enumerate(zip(all_probs, model_names)):
            color = _MODEL_COLORS[mi % len(_MODEL_COLORS)]
            offset = (mi - (n_models - 1) / 2) * bar_h
            y_pos = [yc + offset for yc in y_centers]
            values = [model_probs[ci] / 100 for ci in range(n_classes)]
            ax.barh(y_pos, values, height=bar_h * 0.88, color=color, label=name, alpha=0.88)
            for ci, (val, yp) in enumerate(zip(values, y_pos)):
                if val > 0.01:
                    ax.text(min(val + 0.008, 1.08), yp, f"{model_probs[ci]:.1f}%",
                            va="center", color=color, fontsize=7)

        ax.set_yticks(y_centers)
        ax.set_yticklabels(labels[:n_classes], color="#ddd", fontsize=8.5)
        ax.set_xlim(0, 1.18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
        ax.tick_params(axis="x", colors="#538AC1", labelsize=8)
        ax.tick_params(axis="y", colors="#ddd", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#152331")

        self._canvas.draw()
        best = max(range(n_classes), key=lambda i: all_probs[0][i])
        self._summary_label.setText(f"Prediction: {labels[best]}")

    def clear(self):
        self._metrics = None
        self._clear_legend()
        self._figure.clear()
        self._canvas.draw()
        self._summary_label.setText("")
