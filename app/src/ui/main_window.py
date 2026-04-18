from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QLinearGradient, QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from PIL import Image

from ..pipeline import Pipeline
from .widgets.image_upload import ImageUploadWidget
from .widgets.metrics_panel import MetricsPanel
from .widgets.results_panel import ResultsPanel


class _GradientWidget(QWidget):
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#000000"))
        gradient.setColorAt(1.0, QColor("#152331"))
        painter.fillRect(self.rect(), gradient)


class _RunWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, pipeline: Pipeline, image: Image.Image):
        super().__init__()
        self._pipeline = pipeline
        self._image = image

    def run(self):
        try:
            output = self._pipeline.run(self._image)
            self.finished.emit(output)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, pipeline: Pipeline | None = None):
        super().__init__()
        self._pipeline = pipeline or Pipeline()
        self._worker: _RunWorker | None = None
        self._setup_window()
        self._build_ui()

    def _setup_window(self):
        self.setWindowTitle("RadBrecim — Tear Classifier")
        self.setMinimumSize(1000, 680)
        self._apply_dark_theme()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: transparent;
                color: #ddd;
                font-family: "Open Sans", sans-serif;
            }
            QPushButton {
                background-color: #152331;
                color: #ddd;
                border: 1px solid #538AC1;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 13px;
                font-family: "Open Sans", sans-serif;
            }
            QPushButton:hover { background-color: #538AC1; border-color: #00B3DB; color: #fff; }
            QPushButton:disabled { color: #3a5060; background-color: #0d1820; border-color: #1e3347; }
            QStatusBar { background-color: #0a141d; color: #538AC1; font-size: 11px; font-family: "Open Sans", sans-serif; }
            QSplitter::handle { background-color: #152331; }
        """)

    def _build_ui(self):
        central = _GradientWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        left = self._make_left_panel()
        right = self._make_right_panel()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 580])

        root.addWidget(splitter, stretch=1)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    def _make_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet("background-color: #0a141d; border-bottom: 1px solid #152331;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("RadBrecim")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00B3DB; font-family: 'Open Sans', sans-serif;")
        layout.addWidget(title)
        layout.addStretch()

        subtitle = QLabel("Tear Film Disease Classifier")
        subtitle.setStyleSheet("font-size: 12px; color: #538AC1; font-family: 'Open Sans', sans-serif;")
        layout.addWidget(subtitle)
        return header

    def _make_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background-color: rgba(21, 35, 49, 0.6);")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        section = QLabel("INPUT IMAGE")
        section.setStyleSheet("font-size: 10px; color: #538AC1; letter-spacing: 1px; font-family: 'Open Sans', sans-serif;")
        layout.addWidget(section)

        self._upload = ImageUploadWidget()
        self._upload.image_loaded.connect(self._on_image_loaded)
        layout.addWidget(self._upload, stretch=1)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setFixedHeight(38)
        self._run_btn.setEnabled(False)
        # self._run_btn.clicked.connect(self._run_pipeline)
        layout.addWidget(self._run_btn)

        return panel

    def _make_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._results = ResultsPanel()
        self._results.setMinimumHeight(160)
        layout.addWidget(self._results)

        self._metrics = MetricsPanel()
        layout.addWidget(self._metrics, stretch=1)

        return panel

    def _on_image_loaded(self, image: Image.Image):
        self._run_btn.setEnabled(True)
        self._results.clear()
        self._metrics.clear()
        self._status.showMessage("Image loaded — ready to analyse")

    def _run_pipeline(self):
        image = self._upload.current_image
        if image is None:
            return

        self._run_btn.setEnabled(False)
        self._status.showMessage("Running analysis…")

        # self._worker = _RunWorker(self._pipeline, image)
        # self._worker.finished.connect(self._on_pipeline_done)
        # self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_pipeline_done(self, output: PipelineOutput):
        self._results.update_result(output.prediction)
        self._metrics.update_metrics(output.metrics)
        self._run_btn.setEnabled(True)
        self._status.showMessage(
            f"Done — {output.prediction.label} ({output.prediction.confidence * 100:.1f}%)"
        )

    def _on_pipeline_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._status.showMessage(f"Error: {msg}")
