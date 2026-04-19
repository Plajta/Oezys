from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
)

from ...processing.metrics import MetricData
from ...utils.afm import convertAFMtoImage, is_spm_file
from PIL.ImageQt import ImageQt

_LABELS = ["Diabetes", "Glaucoma", "M.Sclerosis", "Dry Eye", "Healthy"]
_COLUMNS = ["Date", "File"] + _LABELS + ["Prediction"]

_STYLE = """
    QDialog, QWidget { background-color: #0d1820; color: #ddd; font-family: 'Open Sans', sans-serif; }
    QTableWidget { background-color: #0a141d; gridline-color: #152331; border: none; }
    QHeaderView::section { background-color: #152331; color: #538AC1; padding: 4px; border: none; font-size: 11px; }
    QTableWidget::item { padding: 4px; font-size: 11px; }
    QTableWidget::item:selected { background-color: #1e3347; }
    QTableWidget::item:alternate { background-color: #111d27; }
    QPushButton { background-color: #152331; color: #ddd; border: 1px solid #538AC1; border-radius: 6px; padding: 6px 16px; }
    QPushButton:hover { background-color: #538AC1; color: #fff; }
"""


class HistoryDialog(QDialog):
    def __init__(self, records: list[MetricData], parent=None):
        super().__init__(parent)
        self._records = records
        self.setWindowTitle("Analysis History")
        self.setMinimumSize(1000, 520)
        self.setStyleSheet(_STYLE)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Previous Analyses")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #538AC1;")
        layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- table ---
        self._table = QTableWidget(len(self._records), len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)

        for row, m in enumerate(self._records):
            probs = m.Probabilities or [0.0] * 5
            best = _LABELS[probs.index(max(probs))]
            cells = [m.Datetime or "", m.FileName or ""] + [f"{p:.1f}%" for p in probs] + [best]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == len(_COLUMNS) - 1:
                    item.setForeground(QColor("#00B3DB"))
                self._table.setItem(row, col, item)

        self._table.selectionModel().selectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self._table)

        # --- preview panel ---
        self._preview_label = QLabel("Select a row\nto preview image")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(
            "border: 2px dashed #538AC1; border-radius: 8px; color: #538AC1; font-size: 13px; min-width: 240px; min-height: 220px;"
        )

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.addWidget(self._preview_label)

        splitter.addWidget(right)
        splitter.setSizes([680, 280])
        layout.addWidget(splitter, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _on_row_selected(self):
        indexes = self._table.selectedIndexes()
        if not indexes:
            return
        row = indexes[0].row()
        m = self._records[row]
        filepath = getattr(m, "FilePath", "") or ""
        if not filepath or not Path(filepath).exists():
            self._preview_label.setText("Image not available")
            self._preview_label.setPixmap(QPixmap())
            return

        try:
            if is_spm_file(filepath):
                from PIL import Image as PILImage
                pil_img = convertAFMtoImage(filepath)
            else:
                from PIL import Image as PILImage
                pil_img = PILImage.open(filepath)

            pixmap = QPixmap.fromImage(ImageQt(pil_img)).scaled(
                240, 240,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(pixmap)
            self._preview_label.setStyleSheet(
                "border: 2px solid #00B3DB; border-radius: 8px; min-width: 240px; min-height: 220px;"
            )
        except Exception as e:
            self._preview_label.setText(f"Failed to load\n{e}")
