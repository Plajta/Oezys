from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget



class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Diagnosis")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ddd; font-family: 'Open Sans', sans-serif;")
        layout.addWidget(title)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #152331;")
        layout.addWidget(separator)

        self._label = QLabel("—")
        self._label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00B3DB; font-family: 'Open Sans', sans-serif;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._confidence = QLabel("")
        self._confidence.setStyleSheet("font-size: 13px; color: #538AC1; font-family: 'Open Sans', sans-serif;")
        self._confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(self._label)
        layout.addWidget(self._confidence)
        layout.addStretch()

    def update_result(self, result: ModelResult):
        self._label.setText(result.label)
        self._confidence.setText(f"Confidence: {result.confidence * 100:.1f}%")

    def clear(self):
        self._label.setText("—")
        self._confidence.setText("")
