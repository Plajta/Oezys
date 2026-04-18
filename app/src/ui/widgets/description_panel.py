from datetime import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class DescriptionPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("Analysis Info")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ddd; font-family: 'Open Sans', sans-serif;"
        )
        layout.addWidget(title)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #152331;")
        layout.addWidget(separator)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(6)

        def _key(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-size: 12px; color: #538AC1; font-family: 'Open Sans', sans-serif;"
            )
            return lbl

        def _val(text="—"):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            lbl.setStyleSheet(
                "font-size: 12px; color: #ddd; font-family: 'Open Sans', sans-serif;"
            )
            return lbl

        self._name_val = _val()
        self._dt_val = _val()

        grid.addWidget(_key("Image"), 0, 0)
        grid.addWidget(self._name_val, 0, 1)
        grid.addWidget(_key("Analysed at"), 1, 0)
        grid.addWidget(self._dt_val, 1, 1)

        layout.addLayout(grid)

    def update_info(self, image_path: str):
        from pathlib import Path
        self._name_val.setText(Path(image_path).name)
        self._dt_val.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def clear(self):
        self._name_val.setText("—")
        self._dt_val.setText("—")
