from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QPaintEvent
from PyQt6.QtWidgets import QWidget


class LoadingOverlay(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self):
        self._angle = 0
        self.resize(self.parentWidget().size())
        self.raise_()
        self.show()
        self._timer.start(16)

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def resizeEvent(self, event):
        self.resize(self.parentWidget().size())

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # dimmed background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 160))

        cx = self.width() / 2
        cy = self.height() / 2
        r = 28

        # track circle
        pen = QPen(QColor("#152331"), 5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # spinning arc
        pen.setColor(QColor("#00B3DB"))
        painter.setPen(pen)
        painter.drawArc(
            QRectF(cx - r, cy - r, r * 2, r * 2),
            (-self._angle) * 16,
            -120 * 16,
        )

        # label
        painter.setPen(QColor("#538AC1"))
        font = painter.font()
        font.setPointSize(11)
        font.setFamily("Open Sans")
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(0, int(cy) + r + 14, 0, 0),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Analysing…",
        )
