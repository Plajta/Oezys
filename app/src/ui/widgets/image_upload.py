from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PIL import Image
from PIL.ImageQt import ImageQt

from ...utils.afm import convertAFMtoImage, is_spm_file


class ImageUploadWidget(QWidget):
    image_loaded = pyqtSignal(object, str)  # emits (PIL.Image.Image, original_path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Image.Image | None = None
        self._path: str = ""
        self._build_ui()
        self.setAcceptDrops(True)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._preview = QLabel("Drop AFM file here\nor click Browse")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setContentsMargins(0, 0, 0, 0)
        self._preview.setMinimumSize(320, 280)
        self._preview.setStyleSheet(
            "border: 2px dashed #538AC1; border-radius: 8px; color: #538AC1; font-size: 14px; font-family: 'Open Sans', sans-serif;"
        )

        self._btn = QPushButton("Browse")
        self._btn.setFixedWidth(120)
        self._btn.clicked.connect(self._open_file_dialog)

        layout.addWidget(self._preview)
        layout.addWidget(self._btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open AFM File", "", "All Files (*)"
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        if is_spm_file(path):
            image = convertAFMtoImage(path)
        else:
            image = Image.open(path)

        self._path = path
        self._image = image

        pixmap = QPixmap.fromImage(ImageQt(image)).scaled(
            self._preview.width(),
            self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pixmap)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet("border: 2px solid #00B3DB; border-radius: 8px; padding: 0px;")
        self.image_loaded.emit(self._image, self._path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self._load_path(urls[0].toLocalFile())

    @property
    def current_image(self) -> Image.Image | None:
        return self._image
