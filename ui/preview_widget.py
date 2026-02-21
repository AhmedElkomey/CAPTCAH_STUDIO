from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PIL import Image
from PIL.ImageQt import ImageQt

class PreviewWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #444444; background-color: #1a1a1a; border-radius: 4px;")
        self.setMinimumSize(370, 120)
        self._qim = None # Keep reference to avoid GC issues

    def set_image(self, pil_image: Image.Image):
        """Converts PIL Image to QPixmap safely and displays it."""
        self._qim = ImageQt(pil_image)
        pixmap = QPixmap.fromImage(self._qim)
        self.setPixmap(pixmap)
