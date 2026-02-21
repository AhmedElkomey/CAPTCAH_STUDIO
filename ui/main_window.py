from PySide6.QtWidgets import QMainWindow, QTabWidget

from ui.annotator_tab import AnnotatorTab
from ui.generator_tab import GeneratorTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAPTCHA Studio - Generator & Annotator")
        self.resize(1100, 800)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Initialize tabs
        self.annotator_tab = AnnotatorTab()
        self.generator_tab = GeneratorTab()
        
        self.tabs.addTab(self.annotator_tab, "✏️ Annotator")
        self.tabs.addTab(self.generator_tab, "⚙️ Generator")
