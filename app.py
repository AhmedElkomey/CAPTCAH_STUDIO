import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from ui.styles import DARK_THEME

def main():
    app = QApplication(sys.argv)
    
    # Set global application properties
    app.setApplicationName("CAPTCHA Studio")
    app.setStyle("Fusion")
    
    # Apply stylesheet
    app.setStyleSheet(DARK_THEME)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
