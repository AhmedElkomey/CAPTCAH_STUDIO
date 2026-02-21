DARK_THEME = """
QWidget {
    background-color: #1e1e1e;
    color: #cccccc;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #1e1e1e;
}

QTabWidget::pane {
    border: 1px solid #333333;
    border-radius: 4px;
    background: #252526;
}

QTabBar::tab {
    background: #2d2d30;
    color: #999999;
    padding: 8px 16px;
    border: 1px solid #333333;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #252526;
    color: #ffffff;
    border-bottom: 2px solid #007acc;
}

QPushButton {
    background-color: #0e639c;
    color: white;
    border: none;
    padding: 6px 16px;
    border-radius: 3px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1177bb;
}

QPushButton:pressed {
    background-color: #094771;
}

QPushButton:disabled {
    background-color: #333333;
    color: #777777;
}

QLineEdit, QSpinBox, QComboBox {
    background-color: #3c3c3c;
    color: #cccccc;
    border: 1px solid #555555;
    padding: 4px;
    border-radius: 2px;
}

QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #007acc;
}

QComboBox::drop-down {
    border: none;
}

QGroupBox {
    border: 1px solid #444444;
    border-radius: 4px;
    margin-top: 1ex;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
    color: #aaaaaa;
}

QSlider::groove:horizontal {
    border: 1px solid #444444;
    height: 4px;
    background: #3c3c3c;
    margin: 2px 0;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #007acc;
    border: 1px solid #005a9e;
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QProgressBar {
    border: 1px solid #444444;
    border-radius: 3px;
    text-align: center;
    background: #2d2d30;
    color: white;
}

QProgressBar::chunk {
    background-color: #007acc;
    width: 10px;
}

QListWidget {
    background-color: #1e1e1e;
    border: 1px solid #333333;
}

QListWidget::item:selected {
    background-color: #37373d;
}

QGraphicsView {
    border: 1px solid #333333;
    background-color: #1e1e1e;
}
"""
