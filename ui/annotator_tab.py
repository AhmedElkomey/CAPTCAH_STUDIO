import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QProgressBar, QListWidget, QFileDialog, QGraphicsView, QGraphicsScene, 
    QMessageBox, QSplitter, QListWidgetItem
)
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut, QPainter
from PySide6.QtCore import Qt

# No fixed ANNOTATIONS_FILE since it depends on the loaded directory
FLAGGED_LABEL = "[FLAGGED]"

class AnnotatorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.image_dir = "data"
        self.images = []
        self.current_index = -1
        self.annotations = {} # filename -> label
        self.history = [] # list of (filename, previous_label) for undo
        
        self.labels_file = None
        self.flags_file = None
        
        self.init_ui()
        
        # Attempt to load default directory
        if os.path.exists(self.image_dir):
            self.load_directory(self.image_dir)

    def load_annotations(self):
        self.annotations = {}
        if self.labels_file and os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip('\n').split('\t')
                        if len(parts) >= 2:
                            self.annotations[parts[0]] = parts[1]
            except Exception:
                pass
        if self.flags_file and os.path.exists(self.flags_file):
            try:
                with open(self.flags_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        fname = line.strip()
                        if fname:
                            self.annotations[fname] = FLAGGED_LABEL
            except Exception:
                pass

    def save_annotations(self):
        if not self.labels_file:
            return

        flagged = []
        with open(self.labels_file, 'w', encoding='utf-8', newline='') as f:
            for fname, label in self.annotations.items():
                if label == FLAGGED_LABEL:
                    flagged.append(fname)
                    continue
                f.write(f"{fname}\t{label}\n")
        if self.flags_file:
            with open(self.flags_file, 'w', encoding='utf-8', newline='') as f:
                for fname in sorted(flagged):
                    f.write(f"{fname}\n")

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left panel: Image List (Thumbnails/Names)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_open_folder = QPushButton("📂 Open Folder")
        self.btn_open_folder.clicked.connect(self.select_folder)
        left_layout.addWidget(self.btn_open_folder)
        
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_list_selected)
        left_layout.addWidget(self.list_widget)

        # Right panel: Viewer and Editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_filename = QLabel("No image loaded")
        self.lbl_filename.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        self.lbl_filename.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_filename)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_layout.addWidget(self.view)

        # Controls input area
        input_layout = QHBoxLayout()
        self.lbl_input = QLabel("Label (Enter to save):")
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type CAPTCHA text here...")
        self.input_field.returnPressed.connect(self.save_and_next)
        input_layout.addWidget(self.lbl_input)
        input_layout.addWidget(self.input_field)
        
        self.btn_flag = QPushButton("🚩 Flag/Skip")
        self.btn_flag.clicked.connect(self.flag_and_next)
        input_layout.addWidget(self.btn_flag)
        right_layout.addLayout(input_layout)

        # Navigation & Actions
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⬅ Prev")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Next ➡")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export.clicked.connect(self.export_csv)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_export)
        right_layout.addLayout(nav_layout)

        # Progress
        self.lbl_progress_summary = QLabel("Labeled 0/0 • Flagged 0 • Reviewed 0")
        self.lbl_progress_summary.setAlignment(Qt.AlignCenter)
        self.lbl_progress_summary.setStyleSheet(
            "padding: 6px; color: #d4d4d4; font-size: 12px; font-weight: 600;"
        )
        right_layout.addWidget(self.lbl_progress_summary)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0% Labeled")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(24)
        self.progress_bar.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid #3a3a3a;"
            " border-radius: 12px;"
            " background: #1f1f1f;"
            " color: #f3f3f3;"
            " text-align: center;"
            " font-weight: 700;"
            "}"
            "QProgressBar::chunk {"
            " border-radius: 12px;"
            " background-color: #0e639c;"
            "}"
        )
        right_layout.addWidget(self.progress_bar)

        # Splitter to hold both panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 800])
        main_layout.addWidget(splitter)

        # Shortcuts
        QShortcut(QKeySequence("Left"), self, self.prev_image)
        QShortcut(QKeySequence("Right"), self, self.next_image)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Directory", self.image_dir)
        if folder:
            self.load_directory(folder)

    def load_directory(self, folder):
        # Look for images/ subfolder as per schema
        images_sub = os.path.join(folder, "images")
        if os.path.exists(images_sub) and os.path.isdir(images_sub):
            self.image_dir = images_sub
            self.labels_file = os.path.join(folder, "labels.txt")
            self.flags_file = os.path.join(folder, "flags.txt")
        else:
            self.image_dir = folder
            self.labels_file = os.path.join(folder, "labels.txt")
            self.flags_file = os.path.join(folder, "flags.txt")
            
        self.load_annotations()
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
        try:
            files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_exts)]
        except Exception:
            files = []
            
        self.images = sorted(files)
        self.list_widget.clear()
        
        for img in self.images:
            item = QListWidgetItem(img)
            label = self.annotations.get(img)
            if label == FLAGGED_LABEL:
                item.setForeground(Qt.yellow)
            elif label is not None:
                item.setForeground(Qt.green)
            self.list_widget.addItem(item)
            
        self.update_progress()
        
        if self.images:
            # Find first unannotated
            target_idx = 0
            for i, img in enumerate(self.images):
                if img not in self.annotations:
                    target_idx = i
                    break
            self.list_widget.setCurrentRow(target_idx)
            # load_image won't trigger if target_idx == 0 and we were already at -1, unless we force it or rely on signal
            
            if self.list_widget.currentRow() == 0:
                # Force load if index didn't change (e.g., from 0 to 0)
                self.load_image(0)
        else:
            self.current_index = -1
            self.scene.clear()
            self.lbl_filename.setText("No images found.")

    def on_list_selected(self, index):
        if index >= 0:
            self.load_image(index)

    def load_image(self, index):
        if index < 0 or index >= len(self.images):
            return
            
        self.current_index = index
        filename = self.images[index]
        self.lbl_filename.setText(filename)
        
        # Load pixmap
        filepath = os.path.join(self.image_dir, filename)
        pixmap = QPixmap(filepath)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
        # Setup text field
        existing_label = self.annotations.get(filename, "")
        self.input_field.setText("" if existing_label == FLAGGED_LABEL else existing_label)
        self.input_field.setFocus()
        self.input_field.selectAll()
        
        # Keep list selection in sync
        self.list_widget.setCurrentRow(index)

    def save_and_next(self):
        if self.current_index < 0:
            return
            
        filename = self.images[self.current_index]
        label = self.input_field.text().strip()
        
        if not label:
            return # empty label ignored for Enter key
            
        old_val = self.annotations.get(filename)
        self.history.append((filename, old_val))
        
        self.annotations[filename] = label
        self.save_annotations()
        
        item = self.list_widget.item(self.current_index)
        if item:
            item.setForeground(Qt.green)
            
        self.update_progress()
        self.next_image()

    def flag_and_next(self):
        if self.current_index < 0:
            return
            
        filename = self.images[self.current_index]
        old_val = self.annotations.get(filename)
        self.history.append((filename, old_val))
        
        self.annotations[filename] = FLAGGED_LABEL
        self.save_annotations()
        
        item = self.list_widget.item(self.current_index)
        if item:
            item.setForeground(Qt.yellow)
            
        self.update_progress()
        self.next_image()

    def undo(self):
        if not self.history:
            return
            
        filename, old_val = self.history.pop()
        
        if old_val is None:
            if filename in self.annotations:
                del self.annotations[filename]
        else:
            self.annotations[filename] = old_val
            
        self.save_annotations()
        
        # Update list UI
        if filename in self.images:
            idx = self.images.index(filename)
            item = self.list_widget.item(idx)
            if item:
                if old_val is None:
                    # Remove color if un-annotated
                    item.setData(Qt.ForegroundRole, None)
                elif old_val == FLAGGED_LABEL:
                    item.setForeground(Qt.yellow)
                else:
                    item.setForeground(Qt.green)
            
            self.load_image(idx)
            self.update_progress()

    def prev_image(self):
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.load_image(self.current_index + 1)

    def update_progress(self):
        total = len(self.images)
        if total == 0:
            self.lbl_progress_summary.setText("Labeled 0/0 • Flagged 0 • Reviewed 0")
            self.progress_bar.setFormat("0% Labeled")
            self.progress_bar.setValue(0)
            return
            
        labeled = sum(
            1 for img in self.images
            if img in self.annotations and self.annotations[img] != FLAGGED_LABEL
        )
        flagged = sum(
            1 for img in self.images
            if self.annotations.get(img) == FLAGGED_LABEL
        )
        reviewed = labeled + flagged
        pct_labeled = int((labeled / total) * 100)

        self.lbl_progress_summary.setText(
            f"Labeled {labeled}/{total} • Flagged {flagged} • Reviewed {reviewed}"
        )
        self.progress_bar.setValue(pct_labeled)
        self.progress_bar.setFormat(f"{pct_labeled}% Labeled")

    def export_csv(self):
        rows = [
            (fname, label)
            for fname, label in self.annotations.items()
            if fname in self.images and label != FLAGGED_LABEL
        ]
        if not rows:
            QMessageBox.information(self, "Export", "No annotations to export.")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Labels TSV", "labels.txt", "Text Files (*.txt)")
        if save_path:
            try:
                with open(save_path, 'w', newline='', encoding='utf-8') as f:
                    for fname, label in rows:
                        f.write(f"{fname}\t{label}\n")
                QMessageBox.information(self, "Success", f"Exported {len(rows)} items to {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")
