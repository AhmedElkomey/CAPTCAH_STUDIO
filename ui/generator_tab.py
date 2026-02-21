import os
import random
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QScrollArea, QGroupBox, QCheckBox, QComboBox, 
    QSlider, QSpinBox, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
import generator
from bg_gen import TEXTURE_STYLES
from ui.preview_widget import PreviewWidget

class BatchWorker(QThread):
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, count, out_dir, prefix, scatter, jitter):
        super().__init__()
        self.count = count
        self.out_dir = out_dir
        self.prefix = prefix
        self.scatter = scatter
        self.jitter = jitter
        self.is_running = True

    def run(self):
        try:
            images_dir = os.path.join(self.out_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            labels_path = os.path.join(self.out_dir, "labels.txt")
            
            with open(labels_path, 'a', encoding='utf-8') as f:
                for i in range(self.count):
                    if not self.is_running:
                        break
                    
                    img, text = generator.create_captcha(scatter_factor=self.scatter, jitter_factor=self.jitter)
                    filename = f"{self.prefix}{i:04d}_{text}.png"
                    out_path = os.path.join(images_dir, filename)
                    img.save(out_path)
                    
                    f.write(f"{filename}\t{text}\n")
                    f.flush()
                    
                    self.progress.emit(i + 1, self.count)
            if self.is_running:
                self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class GeneratorTab(QWidget):
    def __init__(self):
        super().__init__()
        
        self.worker = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(300) # 300ms debounce
        self.preview_timer.timeout.connect(self.update_preview)
        
        self.init_ui()
        self.trigger_preview_update()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Left Panel (Dynamic Layout Settings)
        settings_widget = QWidget()
        settings_widget.setFixedWidth(400)
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(4)
        
        # 1. Background Config
        bg_group = QGroupBox("Background Settings")
        bg_layout = QFormLayout(bg_group)
        
        self.spin_rich_bg_prob = QSpinBox()
        self.spin_rich_bg_prob.setRange(0, 100)
        self.spin_rich_bg_prob.setValue(int(generator.RICH_BACKGROUND_PROBABILITY * 100))
        self.spin_rich_bg_prob.valueChanged.connect(self.trigger_preview_update)
        bg_layout.addRow("Rich Background %:", self.spin_rich_bg_prob)
        
        self.combo_bg_style = QComboBox()
        self.combo_bg_style.addItem("random")
        self.combo_bg_style.addItems(sorted(TEXTURE_STYLES.keys()))
        self.combo_bg_style.currentIndexChanged.connect(self.trigger_preview_update)
        bg_layout.addRow("Texture Style:", self.combo_bg_style)
        
        self.chk_bg_dist = QCheckBox("Apply Background Mesh Warp")
        self.chk_bg_dist.setChecked(generator.BG_DISTORTION)
        self.chk_bg_dist.toggled.connect(self.trigger_preview_update)
        bg_layout.addRow(self.chk_bg_dist)
        
        self.chk_bg_scribbles = QCheckBox("Add Synthetic Scribbles")
        self.chk_bg_scribbles.setChecked(generator.BG_SCRIBBLES)
        self.chk_bg_scribbles.toggled.connect(self.trigger_preview_update)
        bg_layout.addRow(self.chk_bg_scribbles)
        
        self.chk_bg_vignette = QCheckBox("Add Dark Edge Vignette")
        self.chk_bg_vignette.setChecked(generator.BG_VIGNETTE)
        self.chk_bg_vignette.toggled.connect(self.trigger_preview_update)
        bg_layout.addRow(self.chk_bg_vignette)
        
        self.spin_bg_dist_min = QSpinBox(); self.spin_bg_dist_min.setRange(0, 50); self.spin_bg_dist_min.setValue(int(generator.BG_DISTORTION_MIN * 10))
        self.spin_bg_dist_max = QSpinBox(); self.spin_bg_dist_max.setRange(0, 50); self.spin_bg_dist_max.setValue(int(generator.BG_DISTORTION_MAX * 10))
        dist_range = QHBoxLayout()
        dist_range.addWidget(QLabel("Min:"))
        dist_range.addWidget(self.spin_bg_dist_min)
        dist_range.addWidget(QLabel("Max:"))
        dist_range.addWidget(self.spin_bg_dist_max)
        self.spin_bg_dist_min.valueChanged.connect(self.trigger_preview_update)
        self.spin_bg_dist_max.valueChanged.connect(self.trigger_preview_update)
        bg_layout.addRow("Warp Strength (x10):", dist_range)
        
        self.spin_bg_scribble_op = QSpinBox(); self.spin_bg_scribble_op.setRange(0, 100); self.spin_bg_scribble_op.setValue(int(generator.BG_SCRIBBLE_OPACITY * 100))
        self.spin_bg_scribble_op.valueChanged.connect(self.trigger_preview_update)
        bg_layout.addRow("Scribble Opacity %:", self.spin_bg_scribble_op)
        
        settings_layout.addWidget(bg_group)
        
        # 2. Text & Size Config
        text_group = QGroupBox("Text & Dimensions")
        text_layout = QFormLayout(text_group)
        
        self.spin_width = QSpinBox(); self.spin_width.setRange(100, 1000); self.spin_width.setValue(generator.IMG_WIDTH)
        self.spin_height = QSpinBox(); self.spin_height.setRange(50, 800); self.spin_height.setValue(generator.IMG_HEIGHT)
        self.spin_width.valueChanged.connect(self.trigger_preview_update)
        self.spin_height.valueChanged.connect(self.trigger_preview_update)
        text_layout.addRow("Width:", self.spin_width)
        text_layout.addRow("Height:", self.spin_height)
        
        self.spin_min_ch = QSpinBox(); self.spin_min_ch.setRange(1, 15); self.spin_min_ch.setValue(generator.MIN_CHARS)
        self.spin_max_ch = QSpinBox(); self.spin_max_ch.setRange(1, 15); self.spin_max_ch.setValue(generator.MAX_CHARS)
        self.spin_min_ch.valueChanged.connect(self.trigger_preview_update)
        self.spin_max_ch.valueChanged.connect(self.trigger_preview_update)
        text_layout.addRow("Min Chars:", self.spin_min_ch)
        text_layout.addRow("Max Chars:", self.spin_max_ch)
        
        self.sld_scatter = QSlider(Qt.Horizontal)
        self.sld_scatter.setRange(0, 200); self.sld_scatter.setValue(60)
        self.sld_scatter.valueChanged.connect(self.trigger_preview_update)
        text_layout.addRow("Scatter:", self.sld_scatter)
        
        self.sld_jitter = QSlider(Qt.Horizontal)
        self.sld_jitter.setRange(0, 200); self.sld_jitter.setValue(50)
        self.sld_jitter.valueChanged.connect(self.trigger_preview_update)
        text_layout.addRow("Jitter:", self.sld_jitter)
        
        settings_layout.addWidget(text_group)
        
        # 3. Distractor Config
        dist_group = QGroupBox("Noise & Distractors")
        dist_layout = QFormLayout(dist_group)
        
        self.chk_trans_dist = QCheckBox("Transparent Core Distractors")
        self.chk_trans_dist.setChecked(generator.TRANSPARENT_DISTRACTOR)
        self.chk_trans_dist.toggled.connect(self.trigger_preview_update)
        dist_layout.addRow(self.chk_trans_dist)
        
        self.spin_noise_min = QSpinBox(); self.spin_noise_min.setRange(0, 300); self.spin_noise_min.setValue(generator.NOISE_ELEMENTS_MIN)
        self.spin_noise_max = QSpinBox(); self.spin_noise_max.setRange(0, 300); self.spin_noise_max.setValue(generator.NOISE_ELEMENTS_MAX)
        n_ele = QHBoxLayout(); n_ele.addWidget(self.spin_noise_min); n_ele.addWidget(QLabel("to")); n_ele.addWidget(self.spin_noise_max)
        self.spin_noise_min.valueChanged.connect(self.trigger_preview_update)
        self.spin_noise_max.valueChanged.connect(self.trigger_preview_update)
        dist_layout.addRow("Noise Count:", n_ele)
        
        self.spin_shape_min = QSpinBox(); self.spin_shape_min.setRange(1, 4); self.spin_shape_min.setValue(generator.NOISE_SHAPES_MIN)
        self.spin_shape_max = QSpinBox(); self.spin_shape_max.setRange(1, 4); self.spin_shape_max.setValue(generator.NOISE_SHAPES_MAX)
        n_shp = QHBoxLayout(); n_shp.addWidget(self.spin_shape_min); n_shp.addWidget(QLabel("to")); n_shp.addWidget(self.spin_shape_max)
        self.spin_shape_min.valueChanged.connect(self.trigger_preview_update)
        self.spin_shape_max.valueChanged.connect(self.trigger_preview_update)
        dist_layout.addRow("Mixed Shapes:", n_shp)
        
        # 4. Font Config
        font_group = QGroupBox("Fonts Configuration")
        font_layout = QVBoxLayout(font_group)
        
        self.list_fonts = QListWidget()
        for font in generator.AVAILABLE_FONTS:
            self.list_fonts.addItem(font)
            
        font_buttons = QHBoxLayout()
        self.btn_add_font = QPushButton("➕ Add Fonts")
        self.btn_add_font.clicked.connect(self.add_fonts)
        self.btn_rm_font = QPushButton("➖ Remove Selected")
        self.btn_rm_font.clicked.connect(self.remove_fonts)
        self.btn_reset_font = QPushButton("↺ Reset to Defaults")
        self.btn_reset_font.clicked.connect(self.reset_fonts)
        
        font_buttons.addWidget(self.btn_add_font)
        font_buttons.addWidget(self.btn_rm_font)
        font_buttons.addWidget(self.btn_reset_font)
        
        font_layout.addWidget(self.list_fonts)
        font_layout.addLayout(font_buttons)
        
        settings_layout.addWidget(dist_group)
        settings_layout.addWidget(font_group)
        settings_layout.addStretch()
        
        main_layout.addWidget(settings_widget)
        
        # Right Panel (Preview & Output)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Preview Area
        preview_group = QGroupBox("Live Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_widget = PreviewWidget()
        preview_layout.addWidget(self.preview_widget)
        
        self.btn_refresh = QPushButton("↻ Reroll Random")
        self.btn_refresh.clicked.connect(self.update_preview)
        
        self.btn_help = QPushButton("❓ Help / Info")
        self.btn_help.clicked.connect(self.show_help)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_help)
        preview_layout.addLayout(btn_layout)
        
        right_layout.addWidget(preview_group)
        right_layout.addStretch()
        
        # Output Area
        out_group = QGroupBox("Batch Output")
        out_layout = QFormLayout(out_group)
        
        self.spin_count = QSpinBox(); self.spin_count.setRange(1, 50000); self.spin_count.setValue(generator.NUM_IMAGES)
        out_layout.addRow("Count:", self.spin_count)
        
        dir_layout = QHBoxLayout()
        self.lbl_out_dir = QLabel(os.path.abspath(generator.OUTPUT_DIR))
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_out_dir)
        dir_layout.addWidget(self.lbl_out_dir)
        dir_layout.addWidget(btn_browse)
        out_layout.addRow("Output Dir:", dir_layout)
        
        self.btn_generate = QPushButton("🚀 Generate Batch")
        self.btn_generate.setStyleSheet("background-color: #007acc; font-size: 16px; padding: 10px;")
        self.btn_generate.clicked.connect(self.start_generation)
        out_layout.addRow(self.btn_generate)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        out_layout.addRow(self.progress_bar)
        
        right_layout.addWidget(out_group)
        main_layout.addWidget(right_panel)

    def trigger_preview_update(self):
        self.preview_timer.start()

    def show_help(self):
        help_text = (
            "<h3>Generator Controls</h3>"
            "<b>Rich Background %:</b> Probability (0-100) of using realistic textures (wood, marble, canvas) instead of a solid color.<br>"
            "<b>Texture Style:</b> Force a specific texture type, or leave as 'random'.<br>"
            "<b>Mesh Warp:</b> Distorts the background realistically.<br>"
            "<b>Synthetic Scribbles & Vignette:</b> Adds light pen strokes and darkens edges.<br>"
            "<b>Warp Strength & Scribble Opacity:</b> Adjusts the intensity of the layer effects.<br><br>"
            "<b>Width & Height:</b> The final output dimensions of your CAPTCHA.<br>"
            "<b>Min / Max Chars:</b> Determines the string length of the generated label.<br>"
            "<b>Scatter:</b> The amount of random rotation angle applied to each individual character.<br>"
            "<b>Jitter:</b> The amount of random vertical displacement applied to each character.<br><br>"
            "<b>Fonts Configuration:</b> The list of allowed TrueType/OpenType files. You can add your own local files, remove existing files, or reset to standard Windows fonts.<br><br>"
            "<b>Transparent Core Distractors:</b> Shapes that XOR text but let background noise show through vs blocking noise completely.<br>"
            "<b>Noise Count & Mixed Shapes:</b> The density of background lines, curves, circles, and shapes overlaid on the image.<br><br>"
            "<h3>Batch Output</h3>"
            "Will generate a designated amount of CAPTCHAs into your chosen directory, placing the PNG images into an <code>images/</code> subfolder and writing the correct text strings into a <code>labels.txt</code> TSV file."
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("Help & Controls")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.exec()

    def sync_globals(self):
        """Pushes UI settings into generator module globals where needed"""
        generator.RICH_BACKGROUND_PROBABILITY = self.spin_rich_bg_prob.value() / 100.0
        generator.BG_DISTORTION = self.chk_bg_dist.isChecked()
        generator.BG_SCRIBBLES = self.chk_bg_scribbles.isChecked()
        generator.BG_VIGNETTE = self.chk_bg_vignette.isChecked()
        generator.BG_DISTORTION_MIN = self.spin_bg_dist_min.value() / 10.0
        generator.BG_DISTORTION_MAX = self.spin_bg_dist_max.value() / 10.0
        generator.BG_SCRIBBLE_OPACITY = self.spin_bg_scribble_op.value() / 100.0
        
        generator.TRANSPARENT_DISTRACTOR = self.chk_trans_dist.isChecked()
        generator.NOISE_ELEMENTS_MIN = self.spin_noise_min.value()
        generator.NOISE_ELEMENTS_MAX = self.spin_noise_max.value()
        generator.NOISE_SHAPES_MIN = self.spin_shape_min.value()
        generator.NOISE_SHAPES_MAX = self.spin_shape_max.value()
        
        generator.IMG_WIDTH = self.spin_width.value()
        generator.IMG_HEIGHT = self.spin_height.value()
        generator.MIN_CHARS = self.spin_min_ch.value()
        generator.MAX_CHARS = self.spin_max_ch.value()
        
        generator.AVAILABLE_FONTS = [self.list_fonts.item(i).text() for i in range(self.list_fonts.count())]

    def add_fonts(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Fonts", "C:/Windows/Fonts", "Fonts (*.ttf *.otf *.ttc)")
        if files:
            existing = [self.list_fonts.item(i).text() for i in range(self.list_fonts.count())]
            for f in files:
                if f not in existing:
                    self.list_fonts.addItem(f)
            self.trigger_preview_update()

    def remove_fonts(self):
        for item in self.list_fonts.selectedItems():
            self.list_fonts.takeItem(self.list_fonts.row(item))
        self.trigger_preview_update()

    def reset_fonts(self):
        self.list_fonts.clear()
        default_available = [f for f in generator.FONT_PATHS if os.path.exists(f)]
        for f in default_available:
            self.list_fonts.addItem(f)
        self.trigger_preview_update()

    def update_preview(self):
        self.sync_globals()
        
        # Mocking the `style` string directly by monkey-patching bg_gen.py is bad.
        # However, create_captcha() hardcodes style="random". 
        # We can temporarily overwrite the generate_texture_image function arg if we want,
        # but the request was generator.USE_RICH_BACKGROUNDS flag. For now the UI triggers preview
        # We will use exactly what create_captcha makes.
        
        # Save current state of create_captcha if we need to hack the random style
        # Luckily it uses "random" by default. We can let it be "random" for Phase 5 prototype,
        # or we patch it safely:
        style_choice = self.combo_bg_style.currentText()
        
        import bg_gen
        original_gen = bg_gen.generate_texture_image
        
        def mock_gen(*args, **kwargs):
            if style_choice != "random":
                kwargs["style"] = style_choice
            return original_gen(*args, **kwargs)
            
        bg_gen.generate_texture_image = mock_gen
        
        try:
            scatter = self.sld_scatter.value() / 100.0
            jitter = self.sld_jitter.value() / 100.0
            img, text = generator.create_captcha(scatter_factor=scatter, jitter_factor=jitter)
            self.preview_widget.set_image(img)
        except Exception as e:
            print(f"Preview error: {e}")
        finally:
            bg_gen.generate_texture_image = original_gen

    def browse_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.lbl_out_dir.text())
        if folder:
            self.lbl_out_dir.setText(folder)

    def start_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.is_running = False
            self.worker.wait()
            self.btn_generate.setText("🚀 Generate Batch")
            self.progress_bar.setVisible(False)
            return

        self.sync_globals()
        
        count = self.spin_count.value()
        out_dir = self.lbl_out_dir.text()
        scatter = self.sld_scatter.value() / 100.0
        jitter = self.sld_jitter.value() / 100.0
        
        self.worker = BatchWorker(count, out_dir, "test_", scatter, jitter)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.generation_finished)
        self.worker.error.connect(self.generation_error)
        
        self.progress_bar.setMaximum(count)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_generate.setText("🛑 Stop Generation")
        
        self.worker.start()

    def update_progress(self, val, total):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"Generating {val}/{total}")

    def generation_finished(self):
        self.btn_generate.setText("🚀 Generate Batch")
        QMessageBox.information(self, "Finished", "Batch generation complete!")
        self.progress_bar.setVisible(False)

    def generation_error(self, err):
        self.btn_generate.setText("🚀 Generate Batch")
        QMessageBox.critical(self, "Error", f"Failed to generate: {err}")
        self.progress_bar.setVisible(False)
