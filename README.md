# CAPTCHA Studio: Annotator & Generator
A fully-featured PySide6 Desktop Application designed to rapidly generate synthetic text CAPTCHAs, simulate extreme visual distortions (mesh warping, noise, distractors), and annotate images locally.

---

## 🛠 Features

### 1. CAPTCHA Generator 
* **Dynamic Distortions**: Apply authentic mesh-warp technology natively rendering natural textures like wood, marble, or waves instead of simple solid colors.
* **Granular Controls**: Detailed noise density inputs, precise character rotation jitter bounds, and synthetic scribbling.
* **Adversarial Blob Overlap**: Uses two-blob obstruction with XOR blend at text intersections for harder OCR samples.
* **Typography Injection**: Allows rapid loading of any local `.ttf` or `.otf` font file to expand generating datasets.
* **Batch Production**: Threaded parallel generation. Outputs strictly adhere to OCR training architecture standards (PNG images output to an `/images` directory + mapped horizontally to a TSV `labels.txt` at the root).

### 2. CAPTCHA Annotator
* **Dataset Formatting**: Deeply coupled with the Generator engine. Reads standard `/images` directories and directly edits TSV `labels.txt`.
* **Rapid Interface**: Keyboard navigation bound (←/→ arrow keys for moving, Enter to commit annotations).
* **Progress Dashboard**: Tracks labeled percentage, reviewed count, and flagged count in real time.
* **Crash Resilience**: Built-in state recovery automatically pulls previously completed bounds so progress isn't lost if the app or machine closes.
* **Label Export**: Easily port the annotations to standalone standard `.csv` when ready to be integrated into training scripts.

---

## 🚀 Getting Started

### Option A: Running from Source
Ensure you have Python 3.9+ installed.
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run the application:
   ```powershell
   python app.py
   ```

### Option B: Building an Executable
If you wish to share or deploy this application natively without requiring Python to be installed on the host machine, a build script is provided.

> [!NOTE]
> PyInstaller does not support cross-compilation. This means you **cannot** build a Linux executable on a Windows machine natively. To create a Linux executable, you must run the build script on a Linux environment (or through the provided GitHub Actions workflow).

1. Install PyInstaller into your environment:
   ```powershell
   pip install pyinstaller
   ```
2. Execute the compiler script for your operating system:
   * **Windows:** Execute `.\build.ps1` in PowerShell
   * **Linux:** Execute `./build.sh` in the terminal (ensure it's executable via `chmod +x build.sh`)
3. Wait for the compilation block to finish. When successful, the application will be located inside the new `dist\` folder hierarchy. You can zip that root application directory and run `CAPTCHA_Studio` locally anywhere.

---

## 📁 Architecture Requirements
If you load a directory into the **Annotator Tab**, the tool expects the folder schema to be specifically:

```
[Your Chosen Folder]
│--- labels.txt         <-- Placed here automatically to log TSV mappings.
│--- flags.txt          <-- Stores flagged/skipped images.
│
└─── images/            <-- Place your .jpg or .png images here.
      ├── captcha_001.png
      └── captcha_002.png
```
If you generate a project via the **Generator Tab**, this exact architecture will automatically be scaffolded and written.

---

## 🧱 Project Structure (Scalable Layout)

```
annotator/
├─ app.py                  # Desktop app entrypoint (PySide6)
├─ main.py                 # Compatibility launcher to app.py
├─ generator.py            # CAPTCHA generation core
├─ bg_gen.py               # Background/texture generation core
├─ ui/                     # UI widgets and tabs
├─ services/               # Reusable non-UI business logic
│  ├─ annotation_store.py  # labels.txt / flags.txt load-save logic
│  ├─ batch_generation.py  # batch generation pipeline used by UI worker
└─ requirements.txt
```
