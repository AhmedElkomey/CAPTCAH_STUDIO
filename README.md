# CAPTCHA Studio: Annotator & Generator
A fully-featured PySide6 Desktop Application designed to rapidly generate synthetic text CAPTCHAs, simulate extreme visual distortions (mesh warping, noise, distractors), and annotate images locally.

---

## 🛠 Features

### 1. CAPTCHA Generator 
* **Dynamic Distortions**: Apply authentic mesh-warp technology natively rendering natural textures like wood, marble, or waves instead of simple solid colors.
* **Granular Controls**: Detailed noise density inputs, precise character rotation jitter bounds, and synthetic scribbling.
* **Typography Injection**: Allows rapid loading of any local `.ttf` or `.otf` font file to expand generating datasets.
* **Batch Production**: Threaded parallel generation. Outputs strictly adhere to OCR training architecture standards (PNG images output to an `/images` directory + mapped horizontally to a TSV `labels.txt` at the root).

### 2. CAPTCHA Annotator
* **Dataset Formatting**: Deeply coupled with the Generator engine. Reads standard `/images` directories and directly edits TSV `labels.txt`.
* **Rapid Interface**: Keyboard navigation bound (←/→ arrow keys for moving, Enter to commit annotations).
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

1. Install PyInstaller into your environment:
   ```powershell
   pip install pyinstaller
   ```
2. Execute the PowerShell compiler script:
   ```powershell
   .\build.ps1
   ```
3. Wait for the compilation block to finish. When successful, the application will be located inside the new `dist\` folder hierarchy. You can zip that root application directory and run `CAPTCHA_App.exe` locally anywhere.

---

## 📁 Architecture Requirements
If you load a directory into the **Annotator Tab**, the tool expects the folder schema to be specifically:

```
[Your Chosen Folder]
│--- labels.txt         <-- Placed here automatically to log TSV mappings.
│
└─── images/            <-- Place your .jpg or .png images here.
      ├── captcha_001.png
      └── captcha_002.png
```
If you generate a project via the **Generator Tab**, this exact architecture will automatically be scaffolded and written.
