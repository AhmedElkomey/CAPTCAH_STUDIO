#!/bin/bash
# Build Script for CAPTCHA Desktop App on Linux
# Requires PyInstaller to be installed (`pip install pyinstaller`)

echo -e "\e[36mBuilding CAPTCHA Desktop App with PyInstaller...\e[0m"

# Ensure output gets cleanly rebuilt
if [ -d "dist/CAPTCHA_Studio" ]; then
    rm -rf "dist/CAPTCHA_Studio"
fi
if [ -d "build" ]; then
    rm -rf "build"
fi

# Run PyInstaller
# --noconfirm: overwrite existing
# --onedir: creates a folder with the executable and dependencies
# --windowed: no console window
# --name: the output generic name
# --hidden-import: ensures dynamic imports from PIL and cv2 don't crash the executable
python3 -m PyInstaller --noconfirm \
    --onedir \
    --windowed \
    --name "CAPTCHA_Studio" \
    --hidden-import "PIL" \
    --hidden-import "cv2" \
    --hidden-import "numpy" \
    --add-data "ui:ui" \
    app.py

if [ $? -eq 0 ]; then
    echo -e "\n\e[32mBuild Successful!\e[0m"
    echo -e "\e[33mExecutable is located at: dist/CAPTCHA_Studio/CAPTCHA_Studio\e[0m"
else
    echo -e "\n\e[31mBuild Failed.\e[0m"
fi
