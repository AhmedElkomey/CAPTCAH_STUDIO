#!/usr/bin/env bash
# Build script for CAPTCHA Desktop App (Linux).
# Requires PyInstaller to be installed: python3 -m pip install pyinstaller

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "\e[36mBuilding CAPTCHA Desktop App with PyInstaller...\e[0m"

if ! command -v python3 >/dev/null 2>&1; then
  echo -e "\e[31mPython3 is not available on PATH.\e[0m"
  exit 1
fi
if [[ ! -f "app.py" ]]; then
  echo -e "\e[31mapp.py was not found. Run this script from the repository root.\e[0m"
  exit 1
fi
if [[ ! -d "ui" ]]; then
  echo -e "\e[31mui folder was not found. Packaging cannot continue.\e[0m"
  exit 1
fi

# Ensure output gets cleanly rebuilt.
for path in "dist/CAPTCHA_Studio" "build"; do
  if [[ -e "$path" ]]; then
    rm -rf "$path"
  fi
done

# Run PyInstaller.
python3 -m PyInstaller --noconfirm \
  --clean \
  --onedir \
  --windowed \
  --name "CAPTCHA_Studio" \
  --hidden-import "PIL" \
  --hidden-import "cv2" \
  --hidden-import "numpy" \
  --add-data "ui:ui" \
  app.py

OUT_BIN="dist/CAPTCHA_Studio/CAPTCHA_Studio"
if [[ -x "$OUT_BIN" ]]; then
  echo -e "\n\e[32mBuild Successful!\e[0m"
  echo -e "\e[33mExecutable is located at: $OUT_BIN\e[0m"
  exit 0
fi

echo -e "\n\e[31mBuild failed: expected output not found at $OUT_BIN\e[0m"
exit 1
