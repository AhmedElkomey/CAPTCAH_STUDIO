# Build Script for CAPTCHA Desktop App
# Requires PyInstaller to be installed (`pip install pyinstaller`)

Write-Host "Building CAPTCHA Desktop App with PyInstaller..." -ForegroundColor Cyan

# Ensure output gets cleanly rebuilt
if (Test-Path "dist\CAPTCHA_STUDIO") {
    Remove-Item -Recurse -Force "dist\CAPTCHA_STUDIO"
}
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}

# Run PyInstaller
# --noconfirm: overwrite existing
# --onedir: creates a folder with the exe and dependencies (much faster startup than --onefile)
# --noconsole: hides the terminal window
# --name: the output generic name
# --hidden-import: ensures dynamic imports from PIL and cv2 don't crash the executable
pyinstaller --noconfirm `
    --onedir `
    --windowed `
    --noconsole `
    --name "CAPTCHA_STUDIO" `
    --hidden-import "PIL" `
    --hidden-import "cv2" `
    --hidden-import "numpy" `
    --add-data "ui;ui" `
    app.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild Successful!" -ForegroundColor Green
    Write-Host "Executable is located at: dist\CAPTCHA_STUDIO\CAPTCHA_STUDIO.exe" -ForegroundColor Yellow
}
else {
    Write-Host "`nBuild Failed." -ForegroundColor Red
}
