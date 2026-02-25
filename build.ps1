# Build script for CAPTCHA Desktop App (Windows).
# Requires PyInstaller to be installed: python -m pip install pyinstaller

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Building CAPTCHA Desktop App with PyInstaller..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not available on PATH." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".\app.py")) {
    Write-Host "app.py was not found. Run this script from the repository root." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".\ui")) {
    Write-Host "ui folder was not found. Packaging cannot continue." -ForegroundColor Red
    exit 1
}

# Ensure output gets cleanly rebuilt.
foreach ($path in @(".\dist\CAPTCHA_Studio", ".\build")) {
    if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
    }
}

# Run PyInstaller.
& python -m PyInstaller --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name "CAPTCHA_Studio" `
    --hidden-import "PIL" `
    --hidden-import "cv2" `
    --hidden-import "numpy" `
    --add-data "ui;ui" `
    app.py

$exePath = ".\dist\CAPTCHA_Studio\CAPTCHA_Studio.exe"
if (Test-Path $exePath) {
    Write-Host "`nBuild Successful!" -ForegroundColor Green
    Write-Host "Executable is located at: $exePath" -ForegroundColor Yellow
    exit 0
}

Write-Host "`nBuild failed: expected output not found at $exePath" -ForegroundColor Red
exit 1
