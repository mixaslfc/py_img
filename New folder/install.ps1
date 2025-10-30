# Usage:
#   Right-click > Run with PowerShell
#   or from terminal: powershell -ExecutionPolicy Bypass -File .\install.ps1
# What it does:
#   * Installs Python 3 (latest 3.x) via winget if missing
#   * Creates/uses .venv
#   * Upgrades pip
#   * Installs packages from requirements.txt (or a hardcoded list fallback)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Have-Cmd($name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "==> Checking for Python..."
$pythonCmd = $null
if (Have-Cmd "python") { $pythonCmd = "python" }
elseif (Have-Cmd "py") { $pythonCmd = "py -3" }

if (-not $pythonCmd) {
  Write-Host "Python not found. Attempting install via winget..."
  if (-not (Have-Cmd "winget")) {
    throw "winget is not available. Install Python manually from https://www.python.org/downloads/ and re-run."
  }
  # Install latest Python 3.x
  winget install -e --id Python.Python.3 --source winget --accept-source-agreements --accept-package-agreements
  # Try to pick it up now
  if (Have-Cmd "python") { $pythonCmd = "python" }
  elseif (Have-Cmd "py") { $pythonCmd = "py -3" }
  else {
    throw "Python appears installed but not on PATH yet. Open a new terminal and re-run this script."
  }
}

Write-Host "==> Using Python via: $pythonCmd"
# Show version
& $pythonCmd -V

# Create venv if missing
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
  Write-Host "==> Creating virtual environment .venv ..."
  & $pythonCmd -m venv .venv
}

# Activate
$venvActivate = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
  throw "Virtual environment activation script not found at $venvActivate"
}
Write-Host "==> Activating .venv ..."
. $venvActivate

# Upgrade pip and install packages
Write-Host "==> Upgrading pip ..."
python -m pip install --upgrade pip

if (Test-Path ".\requirements.txt") {
  Write-Host "==> Installing from requirements.txt ..."
  python -m pip install -r requirements.txt
} else {
  Write-Host "==> requirements.txt not found. Installing default list ..."
  python -m pip install scikit-image numpy matplotlib scipy scikit-learn pymaxflow
}

Write-Host "==> Done!"
Write-Host ""
Write-Host "To use this environment in the future:"
Write-Host "  PowerShell> .\.venv\Scripts\Activate.ps1"
Write-Host "  (.venv) PS> python your_script.py"
Write-Host ""
Read-Host -Prompt "Press Enter to exit"
