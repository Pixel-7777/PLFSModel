# Start the local PLFS wage model UI.
#
# Usage:
#   .\run_ui.ps1
#
# Optional: create and activate a virtual environment first:
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   pip install -r requirements.txt

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . ".venv\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "Starting PLFS wage model UI at http://127.0.0.1:8000"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

python app\server.py
