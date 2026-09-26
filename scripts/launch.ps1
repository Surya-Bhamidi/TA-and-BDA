$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv312\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Run scripts\setup.ps1 first.' }
Write-Host 'Dashboard: http://localhost:8501 — press Ctrl+C to stop.'
& $projectPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
