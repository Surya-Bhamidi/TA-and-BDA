$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv312\Scripts\python.exe'
$process = Start-Process -FilePath $projectPython -ArgumentList @('-m','streamlit','run','app.py','--server.address','127.0.0.1','--server.port','8501') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'logs\dashboard.log') -RedirectStandardError (Join-Path $projectRoot 'logs\dashboard-error.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $projectRoot 'artifacts\dashboard.pid')
Write-Output "Dashboard process $($process.Id) started at http://localhost:8501"
