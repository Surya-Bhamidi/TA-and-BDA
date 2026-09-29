param([int]$Port = 8502)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv312\Scripts\python.exe'
$probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
try { $probe.Start() } catch { throw "Port $Port is already occupied. Open the existing UI or choose another -Port; no new dashboard was started." } finally { $probe.Stop() }
$process = Start-Process -FilePath $projectPython -ArgumentList @('-m','streamlit','run','app.py','--server.address','127.0.0.1','--server.port',"$Port") -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'logs\dashboard.log') -RedirectStandardError (Join-Path $projectRoot 'logs\dashboard-error.log') -PassThru
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    if ($process.HasExited) { throw 'Dashboard startup failed. Read logs\dashboard-error.log; the selected port may be occupied.' }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/_stcore/health" -TimeoutSec 1
        if ($health -eq 'ok') { break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if ($health -ne 'ok') { throw 'Dashboard did not become healthy. Read logs\dashboard-error.log.' }
$process.Id | Set-Content -LiteralPath (Join-Path $projectRoot 'artifacts\dashboard.pid')
Write-Output "Dashboard process $($process.Id) verified at http://localhost:$Port"
