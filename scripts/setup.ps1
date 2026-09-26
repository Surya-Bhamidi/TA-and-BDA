param([switch]$SkipPipeline)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv312\Scripts\python.exe')) {
    py -3.12 -m venv .venv312
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 from python.org, then rerun setup.' }
}
$projectPython = Join-Path $projectRoot '.venv312\Scripts\python.exe'
& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath '.runtime\java17')) {
    New-Item -ItemType Directory -Force -Path '.runtime' | Out-Null
    $asset = (Invoke-RestMethod 'https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&image_type=jre&os=windows&vendor=eclipse')[0].binary.package
    Invoke-WebRequest -Uri $asset.link -OutFile '.runtime\java17.zip' -UseBasicParsing
    if ((Get-FileHash -LiteralPath '.runtime\java17.zip' -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.checksum) { throw 'Java checksum mismatch.' }
    Expand-Archive -LiteralPath '.runtime\java17.zip' -DestinationPath '.runtime\java17' -Force
}
if (-not $env:HADOOP_HOME -and -not (Test-Path -LiteralPath 'C:\hadoop\bin\winutils.exe')) {
    Write-Warning 'Windows Spark requires compatible Hadoop native binaries (winutils.exe and hadoop.dll). Set HADOOP_HOME to your trusted Hadoop distribution, or use the included Linux Docker setup.'
}
& $projectPython scripts\download_resources.py
if ($LASTEXITCODE -ne 0) { throw 'Language resource download failed.' }
if (-not $SkipPipeline) {
    & $projectPython run_pipeline.py
    if ($LASTEXITCODE -ne 0) { throw 'Pipeline failed; inspect the error and artifacts/run_all.json.' }
}
Write-Host 'Ready. Double-click START_DASHBOARD.cmd or run scripts\launch.ps1.'
