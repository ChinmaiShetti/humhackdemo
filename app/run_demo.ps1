# Arzi demo launcher: builds what is missing, starts the API + site on one port, opens the browser.
#   powershell -ExecutionPolicy Bypass -File app\run_demo.ps1            # normal start
#   powershell -ExecutionPolicy Bypass -File app\run_demo.ps1 -Build     # rebuild the frontend first
param([switch]$Build, [int]$Port = 8765)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:TEMP = Join-Path $root "cache\tmp"; $env:TMP = $env:TEMP   # nothing large on C: (D-019)

try { Invoke-RestMethod "http://127.0.0.1:11434/api/version" | Out-Null }
catch { Write-Error "Ollama is not running. Start Ollama, then run this again."; exit 1 }

if (-not (Test-Path "app\web\public\field.json")) {
  Write-Host "Building the 3D passage field (once, about 30 s)..."
  & $py app\server\build_field.py
}
if (-not (Test-Path "app\web\public\lab.json")) { & $py app\server\build_lab.py }
if (-not (Test-Path "app\web\node_modules")) {
  Push-Location app\web; npm install --cache (Join-Path $root "cache\npm") --no-audit --no-fund; Pop-Location
}
if ($Build -or -not (Test-Path "app\web\dist\index.html")) {
  Push-Location app\web; npm run build; Pop-Location
}

$url = "http://127.0.0.1:$Port"
try {  # already running (e.g. a second double-click): just open it
  Invoke-RestMethod "$url/api/health" -TimeoutSec 2 | Out-Null
  Write-Host "Arzi is already running at $url"; Start-Process $url; exit 0
} catch {}

Write-Host "Starting Arzi on $url  (lab: $url/lab).  Close this window or press Ctrl+C to stop it."
# open the browser as soon as the server answers (models load first, about 10-20 s)
Start-Job { param($u) foreach ($i in 1..90) {
    try { Invoke-RestMethod "$u/api/health" -TimeoutSec 2 | Out-Null; Start-Process $u; return } catch { Start-Sleep 1 } }
} -ArgumentList $url | Out-Null
& $py -m uvicorn app.server.server:app --host 127.0.0.1 --port $Port
