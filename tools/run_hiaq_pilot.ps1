# EXPLORATORY HiAQ pilot (D-067): neighbours -> judging -> C1-C7 -> selection -> evaluation.
# About 6-7 h, mostly judging. Pause: close the window. Resume: run it again (finished work is
# skipped: neighbours if present, judging per query, retrieval runs per configuration).
$ErrorActionPreference = "Stop"
Set-Location "D:\humanalign-rag"
$env:TEMP = "D:\humanalign-rag\cache\tmp"
$env:PYTHONUTF8 = "1"
$ha = ".venv\Scripts\ha.exe"

function Step([string]$name, [scriptblock]$cmd) {
    Write-Host "=== $name  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED: $name failed (exit $LASTEXITCODE)"; exit 1 }
}

$nb = 0..4 | Where-Object { Test-Path "data\neighbours\hiaq\s$_\U.parquet" }
if (@($nb).Count -lt 5) { Step "neighbours (R_s, 5 seeds)" { & $ha neigh hiaq } }
for ($try = 1; $try -le 5; $try++) {
    Write-Host "=== judging hiaq (attempt $try)  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    & $ha judge hiaq
    if ($LASTEXITCODE -eq 0) { break }
    if ($LASTEXITCODE -eq 3) { Write-Host "BLOCKED: see the message above."; exit 3 }
    Write-Host "judging stopped with exit code $LASTEXITCODE; retrying in 60 s."
    Start-Sleep -Seconds 60
}
if ($LASTEXITCODE -ne 0) { Write-Host "Giving up on judging after 5 attempts."; exit 1 }
Step "free the GPU before the reranker" {
    & .venv\Scripts\python.exe -c "from ha import llm; print(llm.unload_all())" }
Step "C1-C4 on the collected queries" {
    & $ha systems run hiaq --qset auth --systems C1 --systems C2 --systems C3 --systems C4 }
Step "free the GPU before the reranker" {
    & .venv\Scripts\python.exe -c "from ha import llm; print(llm.unload_all())" }
Step "C1-C7 on the collected queries" { & $ha systems run hiaq --qset auth }
Step "C1-C7 on the synthetic queries" { & $ha systems run hiaq --qset U }
Step "query sets" { & $ha select hiaq }
if (-not (Test-Path "results\eval\exploratory-hiaq.json")) {
    Step "evaluation (exploratory)" { & $ha eval hiaq }
}
Write-Host "=== HiAQ pilot done $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
