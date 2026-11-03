# P5 up to the freeze, in order: finish bulk judging (FiQA, Android), run C1-C7 on the synthetic
# queries of both datasets (C5-C7 need the GPU), build the query sets, then the FiQA development
# results (evaluation bootstrap and the H1 preview). Android's sealed test set is never touched.
#
# Pause:  close this window (or Ctrl+C) at any time.
# Resume: run it again. Finished work is skipped: judging resumes per query, retrieval runs are
#         cached per configuration, selection is deterministic, and the development results are
#         computed only once (they write to the results registry).
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

foreach ($ds in @("fiqa", "cqa_android")) {
    for ($try = 1; $try -le 5; $try++) {
        Write-Host "=== judging $ds (attempt $try)  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        & $ha judge $ds
        if ($LASTEXITCODE -eq 0) { break }
        if ($LASTEXITCODE -eq 3) { Write-Host "BLOCKED: see the message above."; exit 3 }
        Write-Host "judging $ds stopped with exit code $LASTEXITCODE; retrying in 60 s."
        Start-Sleep -Seconds 60
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "Giving up on judging $ds after 5 attempts."; exit 1 }
}
Step "free the GPU (unload Ollama) before the reranker" {
    & .venv\Scripts\python.exe -c "from ha import llm; print(llm.unload_all())" }
foreach ($ds in @("fiqa", "cqa_android")) {
    Step "C1-C7 on $ds synthetic queries" { & $ha systems run $ds --qset U }
    Step "query sets for $ds" { & $ha select $ds }
}
if (-not (Test-Path "results\eval\dev-fiqa.json")) {
    Step "FiQA development results (tau_b, bootstrap)" { & $ha eval fiqa }
}
if (-not (Test-Path "results\eval\H1-fiqa.json")) {
    Step "H1 preview on FiQA" { & $ha h1 fiqa }
}
Write-Host "=== P5 done $(Get-Date -Format 'yyyy-MM-dd HH:mm'). Next: write FROZEN.json, then the project lead signs."
