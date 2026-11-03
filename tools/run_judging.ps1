# P5.1 bulk judging (amendment A3): FiQA, then CQADupStack-Android. About 13 h in total.
#
# Pause:  press Ctrl+C or close this window, at any time. At most the query in progress is lost.
# Resume: run this script again. Finished queries are skipped (per-stage logs + judge cache).
#
# A crash (for example Ollama restarting) is retried up to 5 times, 60 s apart.
$ErrorActionPreference = "Stop"
Set-Location "D:\humanalign-rag"
$env:TEMP = "D:\humanalign-rag\cache\tmp"
$env:PYTHONUTF8 = "1"

foreach ($ds in @("fiqa", "cqa_android")) {
    for ($try = 1; $try -le 5; $try++) {
        Write-Host "=== ha judge $ds (attempt $try) $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        & .venv\Scripts\ha.exe judge $ds
        if ($LASTEXITCODE -eq 0) { break }
        if ($LASTEXITCODE -eq 3) { Write-Host "BLOCKED: see the message above."; exit 3 }
        Write-Host "ha judge $ds stopped with exit code $LASTEXITCODE; retrying in 60 s."
        Start-Sleep -Seconds 60
    }
    if ($LASTEXITCODE -ne 0) { Write-Host "Giving up on $ds after 5 attempts."; exit 1 }
}
Write-Host "=== All bulk judging done $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
