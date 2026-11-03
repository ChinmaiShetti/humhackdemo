# Replication R1 (docs/replication_prereg.md, D-069/D-070), steps up to the replication freeze:
# download + build D5/D6 -> splits -> corpus encode/index -> P-std/P-persona pools -> neighbours
# -> C1-C7 on the synthetic queries -> B0/B1/OVL sets. A is NOT unsealed here.
# About 1.5-2 h. Pause: close the window. Resume: run again (built data, pools, neighbours and
# runs are reused; selection is deterministic).
# Datasets: -Sets "a,b,c" (default R1's). R2: run_replication2.bat (scifact, Mathematica, Programmers).
param([string[]]$Sets = @("cqa_webmasters", "nfcorpus"))
$ErrorActionPreference = "Stop"
Set-Location "D:\humanalign-rag"
$env:TEMP = "D:\humanalign-rag\cache\tmp"
$env:PYTHONUTF8 = "1"
$ha = ".venv\Scripts\ha.exe"
$py = ".venv\Scripts\python.exe"
# Windows PowerShell 5.1 may pass "a,b,c" as an array or as one string: accept both.
$sets = @($Sets | ForEach-Object { $_ -split "[,\s]+" } | Where-Object { $_ })

Start-Transcript -Path "logs\run_replication_$(Get-Date -Format 'yyyyMMdd_HHmm').log" -Append | Out-Null

# Every step is idempotent, so a transient failure (e.g. the Ollama server) is retried.
function Step([string]$name, [scriptblock]$cmd) {
    for ($try = 1; $try -le 3; $try++) {
        Write-Host "=== $name (attempt $try)  $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        & $cmd
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "$name failed (exit $LASTEXITCODE); retrying in 60 s."
        Start-Sleep -Seconds 60
    }
    Write-Host "STOPPED: $name failed 3 times"; Stop-Transcript | Out-Null; exit 1
}
function Unload { Step "free the GPU" { & $py -c "from ha import llm; print(llm.unload_all())" } }

foreach ($ds in $sets) {
    Step "build $ds (download, count checks)" { & $ha data build $ds }
    if (-not (Test-Path "data\processed\$ds\splits.parquet")) { Step "splits $ds" { & $ha data splits $ds } }
    Step "encode $ds corpus" { & $ha systems encode $ds }
    Step "BM25 index $ds" { & $ha systems index $ds }
}
foreach ($ds in $sets) {
    foreach ($pool in @("P-std", "P-persona")) {
        Step "generate $ds $pool" { & $ha gen $ds $pool --finalize }
    }
}
foreach ($ds in $sets) {
    $nb = 0..4 | Where-Object { Test-Path "data\neighbours\$ds\s$_\U.parquet" }
    if (@($nb).Count -lt 5) { Unload; Step "neighbours $ds" { & $ha neigh $ds } }
}
foreach ($ds in $sets) {
    Unload
    Step "C1-C4 on $ds synthetic queries" {
        & $ha systems run $ds --qset U --systems C1 --systems C2 --systems C3 --systems C4 }
    Unload
    Step "C1-C7 on $ds synthetic queries" { & $ha systems run $ds --qset U }
}
foreach ($ds in $sets) {
    Step "query sets $ds (B0, B1, OVL)" {
        & $py -c "import json; from ha.select.replication import run_select; [print(json.dumps(m)) for m in run_select('$ds')]" }
}
Write-Host "=== Replication data steps done ($($sets -join ', ')) $(Get-Date -Format 'yyyy-MM-dd HH:mm'). Next: commit, replication freeze, then the single evaluation."
