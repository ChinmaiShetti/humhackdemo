@echo off
rem Replication R2 (docs/replication2_prereg.md): data steps up to the replication freeze.
rem Double-click to start or resume. Close the window to pause.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_replication.ps1" -Sets "scifact,cqa_mathematica,cqa_programmers"
pause
