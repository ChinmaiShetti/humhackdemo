@echo off
rem Double-click to start or resume the R1 replication data steps. Close the window to pause.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_replication.ps1"
pause
