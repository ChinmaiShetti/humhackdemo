@echo off
rem Double-click to start or resume everything in P5 before the freeze. Close the window to pause.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_p5.ps1"
pause
