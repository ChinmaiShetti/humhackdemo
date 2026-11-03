@echo off
rem Double-click to start or resume the exploratory HiAQ pilot. Close the window to pause.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_hiaq_pilot.ps1"
pause
