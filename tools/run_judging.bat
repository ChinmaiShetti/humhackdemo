@echo off
rem Double-click to start or resume P5.1 bulk judging. Close the window (or Ctrl+C) to pause.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_judging.ps1"
pause
