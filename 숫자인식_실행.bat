@echo off
chcp 65001 > nul
cd /d "%~dp0"
python draw_and_recognize.py
pause
