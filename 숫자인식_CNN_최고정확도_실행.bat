@echo off
chcp 65001 > nul
cd /d "%~dp0"
python draw_cnn_recognition.py
pause
