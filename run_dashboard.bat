@echo off
echo ======================================================================
echo Launching MedFed AI - Federated Learning Medical Diagnosis Dashboard
echo ======================================================================
cd /d "%~dp0"
start http://localhost:5000
py -3.11 server.py
pause
