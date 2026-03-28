@echo off
echo Starting Parking Management Dashboard...
cd /d "%~dp0"

echo Checking Python environment...
python --version

echo Starting dashboard...
python apps/parking_dashboard.py

pause
