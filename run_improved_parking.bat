@echo off
echo ========================================
echo  YOLO26 IMPROVED PARKING DETECTION
echo ========================================
echo.
echo ✅ FIXES APPLIED:
echo - 🔴 OCCUPIED spots = RED boxes
echo - 🟢 EMPTY spots = GREEN boxes  
echo - 📏 Larger boxes (25px padding, 6px thickness)
echo - 🎯 Labels centered on each spot (no clustering)
echo - ⏱️ Continuous display (1-second updates)
echo - 📋 JSON output with slot counting
echo.
echo Starting improved parking detection...
echo.

cd /d "%~dp0"
python apps\app.py

pause
