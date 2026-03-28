@echo off
echo ========================================
echo   YOLO26 - OCCUPIED ON CARS FIX
echo ========================================
echo.
echo ✅ FINAL FIXES APPLIED:
echo - 🔴 RED "OCCUPIED" labels appear DIRECTLY ON CARS
echo - 🟢 GREEN "EMPTY" labels appear on parking spots
echo - 🚗 No more separate boxes above parking spots
echo - 📏 Larger boxes for better visibility
echo - ⏱️ Continuous display (1-second updates)
echo - 📋 JSON output with slot counting
echo.
echo 🎯 As per your request:
echo    "occupied waala car ke upper hi aaye"
echo    "red colour main car ke upper"
echo.
echo Starting final improved system...
echo.

cd /d "%~dp0"
python apps\app.py

pause
