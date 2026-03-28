@echo off
echo ========================================
echo  YOLO26 - ENHANCED EMPTY SPOT FIX
echo ========================================
echo.
echo ✅ ENHANCED GREEN EMPTY SPOTS:
echo - 🟢 LARGE "EMPTY" labels in prominent GREEN
echo - 🟢 "UNOCCUPIED" text for extra clarity  
echo - 🟢 Thicker green boxes (8px lines)
echo - 🟢 Larger font size (1.0) for better visibility
echo - 🟢 Consistent with RED occupied spots
echo - 🔴 RED "OCCUPIED" on all cars
echo - 🟢 GREEN "EMPTY" on all empty spaces
echo.
echo 🎯 As per your request:
echo    "jitni places empty haii wha paii green colour"
echo    "sabhi frame main dekho empty dikhe waha paii green aaye"
echo.
echo 🚀 ENHANCED FEATURES:
echo - Empty spots now AS VISIBLE as occupied spots
echo - Prominent green boxes and labels
echo - Double labeling: EMPTY + UNOCCUPIED
echo - Consistent visibility across all frames
echo - Professional parking lot display
echo.
echo Starting enhanced empty spot detection...
echo.

cd /d "%~dp0"
python apps\app.py

pause
