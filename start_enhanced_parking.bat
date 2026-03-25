@echo off
echo =========================================
echo  YOLO26 - ENHANCED SMART PARKING SYSTEM
echo =========================================
echo.
echo 🚀 ENHANCED FEATURES ACTIVATED:
echo.
echo 📍 PARKING TYPE CLASSIFICATION:
echo    ✓ On-Street Parking (roadside)
echo    ✓ Open Parking Lot (large areas)
echo    ✓ Basement Parking (underground)
echo    ✓ Multi-Level Parking (structures)
echo    ✓ Private Parking (residential/office)
echo.
echo 🅿️ SLOT CATEGORIES:
echo    ✓ Car Slots (standard vehicles)
echo    ✓ Bike Slots (motorcycles/bicycles)
echo    ✓ EV Charging Slots (electric vehicles)
echo    ✓ Disabled Slots (accessible parking)
echo    ✓ VIP Slots (reserved parking)
echo.
echo 🎯 ADVANCED DETECTION:
echo    ✓ Temporal Smoothing (no flickering)
echo    ✓ Entry/Exit Event Tracking
necho    ✓ Confidence Scoring (reliability metrics)
echo    ✓ Coordinate Scaling (any resolution)
echo    ✓ Vehicle Type Classification
echo.
echo 📊 JSON OUTPUT FORMAT:
echo    {
echo      "parkingType": "Open Parking Lot",
echo      "totalSlots": 22,
echo      "occupiedSlots": 12,
echo      "freeSlots": 10,
echo      "slots": [
echo        {
echo          "slotId": "A-01",
echo          "type": "Car",
echo          "status": "occupied",
echo          "confidence": 0.95,
echo          "vehicleType": "car"
echo        }
echo      ]
echo    }
echo.
echo 🎨 VISUAL INDICATORS:
echo    🔴 RED = Occupied (vehicle present)
echo    🟢 GREEN = Unoccupied (empty slot)
echo    📈 Occupancy Rate with color coding
echo    📝 Slot ID + Category + Confidence
echo.
echo 🚀 STARTING ENHANCED PARKING SYSTEM...
echo.

cd /d "%~dp0"
python apps\app.py

pause
