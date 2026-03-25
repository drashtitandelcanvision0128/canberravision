#!/usr/bin/env python3
"""
Parking System Startup Script
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=== Parking Occupancy Detection System ===")
    print("Starting system components...")
    
    try:
        # Import and start dashboard
        from apps.parking_dashboard import main as dashboard_main
        dashboard_main()
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"Error starting system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
