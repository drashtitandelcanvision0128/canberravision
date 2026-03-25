#!/usr/bin/env python3
"""
🚗 YOLO Car Plate Detection System - QUICK START GUIDE
=====================================================

This file helps you understand exactly which file to use for your specific needs.
Run this file to see all available options and get started immediately!

Author: Canberra Vision Team
Version: 1.0
"""

import os
import sys
import subprocess
from pathlib import Path

class QuickStartGuide:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.apps_dir = self.project_root / "apps"
        self.tools_dir = self.project_root / "tools"
        
    def show_main_menu(self):
        """Display the main menu with all options"""
        print("\n" + "="*80)
        print("🚗 YOLO CAR PLATE DETECTION SYSTEM - QUICK START MENU")
        print("="*80)
        
        print("\n📋 WHAT DO YOU WANT TO DO?")
        print("-"*50)
        
        options = {
            "1": {
                "title": "🌐 Start Main Web Application (Recommended for Beginners)",
                "file": "apps/app.py",
                "description": "Full-featured web interface for image and video processing",
                "best_for": "New users, general purpose use"
            },
            "2": {
                "title": "🚗 Car + License Plate Detection (Specific Feature)",
                "file": "apps/gradio_car_plate_app.py", 
                "description": "Specialized interface for detecting cars and reading license plates",
                "best_for": "Car plate detection, traffic monitoring"
            },
            "2a": {
                "title": "📸 License Plate Image Detection (NEW)",
                "file": "apps/license_plate_image_detector.py",
                "description": "Upload license plate images, supports multi-colored plates (white, yellow, blue, red)",
                "best_for": "License plate image upload, color recognition"
            },
            "2b": {
                "title": "🎯 Enhanced Angle-Independent Plate Detector (BEST)",
                "file": "apps/enhanced_plate_detector.py",
                "description": "Advanced detection at any angle, smart classification, multi-color support",
                "best_for": "Angled plates, difficult images, highest accuracy"
            },
            "2c": {
                "title": "🚗 GUARANTEED Plate Detector (WORKS EVERY TIME)",
                "file": "apps/guaranteed_plate_detector.py",
                "description": "Simple, robust detection that WILL find license plates like IM4U 555",
                "best_for": "When other detectors fail, guaranteed detection"
            },
            "2d": {
                "title": "🔧 SIMPLE Working Detector (NO ERRORS)",
                "file": "apps/simple_working_plate_detector.py",
                "description": "Simple, reliable detector that works without complex dependencies",
                "best_for": "When main app has errors, simple detection"
            },
            "3": {
                "title": "⚡ GPU-Accelerated Version (Fast Processing)",
                "file": "apps/app_gpu.py",
                "description": "Optimized version with GPU acceleration for faster processing",
                "best_for": "Users with NVIDIA GPUs, large video files"
            },
            "4": {
                "title": "🅿️ Parking Management System",
                "file": "apps/parking_dashboard.py",
                "description": "Parking spot detection and management dashboard",
                "best_for": "Parking lot monitoring, space management"
            },
            "5": {
                "title": "🎮 Interactive Demo with Menu",
                "file": "apps/demo_car_plate_detection.py",
                "description": "Command-line demo with menu-driven interface",
                "best_for": "Learning the system, testing features"
            },
            "6": {
                "title": "🔧 System Diagnostics & Testing",
                "file": "tools/system_test.py",
                "description": "Test all system components and check for issues",
                "best_for": "Troubleshooting, system verification"
            },
            "6a": {
                "title": "📝 Simple License Plate Detector (Command Line)",
                "file": "tools/simple_plate_detector.py",
                "description": "Easy command-line tool for license plate detection in images",
                "best_for": "Batch processing, quick plate detection"
            },
            "7": {
                "title": "📖 View Complete Code Structure Guide",
                "file": "CODE_STRUCTURE_GUIDE.md",
                "description": "Detailed documentation of all files and their purposes",
                "best_for": "Developers, understanding the codebase"
            },
            "8": {
                "title": "❓ Help & Documentation",
                "action": "show_help",
                "description": "Show detailed help and available guides",
                "best_for": "First-time users, getting help"
            }
        }
        
        for key, option in options.items():
            print(f"\n{key}. {option['title']}")
            print(f"   📁 File: {option['file']}")
            print(f"   📝 Description: {option['description']}")
            print(f"   👥 Best for: {option['best_for']}")
        
        return options
    
    def show_help(self):
        """Display detailed help information"""
        print("\n" + "="*80)
        print("📖 DETAILED HELP & DOCUMENTATION")
        print("="*80)
        
        help_files = {
            "CAR_PLATE_DETECTION_GUIDE.md": "🚗 Complete guide for car and license plate detection",
            "FAST_PROCESSING_GUIDE.md": "⚡ Guide for fast GPU-accelerated processing", 
            "IMPLEMENTATION_SUMMARY.md": "🏗️ Technical implementation details",
            "SOLUTION_SUMMARY.md": "💡 Overview of the complete solution",
            "README_STRUCTURED.md": "📋 Structured project overview",
            "CODE_STRUCTURE_GUIDE.md": "🗂️ Complete code structure guide (new)"
        }
        
        print("\n📚 Available Documentation:")
        print("-"*40)
        for filename, description in help_files.items():
            filepath = self.project_root / filename
            if filepath.exists():
                print(f"✅ {description}")
                print(f"   📁 {filename}")
            else:
                print(f"❌ {description}")
                print(f"   📁 {filename} (missing)")
        
        print("\n🛠️ Available Tools:")
        print("-"*40)
        tools = {
            "tools/system_test.py": "🧪 Complete system diagnostics",
            "tools/diagnose_ocr.py": "🔤 OCR text recognition diagnostics", 
            "tools/force_gpu.py": "⚡ GPU configuration and forcing",
            "tools/advanced_color_detection.py": "🎨 Advanced color detection",
            "tools/color_training.py": "🌈 Color model training"
        }
        
        for tool, description in tools.items():
            filepath = self.project_root / tool
            if filepath.exists():
                print(f"✅ {description}")
                print(f"   📁 {tool}")
        
        print("\n🚀 Quick Startup Scripts:")
        print("-"*40)
        scripts = {
            "start.bat": "🎯 Standard startup (Windows)",
            "start_clean.bat": "🧹 Clean startup (Windows)", 
            "start_fast.bat": "⚡ Fast startup (Windows)"
        }
        
        for script, description in scripts.items():
            filepath = self.project_root / script
            if filepath.exists():
                print(f"✅ {description}")
                print(f"   📁 {script}")
    
    def run_file(self, filepath):
        """Run a Python file"""
        try:
            if filepath.endswith('.py'):
                print(f"\n🚀 Starting: {filepath}")
                print("="*50)
                subprocess.run([sys.executable, filepath], check=True)
            elif filepath.endswith('.md'):
                print(f"\n📖 Opening documentation: {filepath}")
                print("="*50)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content[:2000] + "\n... (truncated for display)")
                    print(f"\n📁 Full file available at: {filepath}")
            else:
                print(f"❌ Cannot run file type: {filepath}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {filepath}: {e}")
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    def check_file_exists(self, filepath):
        """Check if a file exists"""
        full_path = self.project_root / filepath
        return full_path.exists()
    
    def run(self):
        """Main execution loop"""
        while True:
            try:
                options = self.show_main_menu()
                
                print("\n" + "="*50)
                choice = input("🎯 Enter your choice (1-8, 2a, 2b, 2c, 2d, 6a) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("\n👋 Thank you for using YOLO Car Plate Detection System!")
                    break
                
                if choice in options:
                    option = options[choice]
                    
                    if option.get('action') == 'show_help':
                        self.show_help()
                    else:
                        filepath = option['file']
                        if self.check_file_exists(filepath):
                            self.run_file(filepath)
                        else:
                            print(f"❌ File not found: {filepath}")
                            print("Please check the file path and try again.")
                else:
                    print("❌ Invalid choice. Please enter 1-8, 2a, 2b, 2c, 2d, 6a or 'q' to quit.")
                
                if choice != '8':  # Don't pause after showing help
                    input("\n⏸️ Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                input("⏸️ Press Enter to continue...")

def main():
    """Main entry point"""
    print("🚗 Initializing YOLO Car Plate Detection System...")
    
    try:
        guide = QuickStartGuide()
        guide.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("Please check your Python installation and try again.")

if __name__ == "__main__":
    main()
