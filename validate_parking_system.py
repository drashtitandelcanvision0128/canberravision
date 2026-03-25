"""
Parking System Validation Script
Simple validation and testing for the YOLO26 Parking Detection System
"""

import cv2
import numpy as np
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))

def validate_system_components():
    """Validate all system components are properly installed"""
    print("=== YOLO26 Parking System Validation ===\n")
    
    # Check required directories
    required_dirs = [
        "parking_dataset",
        "parking_dataset/config",
        "parking_dataset/images",
        "parking_dataset/labels",
        "parking_dataset/models",
        "modules",
        "apps"
    ]
    
    print("📁 Checking directories...")
    all_dirs_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Missing")
            all_dirs_exist = False
            
    # Check required files
    required_files = [
        "parking_dataset/config/parking_zones.yaml",
        "modules/parking_detection.py",
        "modules/real_time_parking.py",
        "modules/model_calibration.py",
        "parking_dataset/create_dataset.py",
        "parking_dataset/train_parking_model.py",
        "apps/parking_dashboard.py"
    ]
    
    print("\n📄 Checking core files...")
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing")
            all_files_exist = False
            
    # Check Python imports
    print("\n🐍 Checking Python imports...")
    try:
        import yaml
        print("  ✅ yaml")
    except ImportError:
        print("  ❌ yaml - Missing")
        all_files_exist = False
        
    try:
        import cv2
        print("  ✅ opencv")
    except ImportError:
        print("  ❌ opencv - Missing")
        all_files_exist = False
        
    try:
        import numpy as np
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy - Missing")
        all_files_exist = False
        
    try:
        from ultralytics import YOLO
        print("  ✅ ultralytics")
    except ImportError:
        print("  ❌ ultralytics - Missing")
        all_files_exist = False
        
    return all_dirs_exist and all_files_exist

def validate_configuration():
    """Validate parking configuration file"""
    print("\n⚙️  Validating configuration...")
    
    config_path = "parking_dataset/config/parking_zones.yaml"
    if not Path(config_path).exists():
        print("  ❌ Configuration file not found")
        return False
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Check zones
        if 'zones' not in config:
            print("  ❌ No zones defined in configuration")
            return False
            
        zones = config['zones']
        print(f"  ✅ Found {len(zones)} zones")
        
        total_spots = 0
        total_cameras = 0
        
        for zone_id, zone_config in zones.items():
            print(f"    Zone {zone_id}: {zone_config.get('name', 'Unknown')}")
            
            # Check zone structure
            required_fields = ['name', 'total_spots', 'cameras', 'camera_ids', 'coordinates']
            for field in required_fields:
                if field not in zone_config:
                    print(f"      ❌ Missing field: {field}")
                    return False
                    
            total_spots += zone_config['total_spots']
            total_cameras += zone_config['cameras']
            
            # Check camera coordinates
            for camera_id in zone_config['camera_ids']:
                if camera_id not in zone_config['coordinates']:
                    print(f"      ❌ Missing coordinates for camera {camera_id}")
                    return False
                    
                camera_spots = zone_config['coordinates'][camera_id]['spots']
                print(f"      Camera {camera_id}: {len(camera_spots)} spots")
                
        print(f"  ✅ Total: {total_spots} spots, {total_cameras} cameras")
        
        # Check detection config
        if 'detection_config' in config:
            det_config = config['detection_config']
            print(f"  ✅ Detection config: confidence={det_config.get('confidence_threshold', 0.85)}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def validate_model_availability():
    """Check if YOLO models are available"""
    print("\n🤖 Validating model availability...")
    
    model_files = [
        "yolov8n.pt",
        "yolov8s.pt", 
        "yolov8m.pt",
        "yolo26n.pt"
    ]
    
    models_found = 0
    for model_file in model_files:
        if Path(model_file).exists():
            size_mb = Path(model_file).stat().st_size / (1024*1024)
            print(f"  ✅ {model_file} ({size_mb:.1f} MB)")
            models_found += 1
        else:
            print(f"  ⚠️  {model_file} - Not found (will be downloaded automatically)")
            
    # Test model loading
    try:
        from ultralytics import YOLO
        print("  🔄 Testing model loading...")
        model = YOLO('yolov8n.pt')  # This will download if not present
        print(f"  ✅ Model loaded successfully: {len(model.names)} classes")
        print(f"    Classes: {', '.join(list(model.names.values())[:5])}...")
        return True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def validate_detection_pipeline():
    """Test the detection pipeline with sample data"""
    print("\n🔍 Validating detection pipeline...")
    
    try:
        from modules.parking_detection import ParkingDetector
        
        # Initialize detector
        detector = ParkingDetector("parking_dataset/config/parking_zones.yaml")
        print("  ✅ ParkingDetector initialized")
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("  ✅ Test frame created")
        
        # Test zone processing (if zones exist)
        if detector.config.get('zones'):
            first_zone = list(detector.config['zones'].keys())[0]
            first_camera = detector.config['zones'][first_zone]['camera_ids'][0]
            
            frames = {first_camera: test_frame}
            zone_result = detector.process_zone(frames, first_zone)
            
            print(f"  ✅ Zone processing successful: {zone_result.zone_id}")
            print(f"    Processed {len(zone_result.spot_details)} spots")
            
            # Test JSON output
            json_output = detector.get_json_output({first_zone: zone_result})
            parsed = json.loads(json_output)
            print("  ✅ JSON output format valid")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Detection pipeline error: {e}")
        return False

def validate_web_dashboard():
    """Test web dashboard components"""
    print("\n🌐 Validating web dashboard...")
    
    try:
        # Check Flask availability
        import flask
        print("  ✅ Flask available")
        
        # Check dashboard file
        dashboard_path = "apps/parking_dashboard.py"
        if Path(dashboard_path).exists():
            print("  ✅ Dashboard script exists")
        else:
            print("  ❌ Dashboard script missing")
            return False
            
        # Test dashboard initialization (without starting server)
        sys.path.append('apps')
        try:
            from parking_dashboard import ParkingWebAPI
            api = ParkingWebAPI()
            print("  ✅ Web API can be initialized")
        except Exception as e:
            print(f"  ⚠️  Web API initialization issue: {e}")
            
        return True
        
    except ImportError:
        print("  ❌ Flask not available - install with: pip install flask flask-cors")
        return False

def run_performance_test():
    """Run basic performance test"""
    print("\n⚡ Running performance test...")
    
    try:
        from modules.parking_detection import ParkingDetector
        
        detector = ParkingDetector("parking_dataset/config/parking_zones.yaml")
        
        # Create larger test frame
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        if detector.config.get('zones'):
            first_zone = list(detector.config['zones'].keys())[0]
            first_camera = detector.config['zones'][first_zone]['camera_ids'][0]
            
            frames = {first_camera: test_frame}
            
            # Measure processing time
            start_time = time.time()
            zone_result = detector.process_zone(frames, first_zone)
            processing_time = time.time() - start_time
            
            print(f"  📊 Performance Results:")
            print(f"    Frame size: {test_frame.shape}")
            print(f"    Spots processed: {len(zone_result.spot_details)}")
            print(f"    Processing time: {processing_time:.3f}s")
            
            if processing_time < 1.0:
                print("  ✅ Meets < 1 second requirement")
            else:
                print("  ⚠️  Exceeds 1 second requirement")
                
            return processing_time < 1.0
            
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def generate_validation_report():
    """Generate final validation report"""
    print("\n📋 Generating validation report...")
    
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "system_status": "validated",
        "components": {},
        "recommendations": []
    }
    
    # Run all validations
    validations = [
        ("components", validate_system_components),
        ("configuration", validate_configuration),
        ("models", validate_model_availability),
        ("detection", validate_detection_pipeline),
        ("dashboard", validate_web_dashboard),
        ("performance", run_performance_test)
    ]
    
    all_passed = True
    for name, validator in validations:
        try:
            result = validator()
            report["components"][name] = "PASS" if result else "FAIL"
            if not result:
                all_passed = False
        except Exception as e:
            report["components"][name] = f"ERROR: {e}"
            all_passed = False
            
    report["system_status"] = "READY" if all_passed else "NEEDS_ATTENTION"
    
    # Add recommendations
    if not all_passed:
        report["recommendations"].extend([
            "Fix failed components before deployment",
            "Ensure all required dependencies are installed",
            "Verify configuration file format and content"
        ])
    else:
        report["recommendations"].extend([
            "System ready for training data collection",
            "Consider running model calibration for better accuracy",
            "Test with real camera feeds before production deployment"
        ])
    
    # Save report
    report_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"  ✅ Report saved: {report_path}")
    return report

def main():
    """Main validation function"""
    print("🚗 YOLO26 Parking Detection System Validation")
    print("=" * 50)
    
    # Run complete validation
    report = generate_validation_report()
    
    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"  Status: {report['system_status']}")
    print(f"  Timestamp: {report['validation_timestamp']}")
    
    print(f"\n🔧 Component Status:")
    for component, status in report['components'].items():
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"  {status_icon} {component}: {status}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    if report['system_status'] == 'READY':
        print(f"\n🎉 System validation completed successfully!")
        print(f"   Your parking detection system is ready to use.")
        print(f"\n🚀 Next steps:")
        print(f"   1. Run: python parking_dataset/create_dataset.py")
        print(f"   2. Run: python parking_dataset/train_parking_model.py") 
        print(f"   3. Run: python apps/parking_dashboard.py")
        print(f"   4. Open: http://localhost:5000")
    else:
        print(f"\n⚠️  System needs attention before deployment.")
        print(f"   Please address the failed components above.")
    
    return report['system_status'] == 'READY'

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
