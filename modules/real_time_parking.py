"""
Real-time Parking Monitoring System
Provides continuous monitoring of parking spaces with live dashboard and alerts
"""

import cv2
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import queue
import logging

from .parking_detection import ParkingDetector, RealTimeParkingSystem, ParkingSpot, ZoneResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ParkingAlert:
    """Represents a parking system alert"""
    alert_id: str
    alert_type: str  # HIGH_OCCUPANCY, LOW_OCCUPANCY, CAMERA_OFFLINE, SYSTEM_ERROR
    zone_id: str
    message: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: str
    resolved: bool = False
    resolved_at: Optional[str] = None

@dataclass
class SystemStatus:
    """Overall system status"""
    total_zones: int
    active_zones: int
    total_cameras: int
    active_cameras: int
    total_spots: int
    occupied_spots: int
    empty_spots: int
    overall_occupancy_rate: float
    system_uptime: str
    last_update: str
    alerts_count: int

class ParkingDashboard:
    """Real-time parking dashboard with visualization"""
    
    def __init__(self, config_path: str = None):
        self.parking_system = RealTimeParkingSystem(config_path)
        self.alerts = []
        self.alert_queue = queue.Queue()
        self.status_history = []
        self.is_running = False
        self.dashboard_thread = None
        self.alert_thread = None
        
        # Alert thresholds
        self.high_occupancy_threshold = 90.0  # %
        self.low_occupancy_threshold = 10.0   # %
        self.camera_offline_timeout = 30      # seconds
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_camera_update = {}
        
        # Callbacks for external integration
        self.alert_callbacks = []
        self.update_callbacks = []
        
    def add_alert_callback(self, callback: Callable[[ParkingAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
        
    def add_update_callback(self, callback: Callable[[Dict[str, ZoneResult]], None]):
        """Add callback for parking updates"""
        self.update_callbacks.append(callback)
        
    def start_monitoring(self, camera_configs: Dict[str, str]):
        """Start real-time monitoring with dashboard"""
        try:
            logger.info("Starting parking monitoring system...")
            
            # Connect to cameras
            self.parking_system.connect_cameras(camera_configs)
            
            # Start parking system monitoring
            self.parking_system.start_monitoring(callback=self._on_parking_update)
            
            # Start alert monitoring
            self.is_running = True
            self.alert_thread = threading.Thread(target=self._alert_monitoring_loop)
            self.alert_thread.daemon = True
            self.alert_thread.start()
            
            logger.info("Parking monitoring system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
            
    def stop_monitoring(self):
        """Stop all monitoring systems"""
        logger.info("Stopping parking monitoring system...")
        
        self.is_running = False
        self.parking_system.stop_monitoring()
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
            
        logger.info("Parking monitoring system stopped")
        
    def _on_parking_update(self, results: Dict[str, ZoneResult]):
        """Handle parking update callback"""
        try:
            # Update camera timestamps
            current_time = time.time()
            for zone_result in results.values():
                for spot in zone_result.spot_details:
                    self.last_camera_update[spot.camera_id] = current_time
                    
            # Call update callbacks
            for callback in self.update_callbacks:
                try:
                    callback(results)
                except Exception as e:
                    logger.error(f"Update callback error: {e}")
                    
            # Store status history
            self._update_status_history(results)
            
            # Check for alerts
            self._check_alerts(results)
            
        except Exception as e:
            logger.error(f"Parking update error: {e}")
            self._create_system_alert("SYSTEM_ERROR", f"Update processing error: {e}")
            
    def _update_status_history(self, results: Dict[str, ZoneResult]):
        """Update status history for analytics"""
        try:
            total_spots = sum(zone.total_spots for zone in results.values())
            occupied_spots = sum(zone.occupied_spots for zone in results.values())
            empty_spots = sum(zone.empty_spots for zone in results.values())
            overall_occupancy = (occupied_spots / total_spots * 100) if total_spots > 0 else 0
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'total_spots': total_spots,
                'occupied_spots': occupied_spots,
                'empty_spots': empty_spots,
                'occupancy_rate': round(overall_occupancy, 1),
                'zone_count': len(results),
                'zones': {zone_id: {
                    'occupied': zone.occupied_spots,
                    'empty': zone.empty_spots,
                    'occupancy_rate': zone.occupancy_rate
                } for zone_id, zone in results.items()}
            }
            
            self.status_history.append(status)
            
            # Keep only last 24 hours of history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.status_history = [
                s for s in self.status_history 
                if datetime.fromisoformat(s['timestamp']) > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Status history update error: {e}")
            
    def _check_alerts(self, results: Dict[str, ZoneResult]):
        """Check for alert conditions"""
        try:
            current_time = time.time()
            
            # Check camera connectivity
            for zone_result in results.values():
                for spot in zone_result.spot_details:
                    camera_id = spot.camera_id
                    last_update = self.last_camera_update.get(camera_id, 0)
                    
                    if current_time - last_update > self.camera_offline_timeout:
                        self._create_camera_alert(camera_id, "CAMERA_OFFLINE", 
                                                f"Camera {camera_id} appears to be offline")
                        
            # Check occupancy thresholds
            for zone_id, zone_result in results.items():
                occupancy_rate = zone_result.occupancy_rate
                
                if occupancy_rate >= self.high_occupancy_threshold:
                    self._create_zone_alert(zone_id, "HIGH_OCCUPANCY",
                                          f"Zone {zone_id} has high occupancy: {occupancy_rate:.1f}%")
                                          
                elif occupancy_rate <= self.low_occupancy_threshold:
                    self._create_zone_alert(zone_id, "LOW_OCCUPANCY",
                                          f"Zone {zone_id} has low occupancy: {occupancy_rate:.1f}%")
                                          
        except Exception as e:
            logger.error(f"Alert checking error: {e}")
            
    def _create_zone_alert(self, zone_id: str, alert_type: str, message: str):
        """Create zone-specific alert"""
        alert_id = f"{zone_id}_{alert_type}_{int(time.time())}"
        severity = "HIGH" if "HIGH" in alert_type else "MEDIUM"
        
        alert = ParkingAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            zone_id=zone_id,
            message=message,
            severity=severity,
            timestamp=datetime.now().isoformat()
        )
        
        self._add_alert(alert)
        
    def _create_camera_alert(self, camera_id: str, alert_type: str, message: str):
        """Create camera-specific alert"""
        alert_id = f"{camera_id}_{alert_type}_{int(time.time())}"
        
        alert = ParkingAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            zone_id="ALL",
            message=message,
            severity="CRITICAL",
            timestamp=datetime.now().isoformat()
        )
        
        self._add_alert(alert)
        
    def _create_system_alert(self, alert_type: str, message: str):
        """Create system-level alert"""
        alert_id = f"SYSTEM_{alert_type}_{int(time.time())}"
        
        alert = ParkingAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            zone_id="SYSTEM",
            message=message,
            severity="HIGH",
            timestamp=datetime.now().isoformat()
        )
        
        self._add_alert(alert)
        
    def _add_alert(self, alert: ParkingAlert):
        """Add alert and trigger callbacks"""
        # Check if similar alert already exists and is not resolved
        existing_similar = [
            a for a in self.alerts 
            if a.alert_type == alert.alert_type 
            and a.zone_id == alert.zone_id 
            and not a.resolved
        ]
        
        if existing_similar:
            return  # Don't duplicate alerts
            
        self.alerts.append(alert)
        self.alert_queue.put(alert)
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
                
        logger.warning(f"ALERT: {alert.message}")
        
    def _alert_monitoring_loop(self):
        """Background thread for alert monitoring"""
        while self.is_running:
            try:
                # Process alert queue
                try:
                    alert = self.alert_queue.get(timeout=1)
                    self._process_alert(alert)
                except queue.Empty:
                    pass
                    
                # Auto-resolve old alerts
                self._auto_resolve_alerts()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                time.sleep(10)
                
    def _process_alert(self, alert: ParkingAlert):
        """Process individual alert"""
        # Here you could add:
        # - Email notifications
        # - SMS alerts
        # - Push notifications
        # - External API calls
        logger.info(f"Processing alert: {alert.alert_id} - {alert.message}")
        
    def _auto_resolve_alerts(self):
        """Automatically resolve old alerts"""
        current_time = datetime.now()
        resolved_threshold = timedelta(minutes=5)
        
        for alert in self.alerts:
            if not alert.resolved:
                alert_time = datetime.fromisoformat(alert.timestamp)
                if current_time - alert_time > resolved_threshold:
                    # Check if condition still exists
                    if not self._alert_condition_exists(alert):
                        alert.resolved = True
                        alert.resolved_at = current_time.isoformat()
                        logger.info(f"Auto-resolved alert: {alert.alert_id}")
                        
    def _alert_condition_exists(self, alert: ParkingAlert) -> bool:
        """Check if alert condition still exists"""
        try:
            current_results = self.parking_system.get_current_results()
            
            if alert.alert_type == "HIGH_OCCUPANCY":
                if alert.zone_id in current_results:
                    return current_results[alert.zone_id].occupancy_rate >= self.high_occupancy_threshold
                    
            elif alert.alert_type == "LOW_OCCUPANCY":
                if alert.zone_id in current_results:
                    return current_results[alert.zone_id].occupancy_rate <= self.low_occupancy_threshold
                    
            elif alert.alert_type == "CAMERA_OFFLINE":
                return False  # Camera offline alerts auto-resolve when camera comes back
                
            return False
            
        except Exception:
            return False
            
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        try:
            current_results = self.parking_system.get_current_results()
            
            total_zones = len(self.parking_system.detector.config['zones'])
            active_zones = len(current_results)
            
            total_cameras = sum(len(zone['camera_ids']) for zone in self.parking_system.detector.config['zones'].values())
            active_cameras = len(self.parking_system.camera_connections)
            
            total_spots = sum(zone.total_spots for zone in current_results.values())
            occupied_spots = sum(zone.occupied_spots for zone in current_results.values())
            empty_spots = sum(zone.empty_spots for zone in current_results.values())
            overall_occupancy = (occupied_spots / total_spots * 100) if total_spots > 0 else 0
            
            uptime = datetime.now() - self.start_time
            uptime_str = str(uptime).split('.')[0]  # Remove microseconds
            
            active_alerts = len([a for a in self.alerts if not a.resolved])
            
            return SystemStatus(
                total_zones=total_zones,
                active_zones=active_zones,
                total_cameras=total_cameras,
                active_cameras=active_cameras,
                total_spots=total_spots,
                occupied_spots=occupied_spots,
                empty_spots=empty_spots,
                overall_occupancy_rate=round(overall_occupancy, 1),
                system_uptime=uptime_str,
                last_update=datetime.now().isoformat(),
                alerts_count=active_alerts
            )
            
        except Exception as e:
            logger.error(f"System status error: {e}")
            return SystemStatus(
                total_zones=0, active_zones=0, total_cameras=0, active_cameras=0,
                total_spots=0, occupied_spots=0, empty_spots=0, overall_occupancy_rate=0,
                system_uptime="00:00:00", last_update=datetime.now().isoformat(), alerts_count=0
            )
            
    def get_active_alerts(self) -> List[ParkingAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now().isoformat()
                logger.info(f"Manually resolved alert: {alert_id}")
                return True
        return False
        
    def get_occupancy_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get occupancy trends for the specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_history = [
                status for status in self.status_history
                if datetime.fromisoformat(status['timestamp']) > cutoff_time
            ]
            
            if not recent_history:
                return {}
                
            # Calculate trends
            timestamps = [datetime.fromisoformat(s['timestamp']) for s in recent_history]
            occupancy_rates = [s['occupancy_rate'] for s in recent_history]
            
            # Zone-wise trends
            zone_trends = {}
            if recent_history:
                for zone_id in recent_history[-1]['zones'].keys():
                    zone_occupancy = [s['zones'].get(zone_id, {}).get('occupancy_rate', 0) 
                                    for s in recent_history if zone_id in s.get('zones', {})]
                    if zone_occupancy:
                        zone_trends[zone_id] = {
                            'current': zone_occupancy[-1] if zone_occupancy else 0,
                            'average': round(np.mean(zone_occupancy), 1) if zone_occupancy else 0,
                            'min': min(zone_occupancy) if zone_occupancy else 0,
                            'max': max(zone_occupancy) if zone_occupancy else 0
                        }
                        
            return {
                'time_period_hours': hours,
                'data_points': len(recent_history),
                'overall_trend': {
                    'current': occupancy_rates[-1] if occupancy_rates else 0,
                    'average': round(np.mean(occupancy_rates), 1) if occupancy_rates else 0,
                    'min': min(occupancy_rates) if occupancy_rates else 0,
                    'max': max(occupancy_rates) if occupancy_rates else 0,
                    'trend_direction': 'up' if len(occupancy_rates) > 1 and occupancy_rates[-1] > occupancy_rates[-2] else 'down'
                },
                'zone_trends': zone_trends,
                'timestamps': [t.isoformat() for t in timestamps],
                'occupancy_rates': occupancy_rates
            }
            
        except Exception as e:
            logger.error(f"Occupancy trends error: {e}")
            return {}
            
    def export_data(self, format_type: str = "json", hours: int = 24) -> str:
        """Export parking data in specified format"""
        try:
            current_results = self.parking_system.get_current_results()
            system_status = self.get_system_status()
            trends = self.get_occupancy_trends(hours)
            active_alerts = self.get_active_alerts()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'system_status': asdict(system_status),
                'current_results': {
                    zone_id: asdict(zone_result) for zone_id, zone_result in current_results.items()
                },
                'occupancy_trends': trends,
                'active_alerts': [asdict(alert) for alert in active_alerts],
                'configuration': {
                    'high_occupancy_threshold': self.high_occupancy_threshold,
                    'low_occupancy_threshold': self.low_occupancy_threshold,
                    'camera_offline_timeout': self.camera_offline_timeout
                }
            }
            
            if format_type.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Data export error: {e}")
            return "{}"
            
    def save_snapshot(self, output_dir: str = "parking_detections") -> str:
        """Save current parking detection snapshot"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON data
            json_data = self.export_data("json")
            json_file = output_path / f"parking_snapshot_{timestamp}.json"
            with open(json_file, 'w') as f:
                f.write(json_data)
                
            logger.info(f"Parking snapshot saved: {json_file}")
            return str(json_file)
            
        except Exception as e:
            logger.error(f"Snapshot save error: {e}")
            return ""

def main():
    """Main function for testing the real-time parking system"""
    print("=== YOLO26 Real-time Parking Dashboard ===")
    
    # Initialize dashboard
    dashboard = ParkingDashboard()
    
    # Define camera configurations (adjust for your setup)
    camera_configs = {
        'cam_01': 0,  # Webcam or RTSP stream
        'cam_02': 'rtsp://camera2_url',
        'cam_03': 'rtsp://camera3_url',
    }
    
    # Add alert callback
    def on_alert(alert: ParkingAlert):
        print(f"🚨 ALERT: {alert.severity} - {alert.message}")
        
    # Add update callback
    def on_update(results: Dict[str, ZoneResult]):
        for zone_id, zone_result in results.items():
            print(f"📊 Zone {zone_id}: {zone_result.occupied_spots}/{zone_result.total_spots} occupied ({zone_result.occupancy_rate:.1f}%)")
            
    dashboard.add_alert_callback(on_alert)
    dashboard.add_update_callback(on_update)
    
    try:
        # Start monitoring
        if dashboard.start_monitoring(camera_configs):
            print("✅ Monitoring started. Press Ctrl+C to stop...")
            
            # Keep running
            while True:
                time.sleep(10)
                
                # Print system status every 30 seconds
                status = dashboard.get_system_status()
                print(f"\n📈 System Status: {status.overall_occupancy_rate:.1f}% occupancy, "
                      f"{status.active_zones}/{status.total_zones} zones active, "
                      f"{status.alerts_count} active alerts")
                      
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring...")
        dashboard.stop_monitoring()
        print("✅ Monitoring stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        dashboard.stop_monitoring()

if __name__ == "__main__":
    main()
