"""
Vehicle Database and Alert System
Manages vehicle information, matching, and alert generation.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import os

class VehicleDatabase:
    """Vehicle database management system."""
    
    def __init__(self, db_path: str = "database/vehicles.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
        self._load_sample_data()
    
    def _initialize_database(self):
        """Initialize SQLite database."""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    license_plate TEXT UNIQUE NOT NULL,
                    make TEXT,
                    model TEXT,
                    color TEXT,
                    year INTEGER,
                    owner_name TEXT,
                    owner_contact TEXT,
                    registration_date TEXT,
                    expiry_date TEXT,
                    is_stolen BOOLEAN DEFAULT 0,
                    alert_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    license_plate TEXT NOT NULL,
                    location TEXT,
                    speed REAL,
                    direction TEXT,
                    camera_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    confidence REAL,
                    FOREIGN KEY (license_plate) REFERENCES vehicles (license_plate)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    license_plate TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT DEFAULT 'medium',
                    message TEXT,
                    is_resolved BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (license_plate) REFERENCES vehicles (license_plate)
                )
            ''')
            
            self.connection.commit()
            print("[INFO] Vehicle database initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Database initialization failed: {e}")
    
    def _load_sample_data(self):
        """Load sample vehicle data."""
        sample_vehicles = [
            {
                'license_plate': 'LD62 WRC',
                'make': 'BMW',
                'model': '3 Series',
                'color': 'Silver',
                'year': 2019,
                'owner_name': 'John Smith',
                'owner_contact': '+44 20 7123 4567',
                'is_stolen': False,
                'alert_reason': None
            },
            {
                'license_plate': 'YY15 FUD',
                'make': 'Audi',
                'model': 'A4',
                'color': 'Black',
                'year': 2015,
                'owner_name': 'Emma Wilson',
                'owner_contact': '+44 20 2345 6789',
                'is_stolen': False,
                'alert_reason': None
            },
            {
                'license_plate': 'XX12 GHD',
                'make': 'Mercedes',
                'model': 'C-Class',
                'color': 'White',
                'year': 2012,
                'owner_name': 'Michael Brown',
                'owner_contact': '+44 20 3456 7890',
                'is_stolen': False,
                'alert_reason': None
            },
            {
                'license_plate': 'AB12 GHT',
                'make': 'Toyota',
                'model': 'Camry',
                'color': 'Blue',
                'year': 2012,
                'owner_name': 'Unknown',
                'owner_contact': None,
                'is_stolen': True,
                'alert_reason': 'Reported stolen'
            },
            {
                'license_plate': 'KP09 ZXE',
                'make': 'Honda',
                'model': 'Civic',
                'color': 'Red',
                'year': 2009,
                'owner_name': 'Sarah Davis',
                'owner_contact': '+44 20 4567 8901',
                'is_stolen': False,
                'alert_reason': None
            },
            # Add some common Indian license plates
            {
                'license_plate': 'MH01AB1234',
                'make': 'Maruti',
                'model': 'Swift',
                'color': 'White',
                'year': 2020,
                'owner_name': 'Rajesh Kumar',
                'owner_contact': '+91 98765 43210',
                'is_stolen': False,
                'alert_reason': None
            },
            {
                'license_plate': 'DL12CD5678',
                'make': 'Hyundai',
                'model': 'i20',
                'color': 'Silver',
                'year': 2021,
                'owner_name': 'Priya Sharma',
                'owner_contact': '+91 87654 32109',
                'is_stolen': False,
                'alert_reason': None
            },
            {
                'license_plate': 'KA03EF9012',
                'make': 'Tata',
                'model': 'Nexon',
                'color': 'Blue',
                'year': 2022,
                'owner_name': 'Amit Patel',
                'owner_contact': '+91 76543 21098',
                'is_stolen': True,
                'alert_reason': 'Reported stolen - Police case #12345'
            }
        ]
        
        for vehicle in sample_vehicles:
            self.add_vehicle(vehicle)
    
    def add_vehicle(self, vehicle_data: Dict) -> bool:
        """Add a new vehicle to the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO vehicles 
                (license_plate, make, model, color, year, owner_name, owner_contact, 
                 is_stolen, alert_reason, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                vehicle_data.get('license_plate'),
                vehicle_data.get('make'),
                vehicle_data.get('model'),
                vehicle_data.get('color'),
                vehicle_data.get('year'),
                vehicle_data.get('owner_name'),
                vehicle_data.get('owner_contact'),
                vehicle_data.get('is_stolen', False),
                vehicle_data.get('alert_reason')
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to add vehicle: {e}")
            return False
    
    def get_vehicle(self, license_plate: str) -> Optional[Dict]:
        """Get vehicle information by license plate."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM vehicles WHERE license_plate = ?', (license_plate,))
            row = cursor.fetchone()
            
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to get vehicle: {e}")
            return None
    
    def search_vehicles(self, query: str, limit: int = 10) -> List[Dict]:
        """Search vehicles by license plate, make, or model."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT * FROM vehicles 
                WHERE license_plate LIKE ? OR make LIKE ? OR model LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            print(f"[ERROR] Failed to search vehicles: {e}")
            return []
    
    def record_sighting(self, sighting_data: Dict) -> bool:
        """Record a vehicle sighting."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO sightings 
                (license_plate, location, speed, direction, camera_id, image_path, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                sighting_data.get('license_plate'),
                sighting_data.get('location'),
                sighting_data.get('speed'),
                sighting_data.get('direction'),
                sighting_data.get('camera_id'),
                sighting_data.get('image_path'),
                sighting_data.get('confidence')
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to record sighting: {e}")
            return False
    
    def get_recent_sightings(self, license_plate: str = None, hours: int = 24) -> List[Dict]:
        """Get recent vehicle sightings."""
        try:
            cursor = self.connection.cursor()
            
            if license_plate:
                cursor.execute('''
                    SELECT * FROM sightings 
                    WHERE license_plate = ? AND timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(hours), (license_plate,))
            else:
                cursor.execute('''
                    SELECT * FROM sightings 
                    WHERE timestamp >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                    LIMIT 100
                '''.format(hours))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            print(f"[ERROR] Failed to get sightings: {e}")
            return []

class AlertSystem:
    """Alert management system for ANPR."""
    
    def __init__(self, database: VehicleDatabase):
        self.db = database
        self.alert_rules = self._initialize_alert_rules()
    
    def _initialize_alert_rules(self) -> Dict:
        """Initialize alert rules."""
        return {
            'stolen_vehicle': {
                'enabled': True,
                'severity': 'high',
                'message': 'Stolen vehicle detected'
            },
            'speed_violation': {
                'enabled': True,
                'severity': 'medium',
                'threshold': 80,  # km/h
                'message': 'Speed limit violation'
            },
            'unregistered_vehicle': {
                'enabled': True,
                'severity': 'low',
                'message': 'Unregistered vehicle detected'
            },
            'suspicious_activity': {
                'enabled': True,
                'severity': 'medium',
                'message': 'Suspicious vehicle activity detected'
            }
        }
    
    def check_alerts(self, license_plate: str, vehicle_data: Dict = None) -> List[Dict]:
        """Check if any alerts should be generated for a vehicle."""
        alerts = []
        
        try:
            # Get vehicle from database
            vehicle = self.db.get_vehicle(license_plate)
            
            if vehicle is None:
                # Unregistered vehicle alert
                if self.alert_rules['unregistered_vehicle']['enabled']:
                    alerts.append({
                        'type': 'unregistered_vehicle',
                        'severity': self.alert_rules['unregistered_vehicle']['severity'],
                        'message': self.alert_rules['unregistered_vehicle']['message'],
                        'license_plate': license_plate
                    })
            else:
                # Stolen vehicle alert
                if vehicle.get('is_stolen') and self.alert_rules['stolen_vehicle']['enabled']:
                    alerts.append({
                        'type': 'stolen_vehicle',
                        'severity': self.alert_rules['stolen_vehicle']['severity'],
                        'message': f"{self.alert_rules['stolen_vehicle']['message']}: {vehicle.get('alert_reason', 'Unknown reason')}",
                        'license_plate': license_plate
                    })
                
                # Speed violation alert
                if vehicle_data and vehicle_data.get('speed', 0) > self.alert_rules['speed_violation']['threshold']:
                    if self.alert_rules['speed_violation']['enabled']:
                        alerts.append({
                            'type': 'speed_violation',
                            'severity': self.alert_rules['speed_violation']['severity'],
                            'message': f"{self.alert_rules['speed_violation']['message']}: {vehicle_data.get('speed', 0)} km/h",
                            'license_plate': license_plate
                        })
                
                # Suspicious activity check (multiple sightings in short time)
                recent_sightings = self.db.get_recent_sightings(license_plate, hours=1)
                if len(recent_sightings) > 5 and self.alert_rules['suspicious_activity']['enabled']:
                    alerts.append({
                        'type': 'suspicious_activity',
                        'severity': self.alert_rules['suspicious_activity']['severity'],
                        'message': f"{self.alert_rules['suspicious_activity']['message']}: {len(recent_sightings)} sightings in last hour",
                        'license_plate': license_plate
                    })
            
            # Store alerts in database
            for alert in alerts:
                self._store_alert(alert)
            
            return alerts
            
        except Exception as e:
            print(f"[ERROR] Alert check failed: {e}")
            return []
    
    def _store_alert(self, alert: Dict):
        """Store alert in database."""
        try:
            cursor = self.db.connection.cursor()
            cursor.execute('''
                INSERT INTO alerts (license_plate, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            ''', (
                alert['license_plate'],
                alert['type'],
                alert['severity'],
                alert['message']
            ))
            
            self.db.connection.commit()
            
        except Exception as e:
            print(f"[ERROR] Failed to store alert: {e}")
    
    def get_active_alerts(self, hours: int = 24) -> List[Dict]:
        """Get active (unresolved) alerts."""
        try:
            cursor = self.db.connection.cursor()
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE is_resolved = 0 AND created_at >= datetime('now', '-{} hours')
                ORDER BY created_at DESC
            '''.format(hours))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            return [dict(zip(columns, row)) for row in rows]
            
        except Exception as e:
            print(f"[ERROR] Failed to get active alerts: {e}")
            return []
    
    def resolve_alert(self, alert_id: int) -> bool:
        """Mark an alert as resolved."""
        try:
            cursor = self.db.connection.cursor()
            cursor.execute('''
                UPDATE alerts 
                SET is_resolved = 1, resolved_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (alert_id,))
            
            self.db.connection.commit()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to resolve alert: {e}")
            return False

class VehicleMatcher:
    """Vehicle matching and similarity scoring system."""
    
    def __init__(self, database: VehicleDatabase):
        self.db = database
    
    def find_similar_vehicles(self, query_plate: str, threshold: float = 0.7) -> List[Dict]:
        """Find vehicles with similar license plates."""
        try:
            # Clean and normalize the query plate
            cleaned_query = self._clean_license_plate(query_plate)
            
            # Get all vehicles from database
            cursor = self.db.connection.cursor()
            cursor.execute('SELECT * FROM vehicles')
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            matches = []
            
            for row in rows:
                vehicle = dict(zip(columns, row))
                db_plate = vehicle['license_plate']
                cleaned_db = self._clean_license_plate(db_plate)
                
                # Calculate similarity score
                similarity = self._calculate_similarity(cleaned_query, cleaned_db)
                
                if similarity >= threshold:
                    matches.append({
                        'vehicle': vehicle,
                        'similarity': similarity,
                        'matched_plate': db_plate
                    })
            
            # Sort by similarity (highest first)
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return matches
            
        except Exception as e:
            print(f"[ERROR] Similar vehicle search failed: {e}")
            return []
    
    def _clean_license_plate(self, plate: str) -> str:
        """Clean and normalize license plate text."""
        if not plate:
            return ""
        
        # Remove spaces and convert to uppercase
        cleaned = plate.replace(' ', '').upper()
        
        # Remove common OCR errors
        replacements = {
            '0': 'O',
            '1': 'I',
            '8': 'B',
            '5': 'S'
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def _calculate_similarity(self, plate1: str, plate2: str) -> float:
        """Calculate similarity between two license plates."""
        if not plate1 or not plate2:
            return 0.0
        
        # Exact match
        if plate1 == plate2:
            return 1.0
        
        # Levenshtein distance
        distance = self._levenshtein_distance(plate1, plate2)
        max_len = max(len(plate1), len(plate2))
        
        if max_len == 0:
            return 0.0
        
        similarity = 1.0 - (distance / max_len)
        return similarity
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

# Test the modules
if __name__ == "__main__":
    # Initialize database and systems
    db = VehicleDatabase()
    alert_system = AlertSystem(db)
    matcher = VehicleMatcher(db)
    
    print("[INFO] Vehicle database and alert system initialized")
    
    # Test vehicle lookup
    vehicle = db.get_vehicle('LD62 WRC')
    if vehicle:
        print(f"[INFO] Found vehicle: {vehicle['make']} {vehicle['model']}")
    
    # Test alert system
    alerts = alert_system.check_alerts('AB12 GHT', {'speed': 95})
    print(f"[INFO] Generated {len(alerts)} alerts")
    
    # Test vehicle matching
    matches = matcher.find_similar_vehicles('LD62 WRC')
    print(f"[INFO] Found {len(matches)} similar vehicles")
