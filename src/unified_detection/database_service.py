"""
Database Service Module for Unified Detection System
PostgreSQL integration for storing detection results
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("[WARNING] psycopg2 not available. Database features disabled.")

from .unified_detector import UnifiedDetectionResult


class DatabaseService:
    """
    Database Service for Unified Detection System
    Handles PostgreSQL connections and data persistence
    """
    
    def __init__(self, 
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 database: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 sslmode: str = 'disable'):
        """
        Initialize Database Service
        
        Args:
            host: Database host (from env if None)
            port: Database port (from env if None)
            database: Database name (from env if None)
            user: Database user (from env if None)
            password: Database password (from env if None)
            sslmode: SSL mode for connection
        """
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', '5432'))
        self.database = database or os.getenv('DB_NAME', 'canberraavisison_detection')
        self.user = user or os.getenv('DB_USER', 'postgres')
        self.password = password or os.getenv('DB_PASSWORD', 'admin')
        self.sslmode = sslmode or os.getenv('DB_SSLMODE', 'disable')
        
        self.connection = None
        self.enabled = POSTGRES_AVAILABLE
        
        if self.enabled:
            self._initialize_database()
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters"""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'sslmode': self.sslmode
        }
    
    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor"""
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(**self._get_connection_params())
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            yield cursor
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def _initialize_database(self):
        """Initialize database schema"""
        try:
            with self._get_cursor() as cursor:
                # Create detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS unified_detections (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        source VARCHAR(50) NOT NULL,
                        frame_number INTEGER DEFAULT 0,
                        source_id VARCHAR(255),
                        processing_time_ms FLOAT,
                        total_detections INTEGER DEFAULT 0,
                        raw_json JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create vehicle detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS vehicle_detections (
                        id SERIAL PRIMARY KEY,
                        detection_id INTEGER REFERENCES unified_detections(id) ON DELETE CASCADE,
                        vehicle_id VARCHAR(50) NOT NULL,
                        vehicle_type VARCHAR(20) NOT NULL,
                        color VARCHAR(20),
                        confidence FLOAT,
                        bbox JSONB,
                        associated_persons JSONB,
                        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create PPE detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ppe_detections (
                        id SERIAL PRIMARY KEY,
                        detection_id INTEGER REFERENCES unified_detections(id) ON DELETE CASCADE,
                        person_id VARCHAR(50) NOT NULL,
                        helmet BOOLEAN DEFAULT FALSE,
                        seatbelt BOOLEAN DEFAULT FALSE,
                        vest BOOLEAN DEFAULT FALSE,
                        confidence FLOAT,
                        bbox JSONB,
                        vehicle_type VARCHAR(20),
                        associated_vehicle_id VARCHAR(50),
                        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create number plate detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS plate_detections (
                        id SERIAL PRIMARY KEY,
                        detection_id INTEGER REFERENCES unified_detections(id) ON DELETE CASCADE,
                        plate_text VARCHAR(50) NOT NULL,
                        confidence FLOAT,
                        bbox JSONB,
                        associated_vehicle_id VARCHAR(50),
                        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create parking detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS parking_detections (
                        id SERIAL PRIMARY KEY,
                        detection_id INTEGER REFERENCES unified_detections(id) ON DELETE CASCADE,
                        slot_id INTEGER NOT NULL,
                        occupied BOOLEAN DEFAULT FALSE,
                        confidence FLOAT,
                        bbox JSONB,
                        associated_vehicle_id VARCHAR(50),
                        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
                    ON unified_detections(timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_source 
                    ON unified_detections(source)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_plate_text 
                    ON plate_detections(plate_text)
                """)
                
                print("[INFO] Database schema initialized successfully")
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize database: {e}")
            self.enabled = False
    
    def save_detection(self, result: UnifiedDetectionResult, 
                       source_id: Optional[str] = None) -> Optional[int]:
        """
        Save a detection result to database
        
        Args:
            result: Detection result to save
            source_id: Optional identifier for the source
            
        Returns:
            Detection ID if successful, None otherwise
        """
        if not self.enabled:
            print("[WARNING] Database not enabled. Skipping save.")
            return None
        
        try:
            with self._get_cursor() as cursor:
                # Insert main detection record
                total_detections = (
                    len(result.ppe_detections) +
                    len(result.vehicle_detections) +
                    len(result.plate_detections) +
                    len(result.parking_detections)
                )
                
                # Convert result to JSON
                from .result_formatter import ResultFormatter
                formatter = ResultFormatter()
                raw_json = formatter.format_result(result)
                
                cursor.execute("""
                    INSERT INTO unified_detections 
                    (timestamp, source, frame_number, source_id, 
                     processing_time_ms, total_detections, raw_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    result.timestamp,
                    result.source,
                    result.frame_number,
                    source_id,
                    result.processing_time_ms,
                    total_detections,
                    json.dumps(raw_json)
                ))
                
                detection_id = cursor.fetchone()['id']
                
                # Insert vehicle detections
                for vehicle in result.vehicle_detections:
                    cursor.execute("""
                        INSERT INTO vehicle_detections
                        (detection_id, vehicle_id, vehicle_type, color, 
                         confidence, bbox, associated_persons)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        detection_id,
                        vehicle.vehicle_id,
                        vehicle.vehicle_type,
                        vehicle.color,
                        vehicle.confidence,
                        json.dumps(vehicle.bbox),
                        json.dumps(vehicle.associated_persons) if vehicle.associated_persons else None
                    ))
                
                # Insert PPE detections
                for ppe in result.ppe_detections:
                    cursor.execute("""
                        INSERT INTO ppe_detections
                        (detection_id, person_id, helmet, seatbelt, vest,
                         confidence, bbox, vehicle_type, associated_vehicle_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        detection_id,
                        ppe.person_id,
                        ppe.helmet,
                        ppe.seatbelt,
                        ppe.vest,
                        ppe.confidence,
                        json.dumps(ppe.bbox),
                        ppe.vehicle_type,
                        ppe.associated_vehicle_id
                    ))
                
                # Insert plate detections
                for plate in result.plate_detections:
                    cursor.execute("""
                        INSERT INTO plate_detections
                        (detection_id, plate_text, confidence, bbox, associated_vehicle_id)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        detection_id,
                        plate.text,
                        plate.confidence,
                        json.dumps(plate.bbox),
                        plate.associated_vehicle_id
                    ))
                
                # Insert parking detections
                for parking in result.parking_detections:
                    cursor.execute("""
                        INSERT INTO parking_detections
                        (detection_id, slot_id, occupied, confidence, 
                         bbox, associated_vehicle_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        detection_id,
                        parking.slot_id,
                        parking.occupied,
                        parking.confidence,
                        json.dumps(parking.bbox),
                        parking.associated_vehicle_id
                    ))
                
                print(f"[INFO] Detection saved to database with ID: {detection_id}")
                return detection_id
                
        except Exception as e:
            print(f"[ERROR] Failed to save detection: {e}")
            return None
    
    def get_detections(self, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       source: Optional[str] = None,
                       limit: int = 100) -> List[Dict]:
        """
        Retrieve detections from database
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            source: Filter by source
            limit: Maximum number of results
            
        Returns:
            List of detection records
        """
        if not self.enabled:
            return []
        
        try:
            with self._get_cursor() as cursor:
                query = "SELECT * FROM unified_detections WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= %s"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= %s"
                    params.append(end_time)
                
                if source:
                    query += " AND source = %s"
                    params.append(source)
                
                query += " ORDER BY timestamp DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            print(f"[ERROR] Failed to retrieve detections: {e}")
            return []
    
    def get_detection_by_plate(self, plate_text: str) -> List[Dict]:
        """
        Find detections by license plate number
        
        Args:
            plate_text: License plate text to search
            
        Returns:
            List of matching detections
        """
        if not self.enabled:
            return []
        
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT d.*, p.plate_text, p.confidence as plate_confidence
                    FROM unified_detections d
                    JOIN plate_detections p ON d.id = p.detection_id
                    WHERE p.plate_text ILIKE %s
                    ORDER BY d.timestamp DESC
                """, (f"%{plate_text}%",))
                
                return cursor.fetchall()
                
        except Exception as e:
            print(f"[ERROR] Failed to search by plate: {e}")
            return []
    
    def get_violations(self, 
                       start_time: Optional[datetime] = None,
                       limit: int = 100) -> List[Dict]:
        """
        Get PPE violations (no helmet on 2-wheeler, no seatbelt on 4-wheeler)
        
        Args:
            start_time: Filter by start time
            limit: Maximum number of results
            
        Returns:
            List of violation records
        """
        if not self.enabled:
            return []
        
        try:
            with self._get_cursor() as cursor:
                # Find 2-wheeler violations (no helmet)
                query = """
                    SELECT d.*, p.person_id, p.helmet, p.seatbelt, 
                           p.vehicle_type, p.confidence
                    FROM unified_detections d
                    JOIN ppe_detections p ON d.id = p.detection_id
                    WHERE (
                        (p.vehicle_type = 'bike' AND p.helmet = FALSE)
                        OR 
                        (p.vehicle_type IN ('car', 'truck', 'bus') AND p.seatbelt = FALSE)
                    )
                """
                
                params = []
                
                if start_time:
                    query += " AND d.timestamp >= %s"
                    params.append(start_time)
                
                query += " ORDER BY d.timestamp DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            print(f"[ERROR] Failed to get violations: {e}")
            return []
    
    def get_statistics(self, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Dict:
        """
        Get detection statistics
        
        Args:
            start_time: Start time for statistics
            end_time: End time for statistics
            
        Returns:
            Statistics dictionary
        """
        if not self.enabled:
            return {}
        
        try:
            with self._get_cursor() as cursor:
                # Base query conditions
                time_condition = ""
                params = []
                
                if start_time:
                    time_condition += " AND timestamp >= %s"
                    params.append(start_time)
                
                if end_time:
                    time_condition += " AND timestamp <= %s"
                    params.append(end_time)
                
                # Total detections
                cursor.execute(f"""
                    SELECT COUNT(*) as total 
                    FROM unified_detections 
                    WHERE 1=1 {time_condition}
                """, params)
                total_detections = cursor.fetchone()['total']
                
                # Vehicle statistics
                cursor.execute(f"""
                    SELECT vehicle_type, COUNT(*) as count
                    FROM vehicle_detections v
                    JOIN unified_detections d ON v.detection_id = d.id
                    WHERE 1=1 {time_condition.replace('timestamp', 'd.timestamp')}
                    GROUP BY vehicle_type
                """, params)
                vehicle_stats = {row['vehicle_type']: row['count'] for row in cursor.fetchall()}
                
                # PPE statistics
                cursor.execute(f"""
                    SELECT 
                        SUM(CASE WHEN helmet THEN 1 ELSE 0 END) as helmets,
                        SUM(CASE WHEN seatbelt THEN 1 ELSE 0 END) as seatbelts,
                        SUM(CASE WHEN vest THEN 1 ELSE 0 END) as vests
                    FROM ppe_detections p
                    JOIN unified_detections d ON p.detection_id = d.id
                    WHERE 1=1 {time_condition.replace('timestamp', 'd.timestamp')}
                """, params)
                ppe_row = cursor.fetchone()
                ppe_stats = {
                    'helmets_detected': ppe_row['helmets'] or 0,
                    'seatbelts_detected': ppe_row['seatbelts'] or 0,
                    'vests_detected': ppe_row['vests'] or 0
                }
                
                # Unique plates
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT plate_text) as unique_plates
                    FROM plate_detections p
                    JOIN unified_detections d ON p.detection_id = d.id
                    WHERE 1=1 {time_condition.replace('timestamp', 'd.timestamp')}
                """, params)
                unique_plates = cursor.fetchone()['unique_plates']
                
                # Violations
                violations = self.get_violations(start_time, limit=1000)
                
                return {
                    'total_detections': total_detections,
                    'vehicle_statistics': vehicle_stats,
                    'ppe_statistics': ppe_stats,
                    'unique_plates': unique_plates,
                    'violations_count': len(violations),
                    'time_range': {
                        'start': start_time.isoformat() if start_time else None,
                        'end': end_time.isoformat() if end_time else None
                    }
                }
                
        except Exception as e:
            print(f"[ERROR] Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            print("[INFO] Database connection closed")


# Global database service instance
_db_service = None


def get_database_service(**kwargs) -> DatabaseService:
    """Get or create global database service instance"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService(**kwargs)
    return _db_service


if __name__ == "__main__":
    print("[INFO] Database Service Module")
    print(f"[INFO] PostgreSQL available: {POSTGRES_AVAILABLE}")
    print("[INFO] Usage: db = DatabaseService()")
