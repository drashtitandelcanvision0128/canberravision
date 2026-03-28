"""
Result Formatter Module for Unified Detection System
Formats detection results into strict JSON output format
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .unified_detector import (
    UnifiedDetectionResult, 
    VehicleInfo, 
    PPEInfo, 
    PlateInfo, 
    ParkingSlotInfo
)


class ResultFormatter:
    """
    Result Formatter for Unified Detection System
    Converts detection results to strict JSON format
    """
    
    def __init__(self, include_metadata: bool = True):
        """
        Initialize Result Formatter
        
        Args:
            include_metadata: Whether to include metadata in output
        """
        self.include_metadata = include_metadata
    
    def format_result(self, result: UnifiedDetectionResult) -> Dict[str, Any]:
        """
        Format UnifiedDetectionResult to strict JSON output
        
        Args:
            result: Detection result to format
            
        Returns:
            Formatted JSON-compatible dictionary
        """
        output = {
            "source": result.source,
            "timestamp": result.timestamp,
            "detections": {
                "ppe": self._format_ppe(result.ppe_detections),
                "vehicles": self._format_vehicles(result.vehicle_detections),
                "number_plates": self._format_plates(result.plate_detections),
                "parking": self._format_parking(result.parking_detections)
            }
        }
        
        # Add metadata if enabled
        if self.include_metadata:
            output["metadata"] = {
                "frame_number": result.frame_number,
                "processing_time_ms": round(result.processing_time_ms, 2),
                "total_detections": (
                    len(result.ppe_detections) +
                    len(result.vehicle_detections) +
                    len(result.plate_detections) +
                    len(result.parking_detections)
                )
            }
        
        return output
    
    def _format_ppe(self, ppe_detections: List[PPEInfo]) -> List[Dict]:
        """Format PPE detections"""
        formatted = []
        
        for ppe in ppe_detections:
            item = {
                "person_id": ppe.person_id,
                "helmet": bool(ppe.helmet),
                "seatbelt": bool(ppe.seatbelt),
                "vest": bool(ppe.vest),
                "confidence": round(float(ppe.confidence), 2),
                "bbox": [round(float(x), 2) for x in ppe.bbox],
                "vehicle_type": ppe.vehicle_type
            }
            
            # Add association if available
            if ppe.associated_vehicle_id:
                item["associated_vehicle_id"] = ppe.associated_vehicle_id
            
            formatted.append(item)
        
        return formatted
    
    def _format_vehicles(self, vehicles: List[VehicleInfo]) -> List[Dict]:
        """Format vehicle detections"""
        formatted = []
        
        for vehicle in vehicles:
            item = {
                "vehicle_id": vehicle.vehicle_id,
                "type": vehicle.vehicle_type,
                "color": vehicle.color,
                "confidence": round(float(vehicle.confidence), 2),
                "bbox": [round(float(x), 2) for x in vehicle.bbox]
            }
            
            # Add associated persons if available
            if vehicle.associated_persons:
                item["associated_persons"] = vehicle.associated_persons
            
            formatted.append(item)
        
        return formatted
    
    def _format_plates(self, plates: List[PlateInfo]) -> List[Dict]:
        """Format number plate detections"""
        formatted = []
        
        for plate in plates:
            item = {
                "text": plate.text,
                "confidence": round(float(plate.confidence), 2),
                "bbox": [round(float(x), 2) for x in plate.bbox]
            }
            
            # Add association if available
            if plate.associated_vehicle_id:
                item["associated_vehicle_id"] = plate.associated_vehicle_id
            
            formatted.append(item)
        
        return formatted
    
    def _format_parking(self, slots: List[ParkingSlotInfo]) -> List[Dict]:
        """Format parking slot detections"""
        formatted = []
        
        for slot in slots:
            item = {
                "slot_id": slot.slot_id,
                "occupied": bool(slot.occupied),
                "confidence": round(float(slot.confidence), 2),
                "bbox": [round(float(x), 2) for x in slot.bbox]
            }
            
            # Add association if available
            if slot.associated_vehicle_id:
                item["associated_vehicle_id"] = slot.associated_vehicle_id
            
            formatted.append(item)
        
        return formatted
    
    def to_json(self, result: UnifiedDetectionResult, indent: int = 2) -> str:
        """
        Convert result to JSON string
        
        Args:
            result: Detection result
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        formatted = self.format_result(result)
        return json.dumps(formatted, indent=indent, ensure_ascii=False)
    
    def save_json(self, result: UnifiedDetectionResult, 
                  output_path: str, indent: int = 2) -> str:
        """
        Save result to JSON file
        
        Args:
            result: Detection result
            output_path: Path to save JSON file
            indent: JSON indentation
            
        Returns:
            Path to saved file
        """
        formatted = self.format_result(result)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, indent=indent, ensure_ascii=False)
        
        return str(output_path)
    
    def format_batch(self, results: List[UnifiedDetectionResult]) -> Dict[str, Any]:
        """
        Format multiple results (for batch processing)
        
        Args:
            results: List of detection results
            
        Returns:
            Formatted batch result
        """
        batch_output = {
            "batch_timestamp": datetime.now().isoformat(),
            "frame_count": len(results),
            "frames": []
        }
        
        for result in results:
            batch_output["frames"].append(self.format_result(result))
        
        # Add summary statistics
        batch_output["summary"] = self._calculate_batch_summary(results)
        
        return batch_output
    
    def _calculate_batch_summary(self, results: List[UnifiedDetectionResult]) -> Dict:
        """Calculate summary statistics for batch results"""
        total_ppe = sum(len(r.ppe_detections) for r in results)
        total_vehicles = sum(len(r.vehicle_detections) for r in results)
        total_plates = sum(len(r.plate_detections) for r in results)
        total_parking = sum(len(r.parking_detections) for r in results)
        
        avg_processing_time = (
            sum(r.processing_time_ms for r in results) / len(results)
            if results else 0
        )
        
        # Count unique vehicles
        unique_vehicles = set()
        for result in results:
            for vehicle in result.vehicle_detections:
                unique_vehicles.add(vehicle.vehicle_id)
        
        # Count unique persons
        unique_persons = set()
        for result in results:
            for person in result.ppe_detections:
                unique_persons.add(person.person_id)
        
        # Count unique plates
        unique_plates = set()
        for result in results:
            for plate in result.plate_detections:
                unique_plates.add(plate.text)
        
        return {
            "total_frames": len(results),
            "total_ppe_detections": total_ppe,
            "total_vehicle_detections": total_vehicles,
            "total_plate_detections": total_plates,
            "total_parking_detections": total_parking,
            "unique_vehicles": len(unique_vehicles),
            "unique_persons": len(unique_persons),
            "unique_plates": len(unique_plates),
            "avg_processing_time_ms": round(avg_processing_time, 2)
        }
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate that output matches required schema
        
        Args:
            output: Formatted output to validate
            
        Returns:
            True if valid
        """
        required_keys = ['source', 'timestamp', 'detections']
        
        # Check top-level keys
        for key in required_keys:
            if key not in output:
                return False
        
        # Check detections structure
        detections = output.get('detections', {})
        detection_types = ['ppe', 'vehicles', 'number_plates', 'parking']
        
        for det_type in detection_types:
            if det_type not in detections:
                return False
            
            if not isinstance(detections[det_type], list):
                return False
        
        return True


# Utility functions for quick formatting
def format_single_result(result: UnifiedDetectionResult, 
                         as_json: bool = False) -> Union[Dict, str]:
    """
    Quick format a single result
    
    Args:
        result: Detection result
        as_json: Return as JSON string instead of dict
        
    Returns:
        Formatted result as dict or JSON string
    """
    formatter = ResultFormatter()
    
    if as_json:
        return formatter.to_json(result)
    else:
        return formatter.format_result(result)


def format_batch_results(results: List[UnifiedDetectionResult],
                         as_json: bool = False) -> Union[Dict, str]:
    """
    Quick format batch results
    
    Args:
        results: List of detection results
        as_json: Return as JSON string instead of dict
        
    Returns:
        Formatted results as dict or JSON string
    """
    formatter = ResultFormatter()
    batch_output = formatter.format_batch(results)
    
    if as_json:
        return json.dumps(batch_output, indent=2, ensure_ascii=False)
    else:
        return batch_output


def print_formatted_result(result: UnifiedDetectionResult):
    """Print a formatted result to console"""
    formatter = ResultFormatter()
    print(formatter.to_json(result))


if __name__ == "__main__":
    print("[INFO] Result Formatter Module")
    print("[INFO] Usage: formatter = ResultFormatter()")
    print("       output = formatter.format_result(result)")
