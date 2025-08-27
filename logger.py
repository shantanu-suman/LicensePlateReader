"""Logging utility for ANPR detection events."""

import csv
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

class ANPRLogger:
    def __init__(self, csv_file: str = "detection_logs.csv", json_file: str = "detection_logs.json"):
        """Initialize ANPR logger."""
        self.csv_file = csv_file
        self.json_file = json_file
        self.init_csv_file()
    
    def init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Timestamp', 'Number_Plate', 'Status', 'Confidence', 'Camera_Index'])
                logging.info(f"Initialized CSV log file: {self.csv_file}")
            except Exception as e:
                logging.error(f"Error initializing CSV file: {e}")
    
    def log_detection(self, plate_number: str, is_registered: bool, confidence: float = 0.0, 
                     camera_index: int = 0, additional_data: Optional[Dict] = None):
        """Log a detection event to both CSV and JSON."""
        timestamp = datetime.now().isoformat()
        status = "Registered" if is_registered else "Unregistered"
        
        # Log to CSV
        self.log_to_csv(timestamp, plate_number, status, confidence, camera_index)
        
        # Log to JSON
        self.log_to_json(timestamp, plate_number, status, confidence, camera_index, additional_data)
    
    def log_to_csv(self, timestamp: str, plate_number: str, status: str, 
                   confidence: float, camera_index: int):
        """Log detection to CSV file."""
        try:
            with open(self.csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, plate_number, status, confidence, camera_index])
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
    
    def log_to_json(self, timestamp: str, plate_number: str, status: str, 
                    confidence: float, camera_index: int, additional_data: Optional[Dict] = None):
        """Log detection to JSON file."""
        try:
            log_entry = {
                'timestamp': timestamp,
                'plate_number': plate_number,
                'status': status,
                'confidence': confidence,
                'camera_index': camera_index
            }
            
            if additional_data:
                log_entry.update(additional_data)
            
            # Read existing data
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r') as file:
                    try:
                        data = json.load(file)
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []
            
            # Append new entry
            data.append(log_entry)
            
            # Keep only last 1000 entries to prevent file from growing too large
            if len(data) > 1000:
                data = data[-1000:]
            
            # Write back to file
            with open(self.json_file, 'w') as file:
                json.dump(data, file, indent=2)
                
        except Exception as e:
            logging.error(f"Error writing to JSON: {e}")
    
    def get_recent_logs(self, limit: int = 50) -> pd.DataFrame:
        """Get recent detection logs as DataFrame."""
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                return df.tail(limit).sort_values('Timestamp', ascending=False)
            else:
                return pd.DataFrame(columns=['Timestamp', 'Number_Plate', 'Status', 'Confidence', 'Camera_Index'])
        except Exception as e:
            logging.error(f"Error reading logs: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        try:
            df = self.get_recent_logs(1000)  # Get more data for statistics
            if df.empty:
                return {}
            
            stats = {
                'total_detections': len(df),
                'registered_count': len(df[df['Status'] == 'Registered']),
                'unregistered_count': len(df[df['Status'] == 'Unregistered']),
                'average_confidence': df['Confidence'].mean(),
                'unique_plates': df['Number_Plate'].nunique(),
                'detection_rate': len(df) / max(1, (datetime.now() - pd.to_datetime(df['Timestamp'].iloc[-1])).total_seconds() / 3600)  # per hour
            }
            
            return stats
        except Exception as e:
            logging.error(f"Error calculating statistics: {e}")
            return {}
    
    def clear_logs(self):
        """Clear all log files."""
        try:
            # Clear CSV
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Number_Plate', 'Status', 'Confidence', 'Camera_Index'])
            
            # Clear JSON
            with open(self.json_file, 'w') as file:
                json.dump([], file)
                
            logging.info("Log files cleared")
        except Exception as e:
            logging.error(f"Error clearing logs: {e}")
