"""Database manager for vehicle registration system."""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional
import logging

class DatabaseManager:
    def __init__(self, db_path: str = "vehicles.db"):
        """Initialize database manager with SQLite database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create vehicles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS vehicles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT UNIQUE NOT NULL,
                        owner_name TEXT,
                        vehicle_type TEXT,
                        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                
                # Create detection_history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detection_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT NOT NULL,
                        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_registered BOOLEAN NOT NULL,
                        confidence REAL
                    )
                ''')
                
                conn.commit()
                logging.info("Database initialized successfully")
                
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise
    
    def add_vehicle(self, plate_number: str, owner_name: str = "", vehicle_type: str = "") -> bool:
        """Add a new vehicle to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO vehicles (plate_number, owner_name, vehicle_type)
                    VALUES (?, ?, ?)
                ''', (plate_number.upper(), owner_name, vehicle_type))
                conn.commit()
                logging.info(f"Added vehicle: {plate_number}")
                return True
        except sqlite3.IntegrityError:
            logging.warning(f"Vehicle {plate_number} already exists")
            return False
        except Exception as e:
            logging.error(f"Error adding vehicle {plate_number}: {e}")
            return False
    
    def remove_vehicle(self, plate_number: str) -> bool:
        """Remove a vehicle from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM vehicles WHERE plate_number = ?', (plate_number.upper(),))
                if cursor.rowcount > 0:
                    conn.commit()
                    logging.info(f"Removed vehicle: {plate_number}")
                    return True
                else:
                    logging.warning(f"Vehicle {plate_number} not found")
                    return False
        except Exception as e:
            logging.error(f"Error removing vehicle {plate_number}: {e}")
            return False
    
    def is_vehicle_registered(self, plate_number: str) -> bool:
        """Check if a vehicle is registered in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 1 FROM vehicles 
                    WHERE plate_number = ? AND status = 'active'
                ''', (plate_number.upper(),))
                return cursor.fetchone() is not None
        except Exception as e:
            logging.error(f"Error checking vehicle registration {plate_number}: {e}")
            return False
    
    def get_all_vehicles(self) -> List[Tuple]:
        """Get all registered vehicles."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT plate_number, owner_name, vehicle_type, registration_date, status
                    FROM vehicles ORDER BY registration_date DESC
                ''')
                return cursor.fetchall()
        except Exception as e:
            logging.error(f"Error fetching vehicles: {e}")
            return []
    
    def log_detection(self, plate_number: str, is_registered: bool, confidence: float = 0.0):
        """Log a detection event to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO detection_history (plate_number, is_registered, confidence)
                    VALUES (?, ?, ?)
                ''', (plate_number.upper(), is_registered, confidence))
                conn.commit()
        except Exception as e:
            logging.error(f"Error logging detection: {e}")
    
    def get_detection_history(self, limit: int = 100) -> pd.DataFrame:
        """Get recent detection history as DataFrame."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT plate_number, detection_time, is_registered, confidence
                    FROM detection_history 
                    ORDER BY detection_time DESC 
                    LIMIT ?
                '''
                return pd.read_sql_query(query, conn, params=[limit])
        except Exception as e:
            logging.error(f"Error fetching detection history: {e}")
            return pd.DataFrame()
