"""Camera handler for video capture and processing."""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable
import logging

class CameraHandler:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """Initialize camera handler."""
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_time = 0
        
    def start_camera(self) -> bool:
        """Start the camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logging.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            logging.info(f"Camera {self.camera_index} started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera capture."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.info("Camera stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame from camera."""
        if not self.is_running or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.last_frame_time = time.time()
                return frame
            else:
                logging.warning("Failed to read frame from camera")
                return None
                
        except Exception as e:
            logging.error(f"Error getting frame: {e}")
            return None
    
    def get_cached_frame(self) -> Optional[np.ndarray]:
        """Get the last cached frame."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def is_frame_fresh(self, max_age: float = 1.0) -> bool:
        """Check if the current frame is fresh (within max_age seconds)."""
        return (time.time() - self.last_frame_time) < max_age
    
    def get_available_cameras(self) -> list:
        """Get list of available camera indices."""
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def change_camera(self, new_index: int) -> bool:
        """Change to a different camera."""
        if self.is_running:
            self.stop_camera()
        
        self.camera_index = new_index
        return self.start_camera()
    
    def get_camera_info(self) -> dict:
        """Get camera information."""
        if not self.cap:
            return {}
        
        try:
            info = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'index': self.camera_index,
                'is_running': self.is_running
            }
            return info
        except:
            return {}
