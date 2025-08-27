"""ANPR detector using OpenCV and EasyOCR."""

import cv2
import easyocr
import numpy as np
import re
from typing import List, Tuple, Optional
import logging

class ANPRDetector:
    def __init__(self, languages=['en']):
        """Initialize ANPR detector with EasyOCR."""
        try:
            self.reader = easyocr.Reader(languages)
            logging.info("EasyOCR initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing EasyOCR: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def detect_license_plates(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Detect license plates in the image.
        Returns list of (text, confidence, bbox) tuples.
        """
        try:
            # Use EasyOCR to detect text
            results = self.reader.readtext(image)
            
            plates = []
            for (bbox, text, confidence) in results:
                # Filter potential license plates based on text pattern and confidence
                if self.is_likely_license_plate(text) and confidence > 0.5:
                    # Convert bbox to (x, y, w, h) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
                    
                    # Clean up the text
                    cleaned_text = self.clean_plate_text(text)
                    if cleaned_text:
                        plates.append((cleaned_text, confidence, (x, y, w, h)))
            
            return plates
            
        except Exception as e:
            logging.error(f"Error detecting license plates: {e}")
            return []
    
    def is_likely_license_plate(self, text: str) -> bool:
        """Check if detected text is likely a license plate."""
        # Remove spaces and convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Basic license plate patterns (can be customized for different regions)
        patterns = [
            r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$',  # ABC123, AB1234C, etc.
            r'^[0-9]{1,3}[A-Z]{1,3}[0-9]{1,4}$',  # 123ABC456, etc.
            r'^[A-Z0-9]{4,8}$',  # General alphanumeric
        ]
        
        # Check length and character composition
        if len(cleaned) < 4 or len(cleaned) > 10:
            return False
        
        # Must contain both letters and numbers
        has_letter = any(c.isalpha() for c in cleaned)
        has_number = any(c.isdigit() for c in cleaned)
        
        if not (has_letter and has_number):
            return False
        
        # Check against patterns
        for pattern in patterns:
            if re.match(pattern, cleaned):
                return True
        
        return False
    
    def clean_plate_text(self, text: str) -> str:
        """Clean and format detected plate text."""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Additional cleaning rules can be added here
        # For example, common OCR mistakes: 0->O, 1->I, etc.
        replacements = {
            'O': '0',  # Sometimes O is confused with 0
            'I': '1',  # Sometimes I is confused with 1
        }
        
        # Apply replacements contextually
        for old, new in replacements.items():
            # Only replace if it makes sense in context
            if cleaned.count(old) < len(cleaned) // 2:  # Don't replace if too many
                cleaned = cleaned.replace(old, new)
        
        return cleaned if len(cleaned) >= 4 else ""
    
    def draw_detections(self, image: np.ndarray, detections: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        """Draw bounding boxes and text on the image."""
        result_image = image.copy()
        
        for text, confidence, (x, y, w, h) in detections:
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_image, (x, y - text_size[1] - 10), 
                         (x + text_size[0], y), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(result_image, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw confidence
            conf_text = f"{confidence:.2f}"
            cv2.putText(result_image, conf_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
