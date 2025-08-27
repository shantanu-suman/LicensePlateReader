"""Configuration settings for the ANPR system."""

import os

# Camera settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 10

# Detection settings
MIN_CONFIDENCE = 0.5
PLATE_MIN_WIDTH = 50
PLATE_MIN_HEIGHT = 20

# Database settings
DATABASE_PATH = "vehicles.db"

# Logging settings
LOG_FILE_PATH = "detection_logs.csv"

# EasyOCR settings
OCR_LANGUAGES = ['en']
OCR_DETAIL = 0  # 0 for simple output, 1 for detailed

# UI settings
REFRESH_INTERVAL = 0.1  # seconds
