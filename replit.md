# ANPR (Automatic Number Plate Recognition) System

## Overview

This is a web-based Automatic Number Plate Recognition (ANPR) system built with Python and Streamlit. The application provides real-time license plate detection and monitoring capabilities using computer vision and OCR technologies. The system captures video input from cameras, processes frames to detect license plates, verifies them against a registered vehicle database, and provides visual feedback through a web dashboard with red/green light indicators.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Framework**: Chosen for rapid prototyping and built-in UI components that simplify dashboard creation
- **Real-time Dashboard**: Displays live camera feed, detection results, and status indicators
- **Session State Management**: Maintains component states across user interactions
- **Interactive Controls**: Provides camera configuration and database management interfaces

### Backend Architecture
- **Modular Component Design**: Separated into distinct modules for maintainability
  - `ANPRDetector`: Handles license plate detection using EasyOCR
  - `CameraHandler`: Manages video capture with threading for concurrent processing
  - `DatabaseManager`: Handles SQLite operations for vehicle registration
  - `ANPRLogger`: Manages detection event logging to CSV and JSON formats

### Computer Vision Pipeline
- **OpenCV Integration**: Used for image preprocessing including grayscale conversion, Gaussian blur, and adaptive thresholding
- **EasyOCR Engine**: Selected over Tesseract for better accuracy with license plates and easier configuration
- **Multi-threaded Processing**: Camera capture runs in separate thread to prevent UI blocking

### Data Storage Solutions
- **SQLite Database**: Lightweight, file-based database chosen for simplicity and no external server requirements
  - `vehicles` table: Stores registered vehicle information
  - `detection_history` table: Tracks all detection events with timestamps
- **File-based Logging**: Dual format logging (CSV/JSON) for different analysis needs

### Authentication and Authorization
- No authentication system implemented - designed as a standalone application
- Database access is direct through the application layer

### Configuration Management
- **Centralized Config**: All system parameters defined in `config.py`
- **Configurable Camera Settings**: Supports different camera indices and resolutions
- **Adjustable Detection Parameters**: Confidence thresholds and plate size filters

## External Dependencies

### Computer Vision Libraries
- **OpenCV (cv2)**: Image processing and camera capture functionality
- **EasyOCR**: Optical character recognition engine for license plate text extraction
- **NumPy**: Array operations for image data manipulation
- **PIL (Pillow)**: Image format handling and conversions

### Web Framework
- **Streamlit**: Web application framework providing the dashboard interface
- **Pandas**: Data manipulation for displaying detection logs and vehicle records

### Database
- **SQLite3**: Built-in Python database engine for local data storage
- No external database server required

### Utility Libraries
- **Threading**: Concurrent camera processing
- **Logging**: Application event tracking
- **CSV/JSON**: Detection event logging formats
- **DateTime**: Timestamp management for detection events

### Development Dependencies
- **Re (Regular Expressions)**: License plate pattern matching
- **Time**: Performance timing and delays
- **OS**: File system operations