"""
ANPR (Automatic Number Plate Recognition) System
A Streamlit-based web application for real-time license plate detection and monitoring.
"""

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
from PIL import Image

# Import custom modules
from anpr_detector import ANPRDetector
from database_manager import DatabaseManager
from camera_handler import CameraHandler
from logger import ANPRLogger
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state
if 'camera_handler' not in st.session_state:
    st.session_state.camera_handler = None
if 'anpr_detector' not in st.session_state:
    st.session_state.anpr_detector = None
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'anpr_logger' not in st.session_state:
    st.session_state.anpr_logger = None
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False
if 'last_detection' not in st.session_state:
    st.session_state.last_detection = None
if 'detection_status' not in st.session_state:
    st.session_state.detection_status = "Unknown"

def initialize_components():
    """Initialize all system components."""
    try:
        if st.session_state.anpr_detector is None:
            with st.spinner("Initializing ANPR detector..."):
                st.session_state.anpr_detector = ANPRDetector(config.OCR_LANGUAGES)
        
        if st.session_state.db_manager is None:
            st.session_state.db_manager = DatabaseManager(config.DATABASE_PATH)
        
        if st.session_state.anpr_logger is None:
            st.session_state.anpr_logger = ANPRLogger(config.LOG_FILE_PATH)
        
        return True
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        logging.error(f"Error initializing components: {e}")
        return False

def start_camera(camera_index):
    """Start camera with given index."""
    try:
        if st.session_state.camera_handler:
            st.session_state.camera_handler.stop_camera()
        
        st.session_state.camera_handler = CameraHandler(
            camera_index, config.FRAME_WIDTH, config.FRAME_HEIGHT
        )
        
        if st.session_state.camera_handler.start_camera():
            st.success(f"Camera {camera_index} started successfully!")
            return True
        else:
            st.error(f"Failed to start camera {camera_index}")
            return False
    except Exception as e:
        st.error(f"Error starting camera: {e}")
        logging.error(f"Error starting camera: {e}")
        return False

def process_frame(frame):
    """Process a single frame for ANPR detection."""
    try:
        detections = st.session_state.anpr_detector.detect_license_plates(frame)
        
        if detections:
            # Get the best detection (highest confidence)
            best_detection = max(detections, key=lambda x: x[1])
            plate_text, confidence, bbox = best_detection
            
            # Check if vehicle is registered
            is_registered = st.session_state.db_manager.is_vehicle_registered(plate_text)
            
            # Log the detection
            st.session_state.db_manager.log_detection(plate_text, is_registered, confidence)
            st.session_state.anpr_logger.log_detection(plate_text, is_registered, confidence)
            
            # Update session state
            st.session_state.last_detection = {
                'plate': plate_text,
                'confidence': confidence,
                'is_registered': is_registered,
                'timestamp': datetime.now(),
                'bbox': bbox
            }
            
            st.session_state.detection_status = "Registered" if is_registered else "Unregistered"
        
        # Draw detections on frame
        annotated_frame = st.session_state.anpr_detector.draw_detections(frame, detections)
        return annotated_frame, detections
        
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return frame, []

def display_status_indicator():
    """Display red/green status indicator."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.session_state.last_detection:
            detection = st.session_state.last_detection
            time_diff = datetime.now() - detection['timestamp']
            
            if time_diff.total_seconds() < 5:  # Show status for 5 seconds
                if detection['is_registered']:
                    st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #ff4444; 
                                    border-radius: 10px; margin: 10px 0;'>
                            <h2 style='color: white; margin: 0;'>🔴 REGISTERED VEHICLE</h2>
                            <p style='color: white; margin: 5px 0;'>{}</p>
                            <p style='color: white; margin: 0; font-size: 14px;'>Confidence: {:.2f}</p>
                        </div>
                    """.format(detection['plate'], detection['confidence']), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #44ff44; 
                                    border-radius: 10px; margin: 10px 0;'>
                            <h2 style='color: white; margin: 0;'>🟢 UNREGISTERED VEHICLE</h2>
                            <p style='color: white; margin: 5px 0;'>{}</p>
                            <p style='color: white; margin: 0; font-size: 14px;'>Confidence: {:.2f}</p>
                        </div>
                    """.format(detection['plate'], detection['confidence']), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='text-align: center; padding: 20px; background-color: #888888; 
                                border-radius: 10px; margin: 10px 0;'>
                        <h2 style='color: white; margin: 0;'>⚪ READY FOR DETECTION</h2>
                        <p style='color: white; margin: 0;'>Monitoring for license plates...</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='text-align: center; padding: 20px; background-color: #888888; 
                            border-radius: 10px; margin: 10px 0;'>
                    <h2 style='color: white; margin: 0;'>⚪ READY FOR DETECTION</h2>
                    <p style='color: white; margin: 0;'>Monitoring for license plates...</p>
                </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function."""
    st.set_page_config(
        page_title="ANPR System",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚗 ANPR (Automatic Number Plate Recognition) System")
    st.markdown("Real-time license plate detection and monitoring system")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Camera selection
        st.subheader("📹 Camera Settings")
        available_cameras = list(range(3))  # Check first 3 camera indices
        selected_camera = st.selectbox("Select Camera", available_cameras, index=0)
        
        if st.button("Start Camera"):
            start_camera(selected_camera)
        
        if st.button("Stop Camera"):
            if st.session_state.camera_handler:
                st.session_state.camera_handler.stop_camera()
                st.success("Camera stopped")
        
        # Detection controls
        st.subheader("🔍 Detection Controls")
        detection_enabled = st.checkbox("Enable Detection", value=True)
        
        # Display camera info
        if st.session_state.camera_handler:
            info = st.session_state.camera_handler.get_camera_info()
            if info:
                st.subheader("📊 Camera Info")
                st.write(f"Resolution: {info.get('width', 'N/A')}x{info.get('height', 'N/A')}")
                st.write(f"FPS: {info.get('fps', 'N/A')}")
                st.write(f"Status: {'Running' if info.get('is_running', False) else 'Stopped'}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📺 Live Camera Feed")
        video_placeholder = st.empty()
        
        # Display status indicator
        display_status_indicator()
    
    with col2:
        st.subheader("📋 Recent Detections")
        detection_placeholder = st.empty()
        
        st.subheader("📈 Statistics")
        stats_placeholder = st.empty()
    
    # Vehicle management section
    st.header("🚙 Vehicle Database Management")
    
    tab1, tab2, tab3 = st.tabs(["Add Vehicle", "Registered Vehicles", "Detection History"])
    
    with tab1:
        st.subheader("Add New Vehicle")
        with st.form("add_vehicle_form"):
            plate_number = st.text_input("License Plate Number").upper()
            owner_name = st.text_input("Owner Name (Optional)")
            vehicle_type = st.selectbox("Vehicle Type", ["Car", "Motorcycle", "Truck", "Van", "Other"])
            
            if st.form_submit_button("Add Vehicle"):
                if plate_number:
                    if st.session_state.db_manager.add_vehicle(plate_number, owner_name, vehicle_type):
                        st.success(f"Vehicle {plate_number} added successfully!")
                    else:
                        st.error(f"Vehicle {plate_number} already exists or error occurred")
                else:
                    st.error("Please enter a license plate number")
    
    with tab2:
        st.subheader("Registered Vehicles")
        vehicles = st.session_state.db_manager.get_all_vehicles()
        
        if vehicles:
            df = pd.DataFrame(vehicles, columns=['Plate Number', 'Owner', 'Type', 'Registration Date', 'Status']) if vehicles else pd.DataFrame([])
            
            # Display vehicles with delete option
            for idx, vehicle in enumerate(vehicles):
                col_info, col_action = st.columns([3, 1])
                with col_info:
                    st.write(f"**{vehicle[0]}** - {vehicle[1]} ({vehicle[2]})")
                with col_action:
                    if st.button(f"Remove", key=f"remove_{idx}"):
                        if st.session_state.db_manager.remove_vehicle(vehicle[0]):
                            st.success(f"Vehicle {vehicle[0]} removed!")
                            st.rerun()
                        else:
                            st.error("Error removing vehicle")
        else:
            st.info("No vehicles registered yet")
    
    with tab3:
        st.subheader("Detection History")
        history = st.session_state.anpr_logger.get_recent_logs(50)
        
        if not history.empty:
            st.dataframe(history, use_container_width=True)
            
            # Download button for logs
            csv = history.to_csv(index=False)
            st.download_button(
                label="Download Detection History",
                data=csv,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No detection history available")
    
    # Auto-refresh for live video and detection
    if st.session_state.camera_handler and st.session_state.camera_handler.is_running:
        while True:
            frame = st.session_state.camera_handler.get_frame()
            
            if frame is not None:
                if detection_enabled:
                    processed_frame, detections = process_frame(frame)
                else:
                    processed_frame = frame
                    detections = []
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                with video_placeholder.container():
                    st.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Update recent detections
                with detection_placeholder.container():
                    recent_logs = st.session_state.anpr_logger.get_recent_logs(10)
                    if not recent_logs.empty:
                        st.dataframe(recent_logs[['Timestamp', 'Number_Plate', 'Status', 'Confidence']], 
                                   use_container_width=True, height=300)
                
                # Update statistics
                with stats_placeholder.container():
                    stats = st.session_state.anpr_logger.get_statistics()
                    if stats:
                        st.metric("Total Detections", stats.get('total_detections', 0))
                        st.metric("Registered", stats.get('registered_count', 0))
                        st.metric("Unregistered", stats.get('unregistered_count', 0))
                        st.metric("Avg Confidence", f"{stats.get('average_confidence', 0):.2f}")
            
            time.sleep(config.REFRESH_INTERVAL)
            
            # Break if user navigates away or camera stops
            if not st.session_state.camera_handler.is_running:
                break

if __name__ == "__main__":
    main()
