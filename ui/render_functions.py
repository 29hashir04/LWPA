"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OPTIMIZED UI RENDERING FUNCTIONS                          ║
║                                                                              ║
║  Production-grade Streamlit UI with:                                         ║
║  - Optimized camera pipeline (30+ FPS)                                       ║
║  - Smooth overlay transitions                                                ║
║  - Real-time performance metrics                                             ║
║  - Robust error handling                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import cv2
import numpy as np
import time
import torch
import logging
import gc
from typing import Dict, Any, Optional, Tuple
from collections import deque, Counter
from dataclasses import dataclass
from enum import Enum
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import threading

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# UI CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class UIConfig:
    """Centralized UI configuration"""
    # Camera settings
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    TARGET_FPS: int = 30
    CAMERA_BUFFER_SIZE: int = 1  # Minimize latency
    
    # Recognition display
    MIN_CONFIDENCE_DISPLAY: float = 0.40
    MIN_VOTES_DISPLAY: int = 4
    
    # Enrollment defaults
    DEFAULT_SEQUENCES: int = 8
    DEFAULT_FRAMES_PER_SEQ: int = 45
    DEFAULT_TIMEOUT_S: int = 20
    
    # Performance
    ENROLLED_CACHE_REFRESH_S: float = 5.0
    FRAME_SKIP_THRESHOLD_MS: float = 100.0  # Skip frames if lagging


UI_CONFIG = UIConfig()


class OverlayState(Enum):
    """Overlay display states"""
    IDLE = "idle"
    NO_PERSON = "no_person"
    PARTIAL = "partial"
    PROCESSING = "processing"
    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"


# =============================================================================
# CACHED MODEL LOADER
# =============================================================================

@st.cache_resource(show_spinner=False)
def _get_cached_gait_model():
    """
    Cache the heavy YOLO pose model across browser sessions.
    This is critical for performance - model loads once, reuses forever.
    """
    from models.gait_model import OptimizedFastPoseGaitModel
    
    with st.spinner("Loading pose detection models..."):
        model = OptimizedFastPoseGaitModel()
        
        # Validate model loaded correctly
        if getattr(model, "pose_model", None) is None:
            raise RuntimeError("Failed to initialize YOLO pose model")
        
        # Log health check
        health = model.health_check()
        logger.info(f"Model health: {health}")
        
        return model


# =============================================================================
# STREAMLIT COMPATIBILITY LAYER
# =============================================================================

def _image(container, img, **kwargs):
    """
    Version-safe st.image wrapper.
    Handles use_container_width vs use_column_width API changes.
    """
    try:
        return container.image(img, use_container_width=True, **kwargs)
    except TypeError:
        return container.image(img, use_column_width=True, **kwargs)


# =============================================================================
# MODEL & DATA INITIALIZATION
# =============================================================================

def initialize_optimized_models():
    """Initialize models with caching and validation"""
    if st.session_state.get('gait_model') is None:
        from data.gait_data_manager import GaitDataManager
        
        try:
            st.session_state.gait_model = _get_cached_gait_model()
            st.session_state.data_manager = GaitDataManager()
            
            # Log GPU status
            if torch.cuda.is_available():
                logger.info(f"GPU Active: {torch.cuda.get_device_name()}")
            else:
                logger.warning("Running on CPU - performance may be limited")
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}", exc_info=True)
            st.session_state.gait_model = None


def get_current_user() -> Optional[Dict]:
    """Get current Firebase user with error handling"""
    try:
        from firebase_setup.firebase_auth_service import get_current_firebase_user
        return get_current_firebase_user()
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None


# =============================================================================
# PERFORMANCE DASHBOARD
# =============================================================================

def render_performance_dashboard():
    """Real-time performance metrics display"""
    st.markdown("### Performance")
    
    gait_model = st.session_state.get('gait_model')
    
    if gait_model:
        metrics = gait_model.get_performance_metrics()
        
        # Main metrics row
        col1, col2 = st.columns(2)
        
        with col1:
            fps = metrics.get('fps', 0)
            fps_indicator = "[OK]" if fps >= 25 else "[SLOW]" if fps >= 15 else "[LOW]"
            st.metric("FPS", f"{fps_indicator} {fps:.1f}")
        
        with col2:
            device = "GPU" if metrics.get('gpu_available') else "CPU"
            st.metric("Device", device)
        
        # Buffer progress
        buffer_fill = metrics.get('buffer_fill', 0)
        st.progress(buffer_fill, text=f"Buffer: {buffer_fill:.0%}")
        
        # GPU info
        if torch.cuda.is_available():
            st.caption(f"GPU: {torch.cuda.get_device_name()}")
            
            # VRAM usage
            try:
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                st.caption(f"VRAM: {allocated:.1f}GB / {reserved:.1f}GB reserved")
            except Exception:
                pass


# =============================================================================
# RECOGNITION SETTINGS
# =============================================================================

def render_recognition_settings():
    """
    Real-time recognition threshold slider.
    
    Allows users to adjust matching sensitivity:
    - Lower threshold = stricter matching (fewer false positives)
    - Higher threshold = lenient matching (fewer false negatives)
    """
    st.markdown("### Recognition Settings")
    
    gait_model = st.session_state.get('gait_model')
    
    if not gait_model:
        st.warning("Model not loaded")
        return
    
    # Get current threshold from model
    current_threshold = getattr(gait_model, 'similarity_threshold', 0.70)
    
    # Threshold slider
    new_threshold = st.slider(
        "Matching Sensitivity",
        min_value=0.70,
        max_value=0.95,
        value=float(current_threshold),
        step=0.01,
        help="Higher = stricter (high security), Lower = lenient (more forgiving)"
    )
    
    # Apply threshold to model
    if new_threshold != current_threshold:
        gait_model.similarity_threshold = new_threshold
        st.session_state.gait_model = gait_model
    
    # Visual indicator
    if new_threshold >= 0.90:
        st.caption("Mode: **Strict** - High security, may reject valid users")
    elif new_threshold >= 0.80:
        st.caption("Mode: **Balanced** - Good accuracy and usability")
    else:
        st.caption("Mode: **Lenient** - More forgiving, may accept similar gaits")


# =============================================================================
# STATUS PANEL
# =============================================================================

def render_status_panel():
    """Recognition status and enrolled patterns display"""
    st.markdown("### Status")
    
    user_data = get_current_user()
    user_id = user_data.get('localId', '') if user_data else ''
    
    data_manager = st.session_state.get('data_manager')
    
    if data_manager and user_id:
        enrolled = data_manager.load_user_gait_data(user_id)
        
        if len(enrolled) == 0:
            st.warning("No patterns enrolled!")
            st.info("Go to **Enroll** tab to add patterns")
        else:
            st.success(f"Monitoring {len(enrolled)} pattern(s)")
            
            # List enrolled names
            for name in enrolled.keys():
                st.markdown(f"- **{name}**")


# =============================================================================
# MAIN APPLICATION UI
# =============================================================================

def render_optimized_main_application():
    """Main recognition application with optimized UI"""
    user_data = get_current_user()
    user_name = user_data.get('displayName', 'User') if user_data else 'User'
    
    # Header row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("## Live Walking Pattern Analysis")
    
    with col3:
            from firebase_setup.firebase_auth_service import render_firebase_logout_button
            render_firebase_logout_button()
    
    # GPU status banner
    if torch.cuda.is_available():
        st.success(f"GPU Accelerated: {torch.cuda.get_device_name()}")
    else:
        st.warning("Running on CPU - Consider using a GPU for better performance")
    
    st.markdown(f"Welcome, **{user_name}**!")
    
    # Initialize models
    initialize_optimized_models()
    
    if st.session_state.get('gait_model') is None:
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # Main layout
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.markdown("### Camera Feed")
        render_camera_section()
        
    with col_side:
        render_performance_dashboard()
        st.markdown("---")
        render_recognition_settings()
        st.markdown("---")
        render_status_panel()


# =============================================================================
# CAMERA SECTION
# =============================================================================

def render_camera_section():
    """Camera controls and feed with WebRTC"""
    
    # Get user and enrolled data
    user_data = get_current_user()
    user_id = user_data.get('localId', '') if user_data else ''
    
    gait_model = st.session_state.get('gait_model')
    data_manager = st.session_state.get('data_manager')
    
    # Initialize enrolled cache
    if "enrolled_cache" not in st.session_state:
        st.session_state.enrolled_cache = {}
        st.session_state.enrolled_cache_user = ""
        st.session_state.enrolled_cache_ts = 0.0
    
    # Refresh cache if stale
    now = time.time()
    if (st.session_state.enrolled_cache_user != user_id or 
        now - st.session_state.enrolled_cache_ts > UI_CONFIG.ENROLLED_CACHE_REFRESH_S):
        
        st.session_state.enrolled_cache = data_manager.load_user_gait_data(user_id)
        st.session_state.enrolled_cache_user = user_id
        st.session_state.enrolled_cache_ts = now
        
        enrolled_names = list(st.session_state.enrolled_cache.keys())
        logger.info(f"Loaded {len(enrolled_names)} enrolled patterns: {enrolled_names}")
    
    enrolled_data = st.session_state.enrolled_cache
    
    # Thread-safe frame processor class
    class FrameProcessor:
        def __init__(self):
            self.overlay_state = OverlayState.IDLE
            self.current_name = ""
            self.current_confidence = 0.0
            self.lock_frames = 0
            self.frame_count = 0
            self.start_time = time.time()
            
        def process(self, frame: av.VideoFrame):
            """Process each video frame from WebRTC"""
            # Convert to numpy array (BGR format from av)
            img = frame.to_ndarray(format="bgr24")
            
            # Mirror for natural feel
            img = cv2.flip(img, 1)
            
            # Update stats
            self.frame_count += 1
            
            # Process frame
            result = process_frame_optimized(img, enrolled_data, gait_model)
            
            # Update overlay state
            new_state, new_name, new_conf = determine_overlay_state(result)
            
            # Lock state for a few frames to prevent flicker
            if self.lock_frames > 0:
                self.lock_frames -= 1
            else:
                if new_state != self.overlay_state or new_name != self.current_name:
                    self.overlay_state = new_state
                    self.current_name = new_name
                    self.current_confidence = new_conf
                    self.lock_frames = 3
                else:
                    # Smooth confidence update
                    self.current_confidence = 0.8 * self.current_confidence + 0.2 * new_conf
            
            # Draw overlay
            frame_with_overlay = draw_optimized_overlay(
                img,
                self.overlay_state,
                self.current_name,
                self.current_confidence,
                result['status']
            )
            
            # Convert back to VideoFrame
            return av.VideoFrame.from_ndarray(frame_with_overlay, format="bgr24")
    
    # Initialize processor
    if 'frame_processor' not in st.session_state:
        st.session_state.frame_processor = FrameProcessor()
    
    processor = st.session_state.frame_processor
    
    # WebRTC configuration with STUN and TURN servers for ngrok/external access
    # TURN servers provide relay fallback when direct peer connection fails
    rtc_configuration = RTCConfiguration(
        {"iceServers": [
            # Google STUN servers for NAT traversal
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            # Free public TURN servers for relay (when STUN fails)
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
        ],
        "iceCandidatePoolSize": 10,  # Generate more ICE candidates
        }
    )
    
    # Display WebRTC streamer with error handling
    try:
        webrtc_ctx = webrtc_streamer(
            key="gait-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=processor.process,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        
        # Status display
        if webrtc_ctx.state.playing:
            elapsed = time.time() - processor.start_time
            fps = processor.frame_count / elapsed if elapsed > 0 else 0
            
            status_text = f"FPS: {fps:.1f}"
            
            if processor.overlay_state == OverlayState.RECOGNIZED:
                status_text += f" | ✅ {processor.current_name} ({processor.current_confidence:.0%})"
                st.success(status_text)
            elif processor.overlay_state == OverlayState.UNKNOWN:
                status_text += " | ⚠️ Unknown Person"
                st.warning(status_text)
            else:
                st.info(status_text)
        else:
            st.info("Click 'START' to begin camera recognition")
    except (KeyError, AttributeError, RuntimeError) as e:
        # Handle streamlit-webrtc errors (Python 3.13 compatibility issues)
        error_msg = str(e)
        if any(keyword in error_msg for keyword in ["frontend", "is_alive", "event loop", "NoneType"]):
            st.warning("⚠️ Camera initialization issue (Python 3.13 compatibility). Please use Python 3.11 for best results.")
            st.info("The app is running but WebRTC may have errors. Consider downgrading to Python 3.11.")
        else:
            raise


# =============================================================================
# LEGACY CAMERA LOOP (REPLACED BY WEBRTC)
# =============================================================================
# The old cv2.VideoCapture loop has been replaced with WebRTC streaming
# for web deployment compatibility. See video_frame_callback in render_camera_section()


# =============================================================================
# FRAME PROCESSING
# =============================================================================

def process_frame_optimized(
    frame: np.ndarray, 
    enrolled_data: Dict[str, np.ndarray],
    gait_model
) -> Dict[str, Any]:
    """
    Process single frame through recognition pipeline.
    
    Returns:
        Dict with status, name, confidence, is_stable
    """
    result = {
        'status': 'no_person',
        'name': 'Unknown',
        'confidence': 0.0,
        'is_stable': False
    }
    
    try:
        # Pose detection
        batch_results = gait_model.detect_person_status_batch([frame])
        
        if not batch_results:
            return result
        
        status, keypoints = batch_results[0]
        result['status'] = status
        
        # Only process full body detections
        if status != 'full_body' or keypoints is None:
            return result
        
        # No enrolled data = always Unknown
        if not enrolled_data:
            result['name'] = 'Unknown'
            result['confidence'] = 0.0
            result['is_stable'] = True
            return result
        
        # Feature extraction
        features = gait_model.extract_gait_features_optimized(keypoints)
                
        if features is None:
            result['name'] = 'Processing...'
            return result
        
        # Recognition
        name, conf = gait_model.recognize_person_optimized(features, enrolled_data)
        
        # Get stable decision from voting buffer
        stable_name, stable_conf, is_stable = gait_model.get_stable_decision()
        
        result['name'] = stable_name
        result['confidence'] = stable_conf
        result['is_stable'] = is_stable
                        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")

    return result


def determine_overlay_state(result: Dict) -> Tuple[OverlayState, str, float]:
    """
    Determine overlay state from processing result.
    
    Returns:
        (state, name, confidence)
    """
    status = result.get('status', 'no_person')
    name = result.get('name', 'Unknown')
    conf = result.get('confidence', 0.0)
    is_stable = result.get('is_stable', False)
    
    if status == 'no_person':
        return OverlayState.NO_PERSON, "", 0.0
    
    if status == 'partial_body':
        return OverlayState.PARTIAL, "", 0.0
    
    if name == 'Processing...':
        return OverlayState.PROCESSING, "", 0.0
    
    if name == 'Unknown':
        return OverlayState.UNKNOWN, "Unknown", 0.0
    
    if is_stable and conf >= UI_CONFIG.MIN_CONFIDENCE_DISPLAY:
        return OverlayState.RECOGNIZED, name, conf
    
    return OverlayState.PROCESSING, name, conf


# =============================================================================
# OVERLAY RENDERING
# =============================================================================

def draw_optimized_overlay(
    frame: np.ndarray,
    state: OverlayState,
    name: str,
    confidence: float,
    detection_status: str
) -> np.ndarray:
    """
    Draw production-quality overlay with smooth transitions.
    
    Visual design:
    - Top bar: Detection status
    - Bottom bar: Recognition result (color-coded)
    - Confidence bar (when recognized)
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # =========================================================================
    # TOP BAR - Detection Status
    # =========================================================================
    
    status_colors = {
        'no_person': (80, 80, 80),      # Gray
        'partial_body': (0, 140, 255),   # Orange
        'full_body': (0, 200, 0)         # Green
    }
    status_color = status_colors.get(detection_status, (80, 80, 80))
    
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Status indicator circle
    cv2.circle(frame, (25, 22), 10, status_color, -1)
    
    # Status text
    status_text = detection_status.replace('_', ' ').title()
    cv2.putText(frame, status_text, (45, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # =========================================================================
    # BOTTOM BAR - Recognition Result
    # =========================================================================
    
    bar_height = 60
    
    if state == OverlayState.NO_PERSON:
        # Gray bar - no person
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (60, 60, 60), -1)
        cv2.putText(frame, "No person detected", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    
    elif state == OverlayState.PARTIAL:
        # Orange bar - partial detection
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 100, 180), -1)
        cv2.putText(frame, "Stand further back for full body detection", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    elif state == OverlayState.PROCESSING:
        # Yellow bar - processing
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 150, 180), -1)
        cv2.putText(frame, "Analyzing gait pattern...", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Animated dots (based on frame count would need state, simplified here)
        cv2.putText(frame, "Keep walking naturally", (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    elif state == OverlayState.UNKNOWN:
        # Red bar - unknown person
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 160), -1)
        
        # Warning icon (triangle)
        pts = np.array([[w//2 - 20, h - 45], [w//2, h - 55], [w//2 + 20, h - 45]], np.int32)
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        cv2.putText(frame, "!", (w//2 - 5, h - 47), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 160), 2)
        
        cv2.putText(frame, "UNKNOWN PERSON", (w//2 - 100, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    elif state == OverlayState.RECOGNIZED:
        # Green bar - recognized
        cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 130, 0), -1)
        
        # Checkmark
        cv2.putText(frame, "[OK]", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Name and confidence
        display_text = f"{name} ({confidence:.0%})"
        cv2.putText(frame, display_text, (80, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Confidence bar
        bar_width = int((w - 40) * confidence)
        cv2.rectangle(frame, (20, h - 55), (20 + bar_width, h - 45), (100, 255, 100), -1)
        cv2.rectangle(frame, (20, h - 55), (w - 20, h - 45), (255, 255, 255), 1)
    
    return frame


# =============================================================================
# ENROLLMENT PAGE
# =============================================================================

def render_optimized_enrollment_page():
    """Enrollment page with optimized auto-capture"""
    st.markdown("### Enroll New Gait Pattern")
    
    user_data = get_current_user()
    user_id = user_data.get('localId', '') if user_data else ''
    
    initialize_optimized_models()
    
    # Validation
    if not user_id:
        st.error("You must be logged in to enroll gait data.")
        return
    
    if not st.session_state.get("gait_model") or not st.session_state.get("data_manager"):
        st.error("Models not initialized. Please refresh the page.")
        return
        
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        person_name = st.text_input("Person Name", placeholder="Enter name to enroll")
        
        with st.expander("Advanced Options", expanded=False):
            target_sequences = st.slider(
                "Number of sequences", 
                min_value=5, max_value=15, 
                value=UI_CONFIG.DEFAULT_SEQUENCES,
                help="More sequences = more accurate recognition"
            )
            frames_per_sequence = st.slider(
                "Frames per sequence", 
                min_value=30, max_value=90, 
                value=UI_CONFIG.DEFAULT_FRAMES_PER_SEQ,
                help="45-60 frames covers ~1.5 gait cycles"
            )
            sequence_timeout = st.slider(
                "Sequence timeout (seconds)", 
                min_value=10, max_value=60, 
                value=UI_CONFIG.DEFAULT_TIMEOUT_S
            )
    
    with col2:
        data_manager = st.session_state.get('data_manager')
        if data_manager:
            count = data_manager.get_user_gait_count(user_id)
            st.metric("Enrolled Patterns", count)
    
    # =========================================================================
    # ENROLLMENT SESSION STATE
    # =========================================================================
    
    if "enroll_session" not in st.session_state:
        st.session_state.enroll_session = {
            "active": False,
            "person_name": "",
            "target_sequences": UI_CONFIG.DEFAULT_SEQUENCES,
            "frames_per_sequence": UI_CONFIG.DEFAULT_FRAMES_PER_SEQ,
            "timeout_s": UI_CONFIG.DEFAULT_TIMEOUT_S,
            "features_list": [],
        }
    
    session = st.session_state.enroll_session
    
    # =========================================================================
    # CONTROL BUTTONS
    # =========================================================================
    
    st.markdown("---")
        
    col_a, col_b = st.columns(2)
        
    with col_a:
        if st.button("Start Auto Enrollment", type="primary", disabled=not bool(person_name) or session["active"]):
            st.session_state.enroll_session = {
                "active": True,
                "person_name": person_name.strip(),
                "target_sequences": int(target_sequences),
                "frames_per_sequence": int(frames_per_sequence),
                "timeout_s": int(sequence_timeout),
                "features_list": [],
            }
            st.rerun()
        
    with col_b:
        if st.button("Reset", disabled=not session["active"]):
            st.session_state.enroll_session = {
                "active": False,
                "person_name": "",
                "target_sequences": UI_CONFIG.DEFAULT_SEQUENCES,
                "frames_per_sequence": UI_CONFIG.DEFAULT_FRAMES_PER_SEQ,
                "timeout_s": UI_CONFIG.DEFAULT_TIMEOUT_S,
                "features_list": [],
            }
            # Clear enrollment processor
            if 'enrollment_processor' in st.session_state:
                del st.session_state.enrollment_processor
            st.rerun()
    
    # =========================================================================
    # RUN ENROLLMENT IF ACTIVE
    # =========================================================================
    
    if session["active"]:
        run_auto_enrollment(user_id)
        
    # =========================================================================
    # INSTRUCTIONS
    # =========================================================================
    
    st.markdown("---")
    st.markdown("""
    ### Enrollment Instructions
    
    1. Enter the person's name above
    2. Click **Start Auto Enrollment**
    3. Walk naturally in front of the camera
    4. Keep your **full body visible** (head to feet)
    5. Walk **towards and away** from the camera
    6. The system will automatically capture gait sequences
    
    **Tips for best results:**
    - Good lighting (avoid shadows)
    - 2-4 meters from camera
    - Consistent walking speed
    """)


def run_auto_enrollment(user_id: str):
    """
    WebRTC-based auto enrollment with progress tracking.
    Captures multiple gait sequences automatically.
    """
    session = st.session_state.enroll_session
    gait_model = st.session_state.gait_model
    data_manager = st.session_state.data_manager
    
    person_name = session.get("person_name", "").strip()
    target_sequences = int(session.get("target_sequences", 8))
    frames_needed = int(session.get("frames_per_sequence", 45))
    
    if not person_name:
        st.error("Please enter a person name.")
        return
    
    # Thread-safe enrollment processor
    class EnrollmentProcessor:
        def __init__(self):
            self.features_list = []
            self.current_poses = []
            self.status_message = 'Initializing...'
            self.progress = 0.0
            self.is_complete = False
            
        def process(self, frame: av.VideoFrame):
            """Process enrollment frames"""
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            
            # Create progress overlay
            progress_frame = img.copy()
            
            if not self.is_complete:
                # Detect pose
                batch_results = gait_model.detect_person_status_batch([img])
                
                if batch_results:
                    status, keypoints = batch_results[0]
                    
                    if keypoints is not None:
                        normalized = gait_model._normalize_pose(keypoints)
                        
                        if normalized is not None and len(normalized) == 34:
                            self.current_poses.append(normalized.astype(np.float32))
                            self.status_message = f"Capturing... {len(self.current_poses)}/{frames_needed} frames"
                            
                            # Check if we have enough frames for a sequence
                            if len(self.current_poses) >= frames_needed:
                                # Compute features
                                sequence = np.stack(self.current_poses[-frames_needed:], axis=0)
                                features = gait_model.compute_features_from_sequence(sequence)
                                
                                if features is not None and np.isfinite(features).all() and not np.allclose(features, 0):
                                    self.features_list.append(features)
                                    self.current_poses = []
                                    self.progress = len(self.features_list) / target_sequences
                                    self.status_message = f"Sequence {len(self.features_list)}/{target_sequences} captured!"
                                    
                                    # Check if enrollment complete
                                    if len(self.features_list) >= target_sequences:
                                        self.is_complete = True
                                        self.status_message = "Enrollment complete!"
                                else:
                                    self.current_poses = []
                                    self.status_message = "Invalid sequence, retrying..."
                        else:
                            self.status_message = "Keep full body visible"
                    else:
                        self.status_message = "No person detected"
            
            # Draw progress bar
            seq_count = len(self.features_list)
            cv2.rectangle(progress_frame, (0, 0), (640, 50), 
                         (0, 100, 0) if not self.is_complete else (0, 200, 0), -1)
            cv2.putText(progress_frame, 
                       f"Sequence {seq_count}/{target_sequences} | Frames: {len(self.current_poses)}/{frames_needed}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return av.VideoFrame.from_ndarray(progress_frame, format="bgr24")
    
    # Initialize processor
    if 'enrollment_processor' not in st.session_state:
        st.session_state.enrollment_processor = EnrollmentProcessor()
    
    processor = st.session_state.enrollment_processor
    
    # =========================================================================
    # WEBRTC STREAMER
    # =========================================================================
    
    st.info(f"Enrolling **{person_name}** - Walk naturally in front of the camera")
    
    # WebRTC configuration with STUN and TURN servers for ngrok/external access
    rtc_configuration = RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
        ],
        "iceCandidatePoolSize": 10,
        }
    )
    
    try:
        webrtc_ctx = webrtc_streamer(
            key="enrollment-camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_frame_callback=processor.process,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        
        # =====================================================================
        # STATUS DISPLAY
        # =====================================================================
        
        progress_bar = st.progress(processor.progress)
        st.info(processor.status_message)
        
        # =====================================================================
        # SAVE ENROLLMENT
        # =====================================================================
        
        if processor.is_complete and len(processor.features_list) >= target_sequences:
            final_features = np.mean(np.array(processor.features_list), axis=0)
            data_manager.save_gait_data(user_id, person_name, final_features)
            
            st.success(f"Successfully enrolled **{person_name}**!")
            logger.info(f"Enrollment complete: {person_name} ({target_sequences} sequences)")
            
            # Reset enrollment session
            st.session_state.enroll_session = {
                "active": False,
                "person_name": "",
                "target_sequences": UI_CONFIG.DEFAULT_SEQUENCES,
                "frames_per_sequence": UI_CONFIG.DEFAULT_FRAMES_PER_SEQ,
                "timeout_s": UI_CONFIG.DEFAULT_TIMEOUT_S,
                "features_list": [],
            }
            
            # Clear enrollment processor
            if 'enrollment_processor' in st.session_state:
                del st.session_state.enrollment_processor
            
            # Invalidate cache
            if st.session_state.get("enrolled_cache_user") == user_id:
                st.session_state.enrolled_cache = {}
                st.session_state.enrolled_cache_ts = 0.0
            
            time.sleep(2)
            st.rerun()
    except (KeyError, AttributeError, RuntimeError) as e:
        # Handle streamlit-webrtc errors (Python 3.13 compatibility issues)
        error_msg = str(e)
        if any(keyword in error_msg for keyword in ["frontend", "is_alive", "event loop", "NoneType"]):
            st.warning("⚠️ Enrollment camera issue (Python 3.13 compatibility). Please use Python 3.11 for best results.")
            st.info("WebRTC may have errors. Consider downgrading to Python 3.11 for stable operation.")
        else:
            raise


# =============================================================================
# LEGACY FUNCTION (Kept for compatibility)
# =============================================================================

def perform_enrollment(person_name: str, target_sequences: int, user_id: str):
    """Legacy enrollment function - redirects to new flow"""
    st.warning("Use the 'Start Auto Enrollment' button for the new enrollment flow.")
