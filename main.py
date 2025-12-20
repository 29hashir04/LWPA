"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              LIVE WALKING PATTERN ANALYSIS (LWPA) - Production               ║
║                                                                              ║
║  Production-grade gait recognition system with:                              ║
║  - GPU-accelerated YOLOv8 pose detection                                     ║
║  - 136-dimensional normalized gait features                                  ║
║  - Temporal smoothing with EMA + voting                                      ║
║  - Firebase authentication with session persistence                          ║
║  - Real-time 30+ FPS processing                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import logging
import queue
import torch

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title="LWPA - Gait Recognition",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# IMPORTS
# =============================================================================
from firebase_setup.firebase_auth_service import (
    render_firebase_login_signup, 
    render_firebase_logout_button,
    check_firebase_authentication, 
    get_current_firebase_user
)
from firebase_setup.firebase_user_manager import (
    render_firebase_data_management
)
from ui.render_functions import (
    render_optimized_main_application,
    render_optimized_enrollment_page
)

# =============================================================================
# CUSTOM CSS - Professional Dark Theme
# =============================================================================
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    .main {
        padding: 1rem;
    }
    
    /* ===== HEADERS ===== */
    h1, h2, h3 {
        color: #00ff88;
        font-weight: 600;
    }
    
    h1 {
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00ff88;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid #00ff88;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
    }
    
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        color: black;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 500;
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 136, 0.2);
        border-color: #00ff88;
    }
    
    /* ===== SUCCESS/ERROR/INFO BOXES ===== */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
    }
    
    .stError {
        background: rgba(255, 0, 0, 0.1);
        border-left: 4px solid #ff4444;
    }
    
    .stInfo {
        background: rgba(0, 136, 255, 0.1);
        border-left: 4px solid #0088ff;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ff88, #00cc6a);
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #00ff88;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a, #1a1a2e);
    }
    
    /* ===== HEADER BANNER ===== */
    .header-banner {
        background: linear-gradient(90deg, #0a0a0a, #1a1a2e, #0a0a0a);
        border: 1px solid #00ff88;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    }
    
    .header-title {
        font-size: 1.5rem;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        margin: 0;
    }
    
    .header-subtitle {
        color: #666;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state with defaults"""
    defaults = {
        'gait_model': None,
        'data_manager': None,
        'camera_running': False,
        'frame_queue': queue.Queue(maxsize=10),
        'detection_status': 'Idle',
        'recognition_result': {'name': 'Unknown', 'confidence': 0.0}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# GPU STATUS LOGGING
# =============================================================================
def log_system_info():
    """Log system information on startup"""
    logger.info("=" * 60)
    logger.info("LWPA - Live Walking Pattern Analysis - Starting")
    logger.info("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU detected - Running on CPU")
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info("=" * 60)

# Log on first run
if 'system_logged' not in st.session_state:
    log_system_info()
    st.session_state.system_logged = True

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point"""
    
    # Check authentication
    if not check_firebase_authentication():
        # Show login page
        render_firebase_login_signup()
        return
    
    # =========================================================================
    # AUTHENTICATED USER VIEW
    # =========================================================================
    
    # Header banner
    st.markdown("""
    <div class="header-banner">
        <p class="header-title">Live Walking Pattern Analysis</p>
        <p class="header-subtitle">Real-Time Gait Recognition System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "Recognition",
        "Enroll", 
        "My Data"
    ])
    
    with tab1:
        render_optimized_main_application()
    
    with tab2:
        render_optimized_enrollment_page()
    
    with tab3:
        render_firebase_data_management()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
