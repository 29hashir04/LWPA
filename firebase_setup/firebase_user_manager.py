"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       FIREBASE GAIT DATA MANAGER                             ║
║                                                                              ║
║  User data management UI for:                                                ║
║  - Viewing enrolled gait patterns                                            ║
║  - Exporting data as JSON                                                    ║
║  - Clearing all user data                                                    ║
║  - Deleting individual patterns                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import numpy as np
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
EXPECTED_FEATURE_LEN = 136  # Current model emits 34*4 features

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _coerce_features(features: Any) -> Optional[np.ndarray]:
    """
    Ensure we only work with real numeric feature vectors.
    
    Args:
        features: Raw feature data (list, array, or other)
    
    Returns:
        1D float32 array, or None if invalid
    """
    if features is None:
        return None
    try:
        arr = np.asarray(features, dtype=np.float32).reshape(-1)
    except Exception:
        return None

    # Must be finite numbers
    if arr.size == 0 or not np.isfinite(arr).all():
        return None

    return arr


def _is_plausible_feature_vector(arr: np.ndarray) -> bool:
    """
    Filters out obviously corrupt or placeholder vectors.
    
    Args:
        arr: Feature array to validate
    
    Returns:
        True if vector appears valid, False otherwise
    """
    if arr is None or arr.size == 0:
        return False
    # Reject all-zeros (common sign of a failed capture)
    if np.allclose(arr, 0.0):
        return False
    return True


# =============================================================================
# MAIN UI FUNCTION
# =============================================================================

def render_firebase_data_management():
    """
    Render the 'My Data' tab UI.
    
    Features:
    - Display enrolled pattern count and stats
    - Export data as JSON
    - Clear all data (with confirmation)
    - Delete individual patterns
    - Refresh cache
    """
    st.markdown("### My Gait Data")
    
    # -------------------------------------------------------------------------
    # USER VALIDATION
    # -------------------------------------------------------------------------
    user_data = st.session_state.get('user_data', {})
    user_id = st.session_state.get('user_id', '')
    
    if not user_id:
        st.warning("Please log in")
        return
    
    # -------------------------------------------------------------------------
    # LOAD DATA MANAGER
    # -------------------------------------------------------------------------
    # Use the SAME data manager as Enrollment/Recognition to avoid stale/duplicate behavior
    from data.gait_data_manager import GaitDataManager
    manager = st.session_state.get("data_manager") or GaitDataManager()
    
    enrolled = manager.load_user_gait_data(user_id)
    
    if not enrolled:
        st.info("No patterns enrolled. Go to 'Enroll' tab to add some!")
        return
    
    # -------------------------------------------------------------------------
    # STATS DISPLAY
    # -------------------------------------------------------------------------
    st.markdown(f"**{len(enrolled)} enrolled patterns**")
    col1, col2 = st.columns(2)
    col1.metric("Patterns", len(enrolled))
    col2.metric("Features", sum(int(np.asarray(f).size) for f in enrolled.values()))
    st.caption(f"Only valid numeric patterns are shown. Expected feature length: {EXPECTED_FEATURE_LEN}.")
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # ACTION BUTTONS
    # -------------------------------------------------------------------------
    col_a, col_b, col_c = st.columns(3)
    
    # Export button
    with col_a:
        if st.button("Export", key="export_btn"):
            export: Dict[str, Any] = {}
            for name, features in enrolled.items():
                arr = _coerce_features(features)
                if not _is_plausible_feature_vector(arr):
                    continue
                export[name] = {
                    'features': arr.tolist(),
                    'feature_count': int(arr.size),
                    'exported_at': datetime.now().isoformat()
                }
            st.download_button(
                "Download JSON",
                json.dumps(export, indent=2),
                f"gait_data_{datetime.now().strftime('%Y%m%d')}.json",
                "application/json"
            )
    
    # Clear all button
    with col_b:
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
        
        if not st.session_state.confirm_clear:
            if st.button("Clear All", key="clear_btn"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Confirm?")
            c1, c2 = st.columns(2)
            if c1.button("Yes"):
                count = manager.clear_all_user_data(user_id)
                try:
                    manager.invalidate_user_cache(user_id)
                except Exception:
                    pass
                if st.session_state.get("enrolled_cache_user") == user_id:
                    st.session_state.enrolled_cache = {}
                    st.session_state.enrolled_cache_ts = 0.0
                st.session_state.confirm_clear = False
                st.success(f"Cleared {count}")
                st.rerun()
            if c2.button("No"):
                st.session_state.confirm_clear = False
                st.rerun()
    
    # Refresh button
    with col_c:
        if st.button("Refresh"):
            try:
                manager.invalidate_user_cache(user_id)
            except Exception:
                pass
            if st.session_state.get("enrolled_cache_user") == user_id:
                st.session_state.enrolled_cache = {}
                st.session_state.enrolled_cache_ts = 0.0
            st.rerun()
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # PATTERN LIST
    # -------------------------------------------------------------------------
    for name, features in enrolled.items():
        arr = _coerce_features(features)
        if not _is_plausible_feature_vector(arr):
            # Shouldn't happen because load_user_gait_data filters already
            continue

        with st.expander(f"{name}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Features:** {int(arr.size)}")
                if arr.size != EXPECTED_FEATURE_LEN:
                    st.warning(f"Unexpected feature length ({arr.size}). This pattern may be from an older version.")
                st.write(f"**Mean:** {float(np.mean(arr)):.4f}")
            
            with col2:
                st.write(f"**Std:** {float(np.std(arr)):.4f}")
            
            with col3:
                safe_key = name.replace(' ', '_').replace('.', '_')
                if st.button("Delete", key=f"del_{safe_key}"):
                    if manager.delete_gait_data(user_id, name):
                        try:
                            manager.invalidate_user_cache(user_id)
                        except Exception:
                            pass
                        if st.session_state.get("enrolled_cache_user") == user_id:
                            st.session_state.enrolled_cache = {}
                            st.session_state.enrolled_cache_ts = 0.0
                        st.success(f"Deleted {name}")
                        st.rerun()
