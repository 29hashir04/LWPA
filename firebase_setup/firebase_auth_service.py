"""
Firebase Authentication Service
With session persistence across page refreshes.
"""
import streamlit as st
import firebase_admin
from firebase_admin import auth as firebase_auth
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_query_params() -> dict:
    """
    Streamlit query params compatibility:
    - New: st.query_params
    - Old: st.experimental_get_query_params()
    """
    try:
        return dict(st.query_params)
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}


def _clear_query_params() -> None:
    """Clear query params in a version-compatible way."""
    try:
        st.query_params.clear()
        return
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def _set_query_param(key: str, value: str) -> None:
    """Set a single query param in a version-compatible way."""
    try:
        st.query_params[key] = value
        return
    except Exception:
        try:
            st.experimental_set_query_params(**{key: value})
        except Exception:
            pass


try:
    from .firebase_config import get_firebase_auth, get_firestore_db
except ImportError:
    def get_firebase_auth():
        return None
    def get_firestore_db():
        return None

# Session persistence
SESSION_FILE = Path(".firebase_sessions.json")
LAST_SESSION_FILE = Path(".firebase_last_session")  # Stores last active session ID
SESSION_EXPIRY_HOURS = 24


class FirebaseSessionManager:
    """Manages persistent Firebase sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        try:
            if SESSION_FILE.exists():
                with open(SESSION_FILE, 'r') as f:
                    self.sessions = json.load(f)
                self._cleanup_expired()
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            self.sessions = {}
    
    def _save_sessions(self):
        try:
            with open(SESSION_FILE, 'w') as f:
                json.dump(self.sessions, f)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _cleanup_expired(self):
        now = datetime.now().isoformat()
        expired = [k for k, v in self.sessions.items() if v.get('expires_at', '') < now]
        for k in expired:
            del self.sessions[k]
        if expired:
            self._save_sessions()
    
    def save_session(self, session_id: str, user_data: Dict, refresh_token: str):
        """Save session for persistence"""
        self.sessions[session_id] = {
            'user_data': user_data,
            'refresh_token': refresh_token,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=SESSION_EXPIRY_HOURS)).isoformat()
        }
        self._save_sessions()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session if valid"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if session.get('expires_at', '') < datetime.now().isoformat():
            del self.sessions[session_id]
            self._save_sessions()
            return None
        
        return session
    
    def delete_session(self, session_id: str):
        """Delete session on logout"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
        # Also clear last session file
        try:
            if LAST_SESSION_FILE.exists():
                LAST_SESSION_FILE.unlink()
        except Exception:
            pass
    
    def save_last_session_id(self, session_id: str):
        """Save the last active session ID for auto-restore"""
        try:
            LAST_SESSION_FILE.write_text(session_id)
            logger.info(f"Saved last session ID: {session_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to save last session ID: {e}")
    
    def get_last_session_id(self) -> Optional[str]:
        """Get the last active session ID"""
        try:
            if LAST_SESSION_FILE.exists():
                session_id = LAST_SESSION_FILE.read_text().strip()
                if session_id:
                    return session_id
        except Exception as e:
            logger.error(f"Failed to read last session ID: {e}")
        return None


class FirebaseAuthService:

    def __init__(self):
        self.firebase_auth = get_firebase_auth()
        self.db = get_firestore_db()
        self.users_collection = "users"
        self.gait_data_collection = "gait_data"
        self.session_manager = FirebaseSessionManager()
    
    def signup_user(self, email: str, password: str, display_name: str) -> Dict[str, Any]:
        try:
            user = self.firebase_auth.create_user_with_email_and_password(email, password)
            self.firebase_auth.update_profile(user['idToken'], display_name=display_name)
            
            return {
                'success': True,
                'user_id': user['localId'],
                'user_data': user,
                'message': f"Account created successfully for {display_name}!"
            }
            
        except Exception as e:
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message:
                return {'success': False, 'message': 'An account with this email already exists. Please login instead.'}
            elif "WEAK_PASSWORD" in error_message:
                return {'success': False, 'message': 'Password is too weak. Use at least 6 characters.'}
            elif "INVALID_EMAIL" in error_message:
                return {'success': False, 'message': 'Please enter a valid email address.'}
            elif "OPERATION_NOT_ALLOWED" in error_message:
                return {'success': False, 'message': 'Email/password signup is not enabled. Contact admin.'}
            else:
                import logging
                logging.error(f"Firebase signup error: {error_message}")
                return {'success': False, 'message': 'Signup failed. Please try again.'}
    
    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        try:
            user = self.firebase_auth.sign_in_with_email_and_password(email, password)
            
            # Create persistent session
            import secrets
            session_id = secrets.token_urlsafe(32)
            self.session_manager.save_session(session_id, user, user['refreshToken'])
            
            # Save as last active session for auto-restore on refresh
            self.session_manager.save_last_session_id(session_id)
            
            return {
                'success': True,
                'user_id': user['localId'],
                'user_data': user,
                'id_token': user['idToken'],
                'refresh_token': user['refreshToken'],
                'session_id': session_id,
                'message': "Login successful!"
            }
                
        except Exception as e:
            error_message = str(e)
            # Handle all invalid credential errors
            if any(err in error_message for err in [
                "INVALID_PASSWORD", 
                "EMAIL_NOT_FOUND", 
                "INVALID_LOGIN_CREDENTIALS",
                "INVALID_EMAIL"
            ]):
                return {'success': False, 'message': 'Invalid email or password. Please check your credentials.'}
            elif "USER_DISABLED" in error_message:
                return {'success': False, 'message': 'This account has been disabled.'}
            elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_message:
                return {'success': False, 'message': 'Too many failed attempts. Please try again later.'}
            elif "USER_NOT_FOUND" in error_message:
                return {'success': False, 'message': 'No account found with this email. Please sign up first.'}
            else:
                # Log the actual error for debugging, but show friendly message
                import logging
                logging.error(f"Firebase auth error: {error_message}")
                return {'success': False, 'message': 'Login failed. Please try again.'}
    
    def restore_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Restore session from stored data"""
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.warning(f"No session found for ID: {session_id[:8]}...")
            return None
        
        try:
            # Get refresh token from session
            refresh_token = session.get('refresh_token')
            if not refresh_token:
                logger.error("No refresh token in session")
                return None
            
            # Try to refresh the token
            logger.debug("Attempting token refresh...")
            refreshed = self.firebase_auth.refresh(refresh_token)
            
            if refreshed and 'idToken' in refreshed:
                # Update session with new tokens
                user_data = session['user_data'].copy()
                user_data['idToken'] = refreshed['idToken']
                
                new_refresh_token = refreshed.get('refreshToken', refresh_token)
                user_data['refreshToken'] = new_refresh_token
                
                # Save updated session
                self.session_manager.save_session(session_id, user_data, new_refresh_token)
                
                logger.info("Token refresh successful")
                return {
                    'success': True,
                    'user_data': user_data,
                    'id_token': refreshed['idToken'],
                    'refresh_token': new_refresh_token
                }
            else:
                logger.error("Token refresh returned no idToken")
                return None
                
        except Exception as e:
            logger.error(f"Session restore failed: {e}")
            # Don't delete session immediately - might be temporary network issue
            # Only delete if it's an auth error
            error_str = str(e).upper()
            if any(err in error_str for err in ['INVALID', 'EXPIRED', 'REVOKED', 'USER_NOT_FOUND']):
                logger.info("Deleting invalid session")
                self.session_manager.delete_session(session_id)
            return None
    
    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            user_doc = self.db.collection(self.users_collection).document(user_id).get()
            if user_doc.exists:
                return user_doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return None
    
    def update_user_data(self, user_id: str, updates: Dict[str, Any]) -> bool:
        try:
            self.db.collection(self.users_collection).document(user_id).update(updates)
            return True
        except Exception as e:
            logger.error(f"Error updating user data: {e}")
            return False
    
    def update_user_gait_count(self, user_id: str, count: int) -> bool:
        return self.update_user_data(user_id, {'gait_data_count': count})
    
    def reset_password(self, email: str) -> Dict[str, Any]:
        try:
            self.firebase_auth.send_password_reset_email(email)
            return {'success': True, 'message': 'Password reset email sent successfully'}
        except Exception as e:
            if "EMAIL_NOT_FOUND" in str(e):
                return {'success': False, 'message': 'No account found with this email address'}
            return {'success': False, 'message': f"Password reset failed: {str(e)}"}
    
    def delete_user_account(self, user_id: str) -> Dict[str, Any]:
        try:
            firebase_auth.delete_user(user_id)
            self.db.collection(self.users_collection).document(user_id).delete()
            
            gait_docs = self.db.collection(self.gait_data_collection).where('user_id', '==', user_id).stream()
            for doc in gait_docs:
                doc.reference.delete()
            
            return {'success': True, 'message': 'Account and all data deleted successfully'}
        except Exception as e:
            return {'success': False, 'message': f"Account deletion failed: {str(e)}"}
    
    def verify_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        try:
            decoded_token = firebase_auth.verify_id_token(id_token, check_revoked=False)
            return decoded_token
        except Exception:
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        try:
            user = self.firebase_auth.refresh(refresh_token)
            return user
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    def logout(self, session_id: str):
        """Logout and clear session"""
        if session_id:
            self.session_manager.delete_session(session_id)


def _try_restore_session():
    """Try to restore session from URL query param OR last session file"""
    try:
        session_id = None
        
        # First try URL query params
        params = _get_query_params()
        raw = params.get('sid', None)
        if isinstance(raw, list):
            session_id = raw[0] if raw else None
        else:
            session_id = raw
        
        # If no URL param, try last session file
        if not session_id:
            auth_service = FirebaseAuthService()
            session_id = auth_service.session_manager.get_last_session_id()
            if session_id:
                logger.info(f"Found last session ID from file: {session_id[:8]}...")
        
        if not session_id:
            logger.debug("No session ID found (URL or file)")
            return False
        
        logger.info(f"Attempting to restore session: {session_id[:8]}...")
        
        auth_service = FirebaseAuthService()
        result = auth_service.restore_session(session_id)
        
        if result and result.get('success'):
            user_data = result['user_data']
            st.session_state['authenticated'] = True
            st.session_state['user_id'] = user_data.get('localId', '')
            st.session_state['user_data'] = user_data
            st.session_state['id_token'] = result['id_token']
            st.session_state['refresh_token'] = result['refresh_token']
            st.session_state['session_id'] = session_id
            logger.info(f"Session restored for user: {user_data.get('email', 'unknown')}")
            return True
        else:
            logger.warning("Session restore failed - invalid or expired session")
            # Clear the bad session file
            try:
                if LAST_SESSION_FILE.exists():
                    LAST_SESSION_FILE.unlink()
            except Exception:
                pass
            _clear_query_params()
            return False
            
    except Exception as e:
        logger.error(f"Session restore error: {e}", exc_info=True)
        return False


def render_firebase_login_signup():
    """Render Firebase login UI with session persistence"""
    
    # Try to restore session first
    if not st.session_state.get('authenticated', False):
        if _try_restore_session():
            st.rerun()
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: #1f77b4; margin-bottom: 2rem;'>Live Walking Pattern Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Reset Password"])
    
    auth_service = FirebaseAuthService()
    
    with tab1:
        st.markdown("### Login to Your Account")
        
        with st.form("firebase_login_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me", value=True)
            login_button = st.form_submit_button("Login", type="primary")
            
            if login_button:
                if email and password:
                    with st.spinner("Authenticating..."):
                        result = auth_service.authenticate_user(email, password)
                    
                    if result['success']:
                        st.success(result['message'])
                        st.session_state['authenticated'] = True
                        st.session_state['user_id'] = result['user_id']
                        st.session_state['user_data'] = result['user_data']
                        st.session_state['id_token'] = result['id_token']
                        st.session_state['refresh_token'] = result['refresh_token']
                        st.session_state['session_id'] = result.get('session_id', '')
                        
                        # Set session ID in URL for persistence
                        if remember_me and result.get('session_id'):
                            _set_query_param('sid', result['session_id'])
                            logger.info(f"Session ID set in URL: {result['session_id'][:8]}...")
                        
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.markdown("### Create New Account")
        
        with st.form("firebase_signup_form"):
            display_name = st.text_input("Full Name", placeholder="Your Full Name")
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", help="At least 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_button = st.form_submit_button("Sign Up", type="primary")
            
            if signup_button:
                if display_name and email and password and confirm_password:
                    if password == confirm_password:
                        if len(password) >= 6:
                            with st.spinner("Creating account..."):
                                result = auth_service.signup_user(email, password, display_name)
                            
                            if result['success']:
                                st.success(result['message'])
                                st.info("Please login with your new account.")
                            else:
                                st.error(result['message'])
                        else:
                            st.error("Password must be at least 6 characters")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")
    
    with tab3:
        st.markdown("### Reset Password")
        
        with st.form("firebase_reset_form"):
            email = st.text_input("Email", placeholder="your.email@example.com")
            reset_button = st.form_submit_button("Send Reset Email", type="primary")
            
            if reset_button:
                if email:
                    with st.spinner("Sending reset email..."):
                        result = auth_service.reset_password(email)
                    
                    if result['success']:
                        st.success(result['message'])
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter your email address")


def render_firebase_logout_button():
    """Render logout button"""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col3:
        if st.button("Logout", key="firebase_logout_btn"):
            # Clear stored session
            session_id = st.session_state.get('session_id')
            if session_id:
                auth_service = FirebaseAuthService()
                auth_service.logout(session_id)
            
            # Clear last session file
            try:
                if LAST_SESSION_FILE.exists():
                    LAST_SESSION_FILE.unlink()
                    logger.info("Cleared last session file on logout")
            except Exception:
                pass
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Clear URL params
            _clear_query_params()
            
            st.rerun()


def check_firebase_authentication():
    """Check authentication with session restoration"""
    
    # If already authenticated in this session, trust it (don't re-verify every time)
    if st.session_state.get('authenticated', False):
        # Quick check - if we have valid session state, we're good
        if st.session_state.get('user_id') and st.session_state.get('user_data'):
            return True
    
    # Try to restore from URL if not authenticated
    logger.debug("Not authenticated, attempting session restore...")
    if _try_restore_session():
        return True
    
    return False


def get_current_firebase_user():
    """Get current authenticated user"""
    if check_firebase_authentication():
        return st.session_state.get('user_data', {})
    return None
