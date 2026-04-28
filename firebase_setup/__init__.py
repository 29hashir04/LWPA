"""
Firebase Setup Package
"""
from .firebase_config import (
    get_firebase_config, 
    get_firestore_db, 
    get_firebase_storage, 
    get_firebase_auth
)
from .firebase_auth_service import (
    FirebaseAuthService, 
    render_firebase_login_signup, 
    render_firebase_logout_button, 
    check_firebase_authentication, 
    get_current_firebase_user
)
from .firebase_user_manager import (
    render_firebase_data_management
)

__all__ = [
    'get_firebase_config', 
    'get_firestore_db', 
    'get_firebase_storage', 
    'get_firebase_auth',
    'FirebaseAuthService', 
    'render_firebase_login_signup', 
    'render_firebase_logout_button', 
    'check_firebase_authentication', 
    'get_current_firebase_user',
    'render_firebase_data_management'
]
