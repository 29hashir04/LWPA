"""
Firebase Configuration Module
Handles Firebase Admin SDK and Pyrebase initialization.
Uses proper logging instead of print statements.
"""
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
import pyrebase
import os
import logging
from typing import Optional, Dict, Any
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton
_config_lock = Lock()
_firebase_config = None


class FirebaseConfig:
    """Firebase configuration manager with proper error handling"""
    
    def __init__(self):
        self.app = None
        self.db = None
        self.storage_bucket = None
        self.pyrebase_app = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase services"""
        try:
            # Check if already initialized
            if firebase_admin._apps:
                self.app = firebase_admin.get_app()
                logger.info("Using existing Firebase app")
            else:
                self._setup_firebase_admin()
            
            # Initialize Firestore
            if self.app:
                self.db = firestore.client()
                self.storage_bucket = storage.bucket()
            else:
                logger.warning("Firebase app not initialized; skipping Firestore/Storage initialization.")
                self.db = None
                self.storage_bucket = None
            
            # Setup Pyrebase for client-side auth
            self._setup_pyrebase()
            
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            logger.info("Using local authentication fallback")
    
    def _setup_firebase_admin(self):
        """Setup Firebase Admin SDK with credentials"""
        try:
            # Try environment variable first
            service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
            
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                logger.info(f"Using service account from: {service_account_path}")
            else:
                # Try Streamlit secrets
                import streamlit as st
                firebase_secrets = st.secrets.get("firebase", {})
                
                if not firebase_secrets:
                    logger.warning("No Firebase credentials found")
                    return
                
                cred_dict = {
                    "type": firebase_secrets.get("type", "service_account"),
                    "project_id": firebase_secrets["project_id"],
                    "private_key_id": firebase_secrets.get("private_key_id", ""),
                    "private_key": firebase_secrets["private_key"].replace('\\n', '\n'),
                    "client_email": firebase_secrets["client_email"],
                    "client_id": firebase_secrets.get("client_id", ""),
                    "auth_uri": firebase_secrets.get("auth_uri", "https://accounts.google.com/o/oauth2/auth"),
                    "token_uri": firebase_secrets.get("token_uri", "https://oauth2.googleapis.com/token"),
                    "auth_provider_x509_cert_url": firebase_secrets.get("auth_provider_x509_cert_url", ""),
                    "client_x509_cert_url": firebase_secrets.get("client_x509_cert_url", "")
                }
                cred = credentials.Certificate(cred_dict)
                logger.info("Using Firebase credentials from Streamlit secrets")
            
            # Initialize app
            project_id = cred.project_id
            self.app = firebase_admin.initialize_app(cred, {
                'storageBucket': f"{project_id}.appspot.com"
            })
            logger.info(f"Firebase Admin initialized for project: {project_id}")
            
        except Exception as e:
            logger.error(f"Firebase Admin setup failed: {e}")
            self.app = None
    
    def _setup_pyrebase(self):
        """Setup Pyrebase for client-side authentication"""
        try:
            import streamlit as st
            firebase_secrets = st.secrets.get("firebase", {})
            
            if not firebase_secrets:
                logger.warning("No Firebase secrets for Pyrebase")
                return
            
            project_id = firebase_secrets.get("project_id", "")
            
            pyrebase_config = {
                "apiKey": firebase_secrets.get("api_key", ""),
                "authDomain": f"{project_id}.firebaseapp.com",
                "databaseURL": f"https://{project_id}.firebaseio.com",
                "projectId": project_id,
                "storageBucket": f"{project_id}.appspot.com",
                "messagingSenderId": firebase_secrets.get("messaging_sender_id", ""),
                "appId": firebase_secrets.get("app_id", "")
            }
            
            self.pyrebase_app = pyrebase.initialize_app(pyrebase_config)
            logger.info("Pyrebase initialized successfully")
            
        except Exception as e:
            logger.warning(f"Pyrebase setup failed: {e}")
            self.pyrebase_app = None
    
    def get_firestore(self):
        """Get Firestore client"""
        return self.db
    
    def get_storage(self):
        """Get Storage bucket"""
        return self.storage_bucket
    
    def get_pyrebase_auth(self):
        """Get Pyrebase auth instance"""
        if self.pyrebase_app:
            return self.pyrebase_app.auth()
        return None

    def is_initialized(self):
        """Check if Firebase is properly initialized"""
        return self.app is not None


def get_firebase_config():
    """Get or create Firebase config singleton (thread-safe)"""
    global _firebase_config
    
    if _firebase_config is None:
        with _config_lock:
            if _firebase_config is None:
                _firebase_config = FirebaseConfig()
    
    return _firebase_config


def get_firestore_db():
    """Get Firestore database client"""
    config = get_firebase_config()
    return config.get_firestore()


def get_firebase_storage():
    """Get Firebase storage bucket"""
    config = get_firebase_config()
    return config.get_storage()


def get_firebase_auth():
    """Get Pyrebase auth instance"""
    config = get_firebase_config()
    return config.get_pyrebase_auth()


def is_firebase_available():
    """Check if Firebase is available"""
    config = get_firebase_config()
    return config.is_initialized()