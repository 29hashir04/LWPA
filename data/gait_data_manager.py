"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         GAIT DATA MANAGER                                    ║
║                                                                              ║
║  Production-grade data management with:                                      ║
║  - User-isolated storage                                                     ║
║  - Validation and sanitization                                               ║
║  - Memory caching with TTL                                                   ║
║  - Thread-safe operations                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pickle
import os
import logging
import threading
import time
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    """Data manager configuration"""
    BASE_DIR: str = "gait_data"
    EXPECTED_FEATURE_LEN: int = 136  # 34 * 4 (mean, std, velocity, range)
    CACHE_TTL_SECONDS: float = 30.0  # Cache invalidation time
    MAX_NAME_LENGTH: int = 50        # Maximum person name length


DATA_CONFIG = DataConfig()


# =============================================================================
# GAIT DATA MANAGER
# =============================================================================

class GaitDataManager:
    """
    Production-grade gait data manager with:
    - User-isolated directory structure
    - Feature vector validation
    - Thread-safe caching
    - Atomic file operations
    """
    
    def __init__(self, base_dir: str = DATA_CONFIG.BASE_DIR):
        """
        Initialize data manager.
        
        Args:
            base_dir: Root directory for gait data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Thread-safe cache
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()
        
        logger.info(f"GaitDataManager initialized: {self.base_dir}")
    
    # =========================================================================
    # PATH MANAGEMENT
    # =========================================================================
    
    def _get_user_dir(self, user_id: str) -> Path:
        """
        Get user-specific directory (creates if needed).
        
        Args:
            user_id: Firebase user ID
        
        Returns:
            Path to user directory
        """
        # Sanitize user ID for filesystem
        safe_id = self._sanitize_for_path(user_id) or "default"
        user_dir = self.base_dir / safe_id
        user_dir.mkdir(exist_ok=True)
        return user_dir
    
    def _sanitize_for_path(self, text: str) -> str:
        """Sanitize string for safe filesystem use"""
        if not text:
            return ""
        # Keep only alphanumeric, underscore, hyphen
        safe = "".join(c for c in text if c.isalnum() or c in ('_', '-'))
        return safe[:100]  # Limit length
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize person name for filename.
        
        Args:
            name: Person name
        
        Returns:
            Safe filename-compatible string
        """
        if not name:
            return "unnamed"
        
        # Replace non-alphanumeric with underscore
        safe = "".join(c if c.isalnum() else '_' for c in name)
        
        # Collapse multiple underscores
        while '__' in safe:
            safe = safe.replace('__', '_')
        
        # Strip leading/trailing underscores
        safe = safe.strip('_')
        
        # Limit length
        safe = safe[:DATA_CONFIG.MAX_NAME_LENGTH]
        
        return safe or "unnamed"
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_features(self, features: Any) -> Optional[np.ndarray]:
        """
        Validate and normalize feature vector.
        
        Args:
            features: Input features (any type)
        
        Returns:
            Validated float32 array, or None if invalid
        """
        if features is None:
            return None
        
        try:
            arr = np.asarray(features, dtype=np.float32).flatten()
        except Exception:
            logger.warning("Failed to convert features to array")
            return None
        
        # Check for empty
        if arr.size == 0:
            logger.warning("Empty feature vector")
            return None
        
        # Check for non-finite values
        if not np.isfinite(arr).all():
            logger.warning("Feature vector contains non-finite values")
            return None
        
        # Check for all zeros (indicates failed capture)
        if np.allclose(arr, 0.0):
            logger.warning("Feature vector is all zeros")
            return None
        
        return arr
    
    def _is_valid_feature_length(self, arr: np.ndarray) -> bool:
        """Check if feature length matches expected"""
        return arr.size == DATA_CONFIG.EXPECTED_FEATURE_LEN
    
    # =========================================================================
    # SAVE OPERATIONS
    # =========================================================================
    
    def save_gait_data(
        self, 
        user_id: str, 
        person_name: str, 
        features: np.ndarray
    ) -> str:
        """
        Save gait data for a person.
        
        Args:
            user_id: Firebase user ID
            person_name: Name of the person
            features: 136-dim feature vector
        
        Returns:
            Path to saved file
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not user_id:
            raise ValueError("user_id is required")
        
        if not person_name or not person_name.strip():
            raise ValueError("person_name is required")
        
        person_name = person_name.strip()
        
        # Validate features
        arr = self._validate_features(features)
        if arr is None:
            raise ValueError("Invalid feature vector")
        
        # Warn if unexpected length
        if not self._is_valid_feature_length(arr):
            logger.warning(
                f"Feature length {arr.size} differs from expected {DATA_CONFIG.EXPECTED_FEATURE_LEN}"
            )
        
        # Prepare data
        data = {
            'features': arr,
            'person_name': person_name,
            'user_id': user_id,
            'feature_dim': int(arr.size),
            'created_at': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        # Get paths
        user_dir = self._get_user_dir(user_id)
        safe_name = self._sanitize_name(person_name)
        filepath = user_dir / f"{safe_name}_gait.pkl"
        
        # Atomic write (write to temp, then rename)
        temp_path = filepath.with_suffix('.pkl.tmp')
        
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Rename (atomic on most filesystems)
            temp_path.replace(filepath)
            
        except Exception as e:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            raise e
        
        # Update cache
        with self._cache_lock:
            if user_id not in self._cache:
                self._cache[user_id] = {}
            self._cache[user_id][person_name] = arr
            self._cache_timestamps[user_id] = time.time()
        
        logger.info(f"Saved gait data: {person_name} ({arr.size} features)")
        return str(filepath)
    
    # =========================================================================
    # LOAD OPERATIONS
    # =========================================================================
    
    def load_user_gait_data(self, user_id: str) -> Dict[str, np.ndarray]:
        """
        Load all gait data for a user.
        
        Args:
            user_id: Firebase user ID
        
        Returns:
            Dict mapping person names to feature vectors
        """
        if not user_id:
            return {}
        
        # Check cache
        with self._cache_lock:
            if user_id in self._cache:
                cache_age = time.time() - self._cache_timestamps.get(user_id, 0)
                if cache_age < DATA_CONFIG.CACHE_TTL_SECONDS:
                    return self._cache[user_id].copy()
        
        # Load from disk
        enrolled = {}
        user_dir = self._get_user_dir(user_id)
        
        for filepath in user_dir.glob("*_gait.pkl"):
            try:
                data = self._load_single_file(filepath)
                if data:
                    name, features = data
                    enrolled[name] = features
                    
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
        
        # Update cache
        with self._cache_lock:
            self._cache[user_id] = enrolled.copy()
            self._cache_timestamps[user_id] = time.time()
        
        logger.info(f"Loaded {len(enrolled)} patterns for user")
        return enrolled
    
    def _load_single_file(self, filepath: Path) -> Optional[tuple]:
        """
        Load and validate single gait file.
        
        Args:
            filepath: Path to .pkl file
        
        Returns:
            (name, features) tuple, or None if invalid
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Handle dict format (new)
            if isinstance(data, dict):
                name = data.get('person_name', filepath.stem.replace('_gait', ''))
                features = data.get('features')
            else:
                # Handle raw array format (legacy)
                name = filepath.stem.replace('_gait', '').replace('_', ' ')
                features = data
            
            # Validate features
            arr = self._validate_features(features)
            if arr is None:
                logger.warning(f"Invalid features in {filepath}")
                return None
            
            return (name, arr)
            
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None
    
    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================
    
    def delete_gait_data(self, user_id: str, person_name: str) -> bool:
        """
        Delete gait pattern for a person.
        
        Args:
            user_id: Firebase user ID
            person_name: Name to delete
        
        Returns:
            True if deleted, False if not found
        """
        user_dir = self._get_user_dir(user_id)
        safe_name = self._sanitize_name(person_name)
        filepath = user_dir / f"{safe_name}_gait.pkl"
        
        if filepath.exists():
            filepath.unlink()
            
            # Update cache
            with self._cache_lock:
                if user_id in self._cache:
                    self._cache[user_id].pop(person_name, None)
            
            logger.info(f"Deleted: {person_name}")
            return True
        
        return False
    
    def clear_all_user_data(self, user_id: str) -> int:
        """
        Delete all gait data for a user.
        
        Args:
            user_id: Firebase user ID
        
        Returns:
            Number of patterns deleted
        """
        user_dir = self._get_user_dir(user_id)
        count = 0
        
        for filepath in user_dir.glob("*_gait.pkl"):
            try:
                filepath.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {filepath}: {e}")
        
        # Clear cache
        with self._cache_lock:
            self._cache.pop(user_id, None)
            self._cache_timestamps.pop(user_id, None)
        
        logger.info(f"Cleared {count} patterns for user")
        return count
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_user_gait_count(self, user_id: str) -> int:
        """Get number of enrolled patterns for user"""
        if not user_id:
            return 0
        user_dir = self._get_user_dir(user_id)
        return len(list(user_dir.glob("*_gait.pkl")))
    
    def invalidate_user_cache(self, user_id: str):
        """Force cache invalidation for user"""
        with self._cache_lock:
            self._cache.pop(user_id, None)
            self._cache_timestamps.pop(user_id, None)
        logger.debug(f"Cache invalidated for user: {user_id}")
    
    def clear_cache(self, user_id: str = None):
        """Clear cache (specific user or all)"""
        with self._cache_lock:
            if user_id:
                self._cache.pop(user_id, None)
                self._cache_timestamps.pop(user_id, None)
            else:
                self._cache.clear()
                self._cache_timestamps.clear()
    
    def get_all_enrolled_names(self, user_id: str) -> List[str]:
        """Get list of all enrolled person names"""
        data = self.load_user_gait_data(user_id)
        return list(data.keys())
    
    # =========================================================================
    # HEALTH CHECK
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """System health check"""
        return {
            'base_dir': str(self.base_dir),
            'base_dir_exists': self.base_dir.exists(),
            'cache_size': len(self._cache),
            'expected_feature_dim': DATA_CONFIG.EXPECTED_FEATURE_LEN
        }
