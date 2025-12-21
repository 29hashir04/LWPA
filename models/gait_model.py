"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OPTIMIZED GAIT RECOGNITION MODEL                          ║
║                                                                              ║
║  Production-grade, GPU-accelerated, real-time gait recognition pipeline      ║
║  - YOLOv8n-pose for 17-keypoint detection                                    ║
║  - 136-dimensional normalized gait features                                  ║
║  - Euclidean distance matching with adaptive thresholds                      ║
║  - Temporal smoothing with EMA and voting                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# PYTORCH 2.6+ COMPATIBILITY PATCH
# YOLOv8 models require weights_only=False for proper loading
# =============================================================================
import torch
import torch.nn as nn

_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, 'weights_only': False} if 'weights_only' not in kwargs else kwargs
)

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import logging
import time
import gc
from collections import deque
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Import YOLO after patching torch.load
from ultralytics import YOLO

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CONSTANTS (Centralized for easy tuning)
# =============================================================================

@dataclass(frozen=True)
class GaitConfig:
    """
    Centralized configuration for gait recognition.
    All magic numbers documented and tunable.
    """
    # -------------------------------------------------------------------------
    # POSE DETECTION
    # -------------------------------------------------------------------------
    POSE_MODEL_PATH: str = "yolov8n-pose.pt"  # Lightweight pose model
    CONFIDENCE_THRESHOLD: float = 0.25         # Keypoint detection confidence
    
    # -------------------------------------------------------------------------
    # TEMPORAL BUFFERING
    # -------------------------------------------------------------------------
    POSE_BUFFER_SIZE: int = 60      # Frames to collect (~1.5 gait cycles at 30fps)
    FEATURE_BUFFER_SIZE: int = 7    # Windows for feature averaging
    DECISION_BUFFER_SIZE: int = 7   # Votes for stable decision
    
    # -------------------------------------------------------------------------
    # RECOGNITION THRESHOLDS
    # -------------------------------------------------------------------------
    DISTANCE_THRESHOLD: float = 18  # Max Euclidean distance for match
    MIN_CONFIDENCE: float = 0.30     # Minimum confidence to show name
    MIN_VOTES: int = 3               # Required votes for stable decision
    
    # -------------------------------------------------------------------------
    # FEATURE DIMENSIONS
    # -------------------------------------------------------------------------
    NUM_KEYPOINTS: int = 17          # COCO pose keypoints
    COORDS_PER_KEYPOINT: int = 2     # x, y coordinates
    POSE_DIM: int = 34               # 17 * 2
    FEATURE_DIM: int = 136           # 34 * 4 (mean, std, velocity, range)
    
    # -------------------------------------------------------------------------
    # GPU OPTIMIZATION
    # -------------------------------------------------------------------------
    USE_FP16: bool = True            # Half-precision inference (2x faster)
    WARMUP_FRAMES: int = 3           # Warmup inference calls
    
    # -------------------------------------------------------------------------
    # PERFORMANCE
    # -------------------------------------------------------------------------
    MAX_INFERENCE_TIME_MS: float = 50.0  # Alert if exceeded
    MEMORY_CLEANUP_INTERVAL: int = 100   # Frames between GC


CONFIG = GaitConfig()


class DetectionStatus(Enum):
    """Detection status enumeration for type safety"""
    NO_PERSON = "no_person"
    PARTIAL_BODY = "partial_body"
    FULL_BODY = "full_body"


# =============================================================================
# CIRCULAR BUFFER FOR O(1) OPERATIONS
# =============================================================================

class CircularPoseBuffer:
    """
    Pre-allocated circular buffer for normalized poses.
    Avoids deque-to-array conversions, O(1) append and full check.
    """
    
    def __init__(self, max_size: int, pose_dim: int):
        self.max_size = max_size
        self.pose_dim = pose_dim
        # Pre-allocate numpy array
        self.buffer = np.zeros((max_size, pose_dim), dtype=np.float32)
        self.index = 0
        self.count = 0
    
    def append(self, pose: np.ndarray) -> None:
        """Add pose to buffer (O(1) operation)"""
        self.buffer[self.index] = pose
        self.index = (self.index + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)
    
    def is_full(self) -> bool:
        """Check if buffer has enough frames"""
        return self.count >= self.max_size
    
    def get_sequence(self) -> np.ndarray:
        """Get ordered sequence of poses"""
        if self.count < self.max_size:
            return self.buffer[:self.count].copy()
        # Reorder circular buffer to temporal order
        return np.vstack([
            self.buffer[self.index:],
            self.buffer[:self.index]
        ])
    
    def clear(self) -> None:
        """Reset buffer"""
        self.buffer.fill(0)
        self.index = 0
        self.count = 0


# =============================================================================
# EXPONENTIAL MOVING AVERAGE FOR SMOOTH DECISIONS
# =============================================================================

class EMASmoothing:
    """
    Exponential Moving Average for feature and confidence smoothing.
    Reduces jitter without the lag of simple averaging.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother
        """
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: np.ndarray) -> np.ndarray:
        """Update EMA with new value"""
        if self.value is None:
            self.value = new_value.copy()
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self) -> None:
        """Reset EMA state"""
        self.value = None


# =============================================================================
# MAIN GAIT MODEL CLASS
# =============================================================================

class OptimizedFastPoseGaitModel:
    """
    Production-grade gait recognition model with GPU acceleration.
    
    Pipeline:
    1. YOLO pose detection (GPU accelerated)
    2. Pose normalization (scale/position invariant)
    3. Temporal buffering (45 frames)
    4. Feature extraction (136-dim)
    5. Distance-based recognition
    6. EMA + voting smoothing
    """
    
    def __init__(self):
        # =====================================================================
        # GPU SETUP
        # =====================================================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = CONFIG.USE_FP16 and torch.cuda.is_available()
        
        logger.info(f"Initializing: Device={self.device}, FP16={self.use_fp16}")
        
        # =====================================================================
        # MODEL INITIALIZATION
        # =====================================================================
        self.pose_model = None
        self._load_optimized_models()
        
        # =====================================================================
        # BUFFERS (Pre-allocated for performance)
        # =====================================================================
        self.pose_buffer = CircularPoseBuffer(
            max_size=CONFIG.POSE_BUFFER_SIZE,
            pose_dim=CONFIG.POSE_DIM
        )
        
        # Feature smoothing with EMA
        self.feature_ema = EMASmoothing(alpha=0.4)
        
        # Decision voting buffer
        self.decision_buffer = deque(maxlen=CONFIG.DECISION_BUFFER_SIZE)
        self.confidence_buffer = deque(maxlen=CONFIG.DECISION_BUFFER_SIZE)
        
        # =====================================================================
        # RECOGNITION PARAMETERS
        # =====================================================================
        self.distance_threshold = CONFIG.DISTANCE_THRESHOLD
        
        # =====================================================================
        # PERFORMANCE MONITORING
        # =====================================================================
        self.inference_times = deque(maxlen=100)
        self.frame_count = 0
        self._last_cleanup = 0
        
        # =====================================================================
        # PRE-COMPUTED VALUES FOR ENROLLED DATA
        # =====================================================================
        self._enrolled_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("OptimizedFastPoseGaitModel initialized")
    
    # =========================================================================
    # MODEL LOADING WITH GPU OPTIMIZATION
    # =========================================================================
    
    def _load_optimized_models(self) -> None:
        """
        Load YOLO model with full GPU optimization.
        Includes warmup for stable first-frame performance.
        """
        try:
            logger.info(f"Loading YOLO model: {CONFIG.POSE_MODEL_PATH}")
            
            # Load model
            self.pose_model = YOLO(CONFIG.POSE_MODEL_PATH)
            
            # GPU optimization
            if torch.cuda.is_available():
                self.pose_model.to(self.device)
                
                # Enable cuDNN autotuner for optimized convolutions
                torch.backends.cudnn.benchmark = True  
                torch.backends.cudnn.enabled = True
                
                # Enable TF32 for Ampere+ GPUs (faster matmul)
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Warmup inference (eliminates first-frame lag)
            self._warmup_model()
            
            logger.info("Model loaded and warmed up")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            self.pose_model = None
    
    def _warmup_model(self) -> None:
        """Run warmup inferences to initialize CUDA kernels"""
        if self.pose_model is None:
            return
        
        logger.info(f"Warming up model ({CONFIG.WARMUP_FRAMES} frames)...")
        
        # Create dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for i in range(CONFIG.WARMUP_FRAMES):
            try:
                _ = self.pose_model(dummy_frame, verbose=False)
            except Exception:
                pass
        
        # Clear CUDA cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # =========================================================================
    # POSE NORMALIZATION (Vectorized for speed)
    # =========================================================================
    
    def _normalize_pose(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize pose keypoints for scale and position invariance.
        
        Normalization steps:
        1. Convert to numpy if tensor
        2. Extract x,y coordinates (ignore confidence)
        3. Center at hip midpoint
        4. Scale by torso length
        
        Args:
            keypoints: Raw keypoints from YOLO (17, 2) or (17, 3)
        
        Returns:
            Normalized 34-dim pose vector, or None if invalid
        """
        if keypoints is None:
            return None
        
        try:
            # Handle PyTorch tensors
            if hasattr(keypoints, 'detach'):
                keypoints = keypoints.detach().cpu().numpy()
            
            # Ensure numpy array
            kp = np.asarray(keypoints, dtype=np.float32)
                
            # Extract x, y coordinates
            if kp.ndim == 1:
                if kp.size == 34:
                    coords = kp.reshape(17, 2)
                else:
                    return None
            elif kp.ndim == 2:
                if kp.shape[0] != 17:
                    return None
                coords = kp[:, :2].copy()
            else:
                return None
            
            # -----------------------------------------------------------------
            # VECTORIZED NORMALIZATION
            # -----------------------------------------------------------------
            
            # Hip center (keypoints 11, 12)
            hip_center = (coords[11] + coords[12]) * 0.5
            
            # Translate to hip origin
            coords -= hip_center
            
            # Shoulder center for torso length (keypoints 5, 6)
            shoulder_center = (coords[5] + coords[6]) * 0.5
            torso_length = np.linalg.norm(shoulder_center)
            
            # Scale by torso length (avoid division by zero)
            if torso_length > 1e-6:
                coords *= (1.0 / torso_length)
            
            return coords.flatten().astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Normalization failed: {e}")
            return None
    
    # =========================================================================
    # POSE DETECTION (Optimized batch processing)
    # =========================================================================
    
    def detect_person_status_batch(self, frames: List[np.ndarray]) -> List[Tuple[str, Optional[np.ndarray]]]:
        """
        Batch pose detection for multiple frames.
        
        Args:
            frames: List of BGR frames
        
        Returns:
            List of (status, keypoints) tuples
        """
        if self.pose_model is None or not frames:
            return [(DetectionStatus.NO_PERSON.value, None)] * len(frames)
        
        try:
            start_time = time.perf_counter()
            
            # Run batch inference
            results = self.pose_model(
                frames, 
                verbose=False,
                half=self.use_fp16  # FP16 inference if supported
            )
            
            # Process results
            batch_results = []
            for result in results:
                status, keypoints = self._process_single_result(result)
                batch_results.append((status, keypoints))
            
            # Performance tracking
            inference_time = (time.perf_counter() - start_time) * 1000
            self.inference_times.append(inference_time)
            self.frame_count += len(frames)
            
            # Performance warning
            if inference_time > CONFIG.MAX_INFERENCE_TIME_MS:
                logger.warning(f"Slow inference: {inference_time:.1f}ms")
            
            # Periodic memory cleanup
            self._maybe_cleanup_memory()
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch detection error: {e}")
            return [(DetectionStatus.NO_PERSON.value, None)] * len(frames)
    
    def _process_single_result(self, result) -> Tuple[str, Optional[np.ndarray]]:
        """
        Process single YOLO result with optimized checks.
        
        Returns:
            (status, keypoints) tuple
        """
        try:
            # Quick null checks
            if not hasattr(result, 'keypoints') or result.keypoints is None:
                return DetectionStatus.NO_PERSON.value, None
            
            keypoints = result.keypoints
            
            if not hasattr(keypoints, 'xy') or keypoints.xy is None:
                return DetectionStatus.NO_PERSON.value, None
            
            if len(keypoints.xy) == 0:
                return DetectionStatus.NO_PERSON.value, None
            
            # Get first person's keypoints
            person_kp = keypoints.xy[0]
            
            if person_kp is None or len(person_kp) != 17:
                return DetectionStatus.NO_PERSON.value, None
            
            # -----------------------------------------------------------------
            # BODY PART CONFIDENCE CHECKING
            # -----------------------------------------------------------------
            
            # Get confidence scores if available
            if hasattr(keypoints, 'conf') and keypoints.conf is not None and len(keypoints.conf) > 0:
                conf = keypoints.conf[0]
                if hasattr(conf, 'cpu'):
                    conf = conf.cpu().numpy()
                else:
                    conf = np.asarray(conf)
            
                # Vectorized confidence check
                threshold = CONFIG.CONFIDENCE_THRESHOLD
                
                # Key body parts indices
                head_indices = [0, 1, 2, 3, 4]      # nose, eyes, ears
                torso_indices = [5, 6, 11, 12]      # shoulders, hips
                leg_indices = [13, 14, 15, 16]      # knees, ankles
                
                head_ok = np.max(conf[head_indices]) > threshold
                torso_ok = np.max(conf[torso_indices]) > threshold
                legs_ok = np.max(conf[leg_indices]) > threshold
                
                if head_ok and torso_ok and legs_ok:
                    return DetectionStatus.FULL_BODY.value, person_kp
                elif head_ok or torso_ok:
                    return DetectionStatus.PARTIAL_BODY.value, person_kp
                else:
                    return DetectionStatus.NO_PERSON.value, None
            
            # No confidence scores, assume full body if 17 keypoints exist
            return DetectionStatus.FULL_BODY.value, person_kp
                
        except Exception:
            return DetectionStatus.NO_PERSON.value, None
    
    # =========================================================================
    # FEATURE EXTRACTION (Vectorized, incremental)
    # =========================================================================
    
    def extract_gait_features_optimized(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 136-dimensional gait features from pose sequence.
        
        Features:
        - Mean pose (34): Average joint positions
        - Std pose (34): Movement variation per joint
        - Mean velocity (34): Average movement speed
        - Range of motion (34): Joint flexibility
        
        Args:
            keypoints: Raw keypoints from detection
        
        Returns:
            136-dim feature vector when buffer is full, else None
        """
        try:
            # Normalize pose
            normalized = self._normalize_pose(keypoints)
            
            if normalized is None or len(normalized) != CONFIG.POSE_DIM:
                    return None
            
            # Add to circular buffer
            self.pose_buffer.append(normalized)
            
            # Check if buffer is full
            if not self.pose_buffer.is_full():
                return None
            
            # Get ordered sequence
            sequence = self.pose_buffer.get_sequence()
            
            # -----------------------------------------------------------------
            # VECTORIZED FEATURE EXTRACTION
            # -----------------------------------------------------------------
            
            # 1. Mean pose (average body position over time)
            mean_pose = np.mean(sequence, axis=0)
            
            # 2. Std pose (how much each joint moves)
            std_pose = np.std(sequence, axis=0)
            
            # 3. Mean velocity (frame-to-frame differences)
            velocities = np.diff(sequence, axis=0)
            mean_velocity = np.mean(np.abs(velocities), axis=0)
            
            # 4. Range of motion (max - min for each joint)
            range_motion = np.ptp(sequence, axis=0)  # ptp = peak-to-peak
            
            # Concatenate to 136-dim feature vector
            features = np.concatenate([
                mean_pose,       # 34
                std_pose,        # 34
                mean_velocity,   # 34
                range_motion     # 34
            ]).astype(np.float32)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    @staticmethod
    def compute_features_from_sequence(sequence: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute gait features from pre-collected pose sequence.
        Used by enrollment process.
        
        Args:
            sequence: (N, 34) array of normalized poses
        
        Returns:
            136-dim feature vector
        """
        try:
            seq = np.asarray(sequence, dtype=np.float32)
            
            if seq.ndim != 2 or seq.shape[1] != CONFIG.POSE_DIM or seq.shape[0] < 2:
                return None
            
            mean_pose = np.mean(seq, axis=0)
            std_pose = np.std(seq, axis=0)
            velocities = np.diff(seq, axis=0)
            mean_velocity = np.mean(np.abs(velocities), axis=0)
            range_motion = np.ptp(seq, axis=0)
            
            features = np.concatenate([mean_pose, std_pose, mean_velocity, range_motion])
            
            # Validate features
            if not np.isfinite(features).all() or np.allclose(features, 0):
                return None
            
            return features.astype(np.float32)
            
        except Exception:
            return None
    
    # =========================================================================
    # RECOGNITION (Distance-based with smoothing)
    # =========================================================================
    
    def recognize_person_optimized(
        self, 
        features: np.ndarray, 
        enrolled_data: Dict[str, np.ndarray]
    ) -> Tuple[str, float]:
        """
        Recognize person using Euclidean distance with EMA smoothing.
        
        Pipeline:
        1. Apply EMA smoothing to features
        2. Compute distance to each enrolled pattern
        3. Find best match
        4. Apply threshold
        5. Update decision buffer
        6. Return voted decision
        
        Args:
            features: 136-dim gait features
            enrolled_data: Dict of {name: features}
        
        Returns:
            (name, confidence) tuple
        """
        if not enrolled_data:
            logger.debug("No enrolled data")
            return 'Unknown', 0.0
        
        try:
            # -----------------------------------------------------------------
            # FEATURE SMOOTHING (EMA)
            # -----------------------------------------------------------------
            smoothed_features = self.feature_ema.update(features)
            
            # -----------------------------------------------------------------
            # DISTANCE COMPUTATION
            # -----------------------------------------------------------------
            distances = []
            
            for person_name, enrolled_features in enrolled_data.items():
                # Dimension check
                if len(smoothed_features) != len(enrolled_features):
                    logger.warning(f"Dimension mismatch for {person_name}")
                    continue
                
                # Euclidean distance
                distance = np.linalg.norm(smoothed_features - enrolled_features)
                distances.append((person_name, distance))
            
            if not distances:
                return 'Unknown', 0.0
            
            # -----------------------------------------------------------------
            # FIND BEST MATCH
            # -----------------------------------------------------------------
            best_name, best_distance = min(distances, key=lambda x: x[1])
            
            # Confidence: exponential decay based on distance
            # confidence = exp(-distance / threshold) gives:
            #   - distance=0 -> confidence=1.0
            #   - distance=threshold -> confidence~0.37
            #   - distance=2*threshold -> confidence~0.14
            confidence = float(np.exp(-best_distance / self.distance_threshold))
            
            # -----------------------------------------------------------------
            # THRESHOLD CHECK
            # -----------------------------------------------------------------
            if best_distance <= self.distance_threshold:
                decision = best_name
                logger.info(f"MATCH: {best_name} (dist={best_distance:.2f}, conf={confidence:.0%})")
            else:
                decision = 'Unknown'
                logger.info(f"NO MATCH: dist={best_distance:.2f} > {self.distance_threshold}")
            
            # -----------------------------------------------------------------
            # DECISION SMOOTHING (Voting)
            # -----------------------------------------------------------------
            self.decision_buffer.append(decision)
            self.confidence_buffer.append(confidence)
            
            return decision, confidence
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return 'Unknown', 0.0
    
    def get_stable_decision(self) -> Tuple[str, float, bool]:
        """
        Get stable decision based on voting buffer.
        
        Returns:
            (name, avg_confidence, is_stable) tuple
        """
        if len(self.decision_buffer) < CONFIG.MIN_VOTES:
            return 'Processing...', 0.0, False
        
        # Count votes
        from collections import Counter
        votes = Counter(self.decision_buffer)
        most_common, count = votes.most_common(1)[0]
        
        # Average confidence
        avg_conf = float(np.mean(list(self.confidence_buffer)))
        
        # Check stability criteria
        is_stable = (
            count >= CONFIG.MIN_VOTES and 
            avg_conf >= CONFIG.MIN_CONFIDENCE
        )
        
        if is_stable:
            return most_common, avg_conf, True
        elif most_common == 'Unknown' and count >= CONFIG.MIN_VOTES - 1:
            # Be slightly more lenient for Unknown (security)
            return 'Unknown', avg_conf, True
        else:
            return 'Processing...', avg_conf, False
    
    # =========================================================================
    # BUFFER MANAGEMENT
    # =========================================================================
    
    def reset_buffers(self) -> None:
        """Reset all buffers for fresh recognition"""
        self.pose_buffer.clear()
        self.feature_ema.reset()
        self.decision_buffer.clear()
        self.confidence_buffer.clear()
        logger.info("Buffers reset")
    
    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================
    
    def _maybe_cleanup_memory(self) -> None:
        """Periodic GPU/CPU memory cleanup"""
        if self.frame_count - self._last_cleanup >= CONFIG.MEMORY_CLEANUP_INTERVAL:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self._last_cleanup = self.frame_count
    
    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get real-time performance metrics"""
        if not self.inference_times:
            return {
                'fps': 0.0,
                'avg_inference_ms': 0.0,
                'total_frames': 0,
                'gpu_available': torch.cuda.is_available()
            }
        
        avg_time = np.mean(self.inference_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        return {
            'fps': fps,
            'avg_inference_ms': avg_time,
            'total_frames': self.frame_count,
            'gpu_available': torch.cuda.is_available(),
            'buffer_fill': self.pose_buffer.count / CONFIG.POSE_BUFFER_SIZE
        }
    
    # =========================================================================
    # HEALTH CHECK
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """System health check for production monitoring"""
        return {
            'model_loaded': self.pose_model is not None,
            'device': str(self.device),
            'fp16_enabled': self.use_fp16,
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
            'buffer_size': CONFIG.POSE_BUFFER_SIZE,
            'threshold': self.distance_threshold
        }
