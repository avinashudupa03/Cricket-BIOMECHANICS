"""Centralized YOLO + MediaPipe configuration.

All YOLO/MP integration parameters in one place for easy tuning.
"""

from pathlib import Path

# ============================================================
# Base paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
INPUT_VIDEOS_DIR = BASE_DIR / "input_videos"
OUTPUT_DATA_DIR = BASE_DIR / "output_data"
OUTPUT_VIDEOS_DIR = BASE_DIR / "output_videos"
REPORTS_DIR = BASE_DIR / "reports"

# ============================================================
# MediaPipe ROI Configuration
# ============================================================
# When running on batsman crops, we only need 1 pose (the batsman)
ROI_NUM_POSES = 1
ROI_MIN_POSE_DETECTION_CONFIDENCE = 0.3
ROI_MIN_POSE_PRESENCE_CONFIDENCE = 0.3
ROI_MIN_TRACKING_CONFIDENCE = 0.3

# ROI padding (fraction of bbox size to add around the batsman)
ROI_PADDING_X = 0.15  # 15% padding horizontally
ROI_PADDING_Y = 0.10  # 10% padding vertically

# Minimum ROI size to run MediaPipe
MIN_ROI_SIZE = 64

# ============================================================
# YOLO Configuration
# ============================================================
# Use yolo11s.pt (small) for better small object detection vs yolo11n.pt (nano)
# Can be overridden by setting YOLO_MODEL_NAME env var
YOLO_MODEL_NAME = "yolo11s.pt"
YOLO_CONF_THRESHOLD = 0.20  # Reduced from 0.25 for better small object detection
YOLO_PERSON_CLASS = 0  # COCO class 0 = person
YOLO_TRACKER_CONFIG = "bytetrack_custom.yaml"
YOLO_PERSIST = True
YOLO_VERBOSE = False

# ============================================================
# Batsman Selection Configuration
# ============================================================

# Temporal stability
BATSMAN_LOCK_FRAMES = 15          # Frames to maintain lock before allowing switch
BATSMAN_SWITCH_THRESHOLD = 2.0    # Score ratio required to switch batsman
BATSMAN_LOST_TOLERANCE = 20       # Frames to keep lock after person lost
MIN_LOCK_FRAMES_BEFORE_SWITCH = 5 # Minimum frames locked before any switch allowed

# Evidence weights (must sum to ~1.0 for normalized scoring)
# Reduced position weight, increased pose and movement weights for better discrimination
POSITION_WEIGHT = 0.25            # Reduced from 0.35 - less reliance on fixed position
SIZE_WEIGHT = 0.25                # Bounding box size (batsman typically larger/closer)
POSE_WEIGHT = 0.25                # Increased from 0.20 - pose proxy from bbox geometry
TEMPORAL_WEIGHT = 0.15            # Temporal consistency
MOVEMENT_WEIGHT = 0.10            # Increased from 0.05 - movement pattern consistency

# Position priors - expanded for different shot types
BATTING_REGION_CENTER = (0.50, 0.45)  # Normalized (x, y) - batsman typically center-right
BATTING_REGION_SIGMA = (0.25, 0.20)   # Increased from (0.15, 0.12) - wider for different shot types

# Body proportions (normalized by bbox height)
MIN_BATSMAN_BBOX_HEIGHT = 0.10      # Reduced from 0.15 - allow smaller batsmen (lofted shots)
IDEAL_BATSMAN_BBOX_HEIGHT = 0.35    # Ideal normalized bbox height

# Wicketkeeper/Non-striker rejection zones
WICKETKEEPER_Y_MIN = 0.55           # Wicketkeeper typically behind stumps (y > 0.55)
NON_STRIKER_X_MAX = 0.30            # Non-striker at other end (x < 0.30)
NON_STRIKER_Y_MIN = 0.50            # Non-striker also often crouched

# Confidence thresholds
BATSMAN_CONFIDENCE_THRESHOLD = 0.25  # Reduced from 0.35 - allow lower scores for lofted shots
MIN_SCORE_DIFFERENTIAL = 0.15        # Minimum score difference between top candidates

# Tracking recovery
MAX_RECOVERY_FRAMES = 15             # Increased from 10 - keep using last bbox for more frames after YOLO loses track

MIN_LOCK_FRAMES_BEFORE_SWITCH = 5    # Minimum frames locked before any switch allowed

# ============================================================
# Debug Configuration
# ============================================================
DEBUG_MODE = False
DEBUG_VERBOSE = False
DEBUG_SAVE_ROI = False

# ============================================================
# MediaPipe (Original Pipeline) Configuration
# ============================================================
NUM_POSES = 4
MIN_POSE_DETECTION_CONFIDENCE = 0.25
MIN_POSE_PRESENCE_CONFIDENCE = 0.25
MIN_TRACKING_CONFIDENCE = 0.25

MAX_GAP_FRAMES = 15
MAX_MERGE_GAP_FRAMES = 30
POS_GATE = 1.8
ACCEPT_SCORE = 2.2
FULL_BODY_HEIGHT = 0.12
MP_BATSMAN_CONFIDENCE_THRESHOLD = 0.45
BATSMAN_LOCK_GATE = 1.3

# ============================================================
# Frame Processing
# ============================================================
FAST_MODE = False  # Skip interpolation/smoothing for speed

# ============================================================
# Path Helpers
# ============================================================
def get_model_path(model_name: str = "pose_landmarker_heavy.task") -> Path:
    """Get path to MediaPipe model file."""
    return MODELS_DIR / model_name

def get_yolo_model_path(model_name: str = YOLO_MODEL_NAME) -> Path:
    """Get path to YOLO model file."""
    return Path(__file__).parent / model_name