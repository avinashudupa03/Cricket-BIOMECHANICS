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
# Use yolov8n.pt (nano) - YOLO11 models have detection issues in current ultralytics version
# Can be overridden by setting YOLO_MODEL_NAME env var
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.25  # Confidence threshold for person detection
YOLO_PERSON_CLASS = 0  # COCO class 0 = person
YOLO_TRACKER_CONFIG = "bytetrack_custom.yaml"
YOLO_PERSIST = True
YOLO_VERBOSE = False

# ============================================================
# Batsman ROI Configuration
# ============================================================
# Normalized coordinates (0.0 to 1.0) for the batsman region of interest.
# YOLO will operate on this cropped ROI rather than the full frame.
# The ROI defines the area where the batsman is expected to be.
#
# OVERVIEW:
# - These are NORMALIZED coordinates (0.0 to 1.0), NOT pixel coordinates
# - x1, y1 = top-left corner; x2, y2 = bottom-right corner
# - (0, 0) = top-left of frame, (1, 1) = bottom-right of frame
# - The ROI should cover the batsman's typical position at the crease
#
# USAGE:
# 1. Edit the BATSMAN_ROI dictionary below to change ROI for your camera setup
# 2. Or override via command line: --roi x1 y1 x2 y2 (normalized 0-1)
# 3. Generate debug video (--debug-video) to visually verify ROI placement
#
# EXAMPLE CAMERA SETUPS:
# - Standard broadcast view: x1=0.30, y1=0.15, x2=0.70, y2=0.75
# - Wide angle / side view:  x1=0.20, y1=0.10, x2=0.80, y2=0.80
# - Tight / zoomed view:     x1=0.35, y1=0.20, x2=0.65, y2=0.70
# - Behind bowler view:      x1=0.25, y1=0.20, x2=0.75, y2=0.80
#
# TUNING TIPS:
# - If wicketkeeper/bowler detected: shrink ROI horizontally (increase x1, decrease x2)
# - If batsman cut off: expand ROI vertically (decrease y1, increase y2)
# - If no detections: increase ROI_CROP_PAD_X/Y to give YOLO more context

# Centralized ROI configuration dictionary (easy to modify/override)
BATSMAN_ROI = {
    "x1": 0.30,   # Left boundary (normalized)
    "y1": 0.15,   # Top boundary (normalized)
    "x2": 0.70,   # Right boundary (normalized)
    "y2": 0.75,   # Bottom boundary (normalized)
}

# Backwards-compatible individual constants (derived from BATSMAN_ROI)
BATSMAN_ROI_X1 = BATSMAN_ROI["x1"]
BATSMAN_ROI_Y1 = BATSMAN_ROI["y1"]
BATSMAN_ROI_X2 = BATSMAN_ROI["x2"]
BATSMAN_ROI_Y2 = BATSMAN_ROI["y2"]

# ROI crop padding (fraction of frame size to add around ROI)
# This helps catch detections at ROI boundaries and gives YOLO context
ROI_CROP_PAD_X = 0.05   # 5% padding horizontally
ROI_CROP_PAD_Y = 0.05   # 5% padding vertically

# Minimum overlap with ROI required for a detection to be considered a batsman candidate
# (after converting back to full-frame coordinates)
BATSMAN_ROI_MIN_OVERLAP = 0.30  # 30% overlap minimum (IoU or coverage)

# ============================================================
# Batsman Selection Configuration
# ============================================================

# ============================================================
# Batsman Selection Configuration
# ============================================================

# Temporal stability
BATSMAN_LOCK_FRAMES = 15          # Frames to maintain lock before allowing switch
BATSMAN_SWITCH_THRESHOLD = 2.0    # Score ratio required to switch batsman
BATSMAN_LOST_TOLERANCE = 20       # Frames to keep lock after person lost
MIN_LOCK_FRAMES_BEFORE_SWITCH = 5 # Minimum frames locked before any switch allowed

# NEW: Physical-person tracking (track-ID independent)
# Maximum frames to maintain batsman state after YOLO loses detection
BATSMAN_MAX_LOST_FRAMES = 15      # Configurable recovery window for YOLO dropouts

# Continuity scoring weights (for track-ID independent selection)
CONTINUITY_SPATIAL_WEIGHT = 0.30      # Distance from previous batsman center
CONTINUITY_IOU_WEIGHT = 0.25          # IoU with previous bbox
CONTINUITY_SIZE_WEIGHT = 0.20         # Size similarity with previous bbox
CONTINUITY_ROI_POSITION_WEIGHT = 0.15 # Position within batting ROI
CONTINUITY_MOVEMENT_WEIGHT = 0.10     # Movement consistency

# Maximum displacement for continuity (normalized)
MAX_CONTINUITY_DISPLACEMENT = 0.15    # Max center movement per frame (normalized)

# Minimum IoU for spatial continuity
MIN_CONTINUITY_IOU = 0.20

# Size similarity tolerance
MAX_SIZE_RATIO_DIFF = 0.50            # Max allowed size ratio difference

# Evidence weights (must sum to ~1.0 for normalized scoring)
POSITION_WEIGHT = 0.20            # Reduced - less reliance on fixed position
SIZE_WEIGHT = 0.20                # Bounding box size (batsman typically larger/closer)
POSE_WEIGHT = 0.20                # Pose proxy from bbox geometry
TEMPORAL_WEIGHT = 0.15            # Temporal consistency
MOVEMENT_WEIGHT = 0.10            # Movement pattern consistency
CONTINUITY_WEIGHT = 0.15          # NEW: Physical-person continuity score

# Position priors - expanded for different shot types
BATTING_REGION_CENTER = (0.50, 0.45)  # Normalized (x, y) - batsman typically center-right
BATTING_REGION_SIGMA = (0.25, 0.20)   # Increased from (0.15, 0.12) - wider for different shot types

# Body proportions (normalized by bbox height)
MIN_BATSMAN_BBOX_HEIGHT = 0.08      # Reduced - allow smaller batsmen
IDEAL_BATSMAN_BBOX_HEIGHT = 0.35    # Ideal normalized bbox height

# Wicketkeeper/Non-striker rejection zones
WICKETKEEPER_Y_MIN = 0.55           # Wicketkeeper typically behind stumps (y > 0.55)
NON_STRIKER_X_MAX = 0.30            # Non-striker at other end (x < 0.30)
NON_STRIKER_Y_MIN = 0.50            # Non-striker also often crouched

# Confidence thresholds
BATSMAN_CONFIDENCE_THRESHOLD = 0.20  # Reduced - allow lower scores
MIN_SCORE_DIFFERENTIAL = 0.10        # Minimum score difference between top candidates

# Tracking recovery
MAX_RECOVERY_FRAMES = 15             # Keep using last bbox for frames after YOLO loses track

MIN_LOCK_FRAMES_BEFORE_SWITCH = 5    # Minimum frames locked before any switch allowed

# ============================================================
# MediaPipe ROI Configuration (for small bbox handling)
# ============================================================
# Minimum crop size for MediaPipe (pixels)
MIN_MEDIAPIPE_CROP_SIZE = 48        # Reduced from 64 to handle small bboxes

# Crop expansion for small bboxes (fraction of bbox size)
MEDIAPIPE_CROP_EXPANSION_X = 0.25   # Expand crop by 25% horizontally
MEDIAPIPE_CROP_EXPANSION_Y = 0.20   # Expand crop by 20% vertically

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