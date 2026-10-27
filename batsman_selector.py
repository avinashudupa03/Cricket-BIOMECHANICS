"""Batsman Selection Module

Scores YOLO-tracked persons to identify the batsman using multiple evidence sources.
Designed to be robust against wicketkeeper/non-striker confusion and YOLO track-ID changes.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import centralized YOLO config
from config_yolo import (
    BATSMAN_LOCK_FRAMES, BATSMAN_SWITCH_THRESHOLD, BATSMAN_LOST_TOLERANCE,
    MIN_LOCK_FRAMES_BEFORE_SWITCH,
    POSITION_WEIGHT, SIZE_WEIGHT, POSE_WEIGHT, TEMPORAL_WEIGHT, MOVEMENT_WEIGHT,
    CONTINUITY_WEIGHT,
    BATTING_REGION_CENTER, BATTING_REGION_SIGMA,
    MIN_BATSMAN_BBOX_HEIGHT, IDEAL_BATSMAN_BBOX_HEIGHT,
    WICKETKEEPER_Y_MIN, NON_STRIKER_X_MAX, NON_STRIKER_Y_MIN,
    BATSMAN_CONFIDENCE_THRESHOLD, MIN_SCORE_DIFFERENTIAL,
    MAX_RECOVERY_FRAMES, MIN_LOCK_FRAMES_BEFORE_SWITCH,
    BATSMAN_MAX_LOST_FRAMES,
    CONTINUITY_SPATIAL_WEIGHT, CONTINUITY_IOU_WEIGHT, CONTINUITY_SIZE_WEIGHT,
    CONTINUITY_ROI_POSITION_WEIGHT, CONTINUITY_MOVEMENT_WEIGHT,
    MAX_CONTINUITY_DISPLACEMENT, MIN_CONTINUITY_IOU, MAX_SIZE_RATIO_DIFF,
    BATSMAN_ROI, BATSMAN_ROI_MIN_OVERLAP, ROI_OVERLAP_WEIGHT,
)

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def gaussian_prior(center: Tuple[float, float], 
                   mean: Tuple[float, float], 
                   sigma: Tuple[float, float]) -> float:
    """Gaussian position prior for batting region."""
    if center is None:
        return 0.0
    x, y = center
    mx, my = mean
    sx, sy = sigma
    dx = (x - mx) / max(sx, 1e-6)
    dy = (y - my) / max(sy, 1e-6)
    return math.exp(-0.5 * (dx * dx + dy * dy))


def bbox_iou(box1, box2) -> float:
    """IoU of two bboxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def clamp(value, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, value))


def roi_bbox() -> Tuple[float, float, float, float]:
    """Return the batsman ROI as (x1, y1, x2, y2) in normalized coordinates."""
    return (BATSMAN_ROI["x1"], BATSMAN_ROI["y1"], BATSMAN_ROI["x2"], BATSMAN_ROI["y2"])


def bbox_roi_iou(bbox: Tuple[float, float, float, float], 
                 roi: Tuple[float, float, float, float] = None) -> float:
    """IoU of a bbox with the batsman ROI (both normalized 0-1)."""
    if roi is None:
        roi = roi_bbox()
    return bbox_iou(bbox, roi)


def bbox_roi_coverage(bbox: Tuple[float, float, float, float], 
                      roi: Tuple[float, float, float, float] = None) -> float:
    """Fraction of bbox that overlaps with ROI (intersection / bbox_area)."""
    if roi is None:
        roi = roi_bbox()
    x1 = max(bbox[0], roi[0])
    y1 = max(bbox[1], roi[1])
    x2 = min(bbox[2], roi[2])
    y2 = min(bbox[3], roi[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return inter / bbox_area


def bbox_center_in_roi(bbox: Tuple[float, float, float, float], 
                       roi: Tuple[float, float, float, float] = None) -> bool:
    """Check if bbox center is inside ROI."""
    if roi is None:
        roi = roi_bbox()
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def passes_roi_filter(bbox: Tuple[float, float, float, float], 
                      min_overlap: float = BATSMAN_ROI_MIN_OVERLAP) -> bool:
    """Check if a bbox (normalized) has sufficient overlap with the batsman ROI.
    
    Passes if:
    - Box center is inside ROI, OR
    - IoU with ROI >= min_overlap, OR  
    - Coverage (intersection / box_area) >= min_overlap
    """
    if bbox_center_in_roi(bbox):
        return True
    if bbox_roi_iou(bbox) >= min_overlap:
        return True
    if bbox_roi_coverage(bbox) >= min_overlap:
        return True
    return False


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PersonTrack:
    """Tracked person with all evidence for scoring."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized 0-1)
    confidence: float
    center: Tuple[float, float]  # normalized center
    bbox_height: float
    bbox_width: float
    
    # Pose evidence (from MediaPipe when available)
    pose_landmarks: Optional[List] = None
    pose_visibility: Optional[List[float]] = None
    pose_center: Optional[Tuple[float, float]] = None
    shoulder_width_ratio: float = 0.0
    leg_spread_ratio: float = 0.0
    shoulder_asymmetry: float = 0.0
    is_side_on: bool = False
    
    # Temporal
    frames_tracked: int = 0
    frames_lost: int = 0
    last_seen_frame: int = 0
    position_history: List[Tuple[float, float]] = field(default_factory=list)

    # ROI evidence (normalized 0-1)
    roi_overlap: float = 0.0      # fraction of bbox inside the batsman ROI
    inside_roi: bool = False      # bbox center inside batsman ROI

    # Scores
    position_score: float = 0.0
    size_score: float = 0.0
    pose_score: float = 0.0
    temporal_score: float = 0.0
    movement_score: float = 0.0
    continuity_score: float = 0.0
    total_score: float = 0.0


@dataclass
class BatsmanSelection:
    """Result of batsman selection."""
    selected_track_id: Optional[int]
    confidence: float
    all_scores: Dict[int, float]
    all_continuity_scores: Dict[int, float]
    reason: str
    frame: int
    # Physical-person tracking info
    physical_batsman_id: Optional[int] = None
    track_id_changed: bool = False
    batsman_lost: bool = False
    recovery_active: bool = False


# ---------------------------------------------------------------------------
# Continuity Scoring Functions
# ---------------------------------------------------------------------------

def get_last_known_state(candidate: PersonTrack) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float, float, float]], Optional[float], Optional[float]]:
    """
    Get the last known center, bbox, height, width from candidate.
    Uses current state if available, otherwise falls back to position history.
    """
    center = candidate.center
    bbox = candidate.bbox
    bbox_height = candidate.bbox_height
    bbox_width = candidate.bbox_width
    
    # If current state is None/NaN, use last known from position history
    if (center is None or (hasattr(center, '__len__') and len(center) == 2 and (center[0] != center[0] or center[1] != center[1]))) and candidate.position_history:
        # Find last valid center in history
        for p in reversed(candidate.position_history):
            if p is not None and p[0] == p[0] and p[1] == p[1]:  # not NaN
                center = p
                break
    
    if (bbox is None or (hasattr(bbox, '__len__') and len(bbox) == 4 and any(b != b for b in bbox))) and len(candidate.position_history) >= 2:
        # Reconstruct bbox from last two positions (approximate)
        last_center = center
        if last_center is not None:
            # Use average bbox size from history if available
            pass  # Keep bbox as None if we can't reconstruct
    
    return center, bbox, bbox_height, bbox_width


def compute_continuity_score(candidate: PersonTrack, 
                             prev_state: Optional[Dict]) -> float:
    """
    Compute physical-person continuity score independent of YOLO track ID.
    
    Args:
        candidate: Current candidate track
        prev_state: Previous batsman physical state (center, bbox, size, etc.)
        
    Returns:
        Continuity score [0, 1] - higher = more likely same physical person
    """
    if prev_state is None:
        return 0.5  # Neutral if no previous state
    
    # Use last known state for continuity calculation
    center, bbox, bbox_height, bbox_width = get_last_known_state(candidate)
    
    score = 0.0
    
    # 1. Spatial continuity: distance from previous center
    prev_center = prev_state.get('center')
    if prev_center is not None and center is not None:
        dist = math.hypot(center[0] - prev_center[0], 
                         center[1] - prev_center[1])
        # Gaussian decay with max displacement
        spatial_score = math.exp(-0.5 * (dist / MAX_CONTINUITY_DISPLACEMENT) ** 2)
        score += CONTINUITY_SPATIAL_WEIGHT * spatial_score
    
    # 2. IoU continuity: overlap with previous bbox
    prev_bbox = prev_state.get('bbox')
    if prev_bbox is not None and bbox is not None:
        iou = bbox_iou(bbox, prev_bbox)
        iou_score = clamp(iou / MIN_CONTINUITY_IOU) if iou > 0 else 0.0
        score += CONTINUITY_IOU_WEIGHT * iou_score
    
    # 3. Size continuity: similarity in bbox height/width
    if prev_state.get('bbox_height') is not None and bbox_height is not None and bbox_height > 0:
        size_ratio = bbox_height / max(prev_state['bbox_height'], 1e-6)
        if 1.0 - MAX_SIZE_RATIO_DIFF <= size_ratio <= 1.0 + MAX_SIZE_RATIO_DIFF:
            size_score = 1.0 - abs(1.0 - size_ratio) / MAX_SIZE_RATIO_DIFF
        else:
            size_score = 0.0
        score += CONTINUITY_SIZE_WEIGHT * size_score
    
    # 4. ROI position: candidate's position within batting ROI
    roi = roi_bbox()
    if center is not None:
        # Distance from ROI center
        roi_cx = (roi[0] + roi[2]) / 2
        roi_cy = (roi[1] + roi[3]) / 2
        dist_from_roi_center = math.hypot(center[0] - roi_cx, 
                                          center[1] - roi_cy)
        roi_radius = math.hypot(roi[2] - roi[0], roi[3] - roi[1]) / 2
        roi_score = max(0.0, 1.0 - dist_from_roi_center / roi_radius)
        score += CONTINUITY_ROI_POSITION_WEIGHT * roi_score
    
    # 5. Movement consistency (from position history)
    if len(candidate.position_history) >= 3:
        steps = []
        for i in range(1, len(candidate.position_history)):
            p1 = candidate.position_history[i-1]
            p2 = candidate.position_history[i]
            # Skip NaN positions
            if p1 is None or p2 is None:
                continue
            if p1[0] != p1[0] or p1[1] != p1[1] or p2[0] != p2[0] or p2[1] != p2[1]:
                continue
            steps.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if len(steps) >= 2:
            mean_step = np.mean(steps)
            std_step = np.std(steps)
            cv = std_step / max(mean_step, 1e-6)
            movement_score = clamp(1.0 - min(cv, 1.0))
            score += CONTINUITY_MOVEMENT_WEIGHT * movement_score
    
    return clamp(score)


# ---------------------------------------------------------------------------
# Batsman Selector Class
# ---------------------------------------------------------------------------

class BatsmanSelector:
    """
    Selects the batsman from YOLO-tracked persons using multi-evidence scoring.
    
    Features:
    - Track-ID independent physical-person tracking
    - Spatial/temporal continuity across YOLO track-ID changes
    - Configurable recovery window for YOLO dropouts
    - Temporal hysteresis with continuity-aware switching
    """
    
    def __init__(self):
        # YOLO track ID currently selected
        self.current_batsman_id: Optional[int] = None
        self.lock_frames: int = 0
        
        # First-frame selection state
        self._first_frame_processed: bool = False
        
        # Physical-person state (independent of YOLO track ID)
        self.physical_batsman_id: int = 0  # Incrementing ID for physical person
        self.prev_batsman_state: Optional[Dict] = None
        self.batsman_lost_frames: int = 0
        self.batsman_recovery_active: bool = False
        
        # Track history
        self.track_history: Dict[int, PersonTrack] = {}
        self.selection_history: List[BatsmanSelection] = []
        
    def update(self, 
               tracks: Dict[int, PersonTrack], 
               frame_num: int,
               mediapipe_poses: Optional[Dict[int, any]] = None) -> BatsmanSelection:
        """
        Update tracks with new frame data and select batsman.
        
        Args:
            tracks: Dict of track_id -> PersonTrack (from YOLO tracking)
            frame_num: Current frame number
            mediapipe_poses: Optional dict of track_id -> MediaPipe pose data
            
        Returns:
            BatsmanSelection with selected track_id and metadata
        """
        # Update track history with new frame data
        self._update_tracks(tracks, frame_num, mediapipe_poses)
        
        # Score all active tracks with continuity
        scores, continuity_scores = self._score_all_tracks()
        
        # Select batsman with continuity-aware hysteresis
        selection = self._select_with_continuity(scores, continuity_scores, frame_num)
        
        self.selection_history.append(selection)
        return selection
    
    def _update_tracks(self, tracks, frame_num, mediapipe_poses):
        """Update track history with new detections."""
        active_ids = set(tracks.keys())
        
        # Update existing tracks
        for track_id, track in tracks.items():
            if track_id in self.track_history:
                hist = self.track_history[track_id]
                # Update with new detection
                hist.bbox = track.bbox
                hist.confidence = track.confidence
                hist.center = track.center
                hist.bbox_height = track.bbox_height
                hist.bbox_width = track.bbox_width
                hist.frames_lost = 0
                hist.frames_tracked += 1
                hist.last_seen_frame = frame_num
                hist.position_history.append(track.center)
                if len(hist.position_history) > 30:
                    hist.position_history.pop(0)
            else:
                # New track
                self.track_history[track_id] = track
                track.frames_tracked = 1
                track.last_seen_frame = frame_num
                track.position_history = [track.center]
        
        # Increment lost frames for tracks not in current frame
        for track_id in list(self.track_history.keys()):
            if track_id not in active_ids:
                self.track_history[track_id].frames_lost += 1
        
        # Update pose data if available
        if mediapipe_poses:
            for track_id, pose_data in mediapipe_poses.items():
                if track_id in self.track_history:
                    self._extract_pose_features(self.track_history[track_id], pose_data)
    
    def _extract_pose_features(self, track: PersonTrack, pose_data):
        """Extract cricket-relevant pose features from MediaPipe landmarks - placeholder."""
        pass
    
    def _score_all_tracks(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Score all active tracks.

        ROI overlap is the DOMINANT factor: any detection whose center is inside
        the configured batsman ROI outranks any detection outside the ROI, so a
        high-confidence non-striker can never override a valid batsman inside
        the ROI.

        On the first usable frame, only detections inside the batsman ROI are
        considered (unless no ROI candidate exists), per the required behaviour.
        """
        scores = {}
        continuity_scores = {}

        first_frame = (not self._first_frame_processed
                       and self.current_batsman_id is None
                       and len(self.selection_history) == 0)

        for track_id, track in self.track_history.items():
            # Score tracks that are currently detected OR within lost tolerance
            # This allows continuity-based tracking during brief YOLO dropouts
            if track.frames_lost > BATSMAN_LOST_TOLERANCE:
                continue

            is_currently_detected = (track.frames_lost == 0)

            # --- ROI evidence (computed for currently detected tracks) ---
            roi_overlap = 0.0
            inside_roi = False
            if is_currently_detected and track.center is not None:
                roi_overlap = bbox_roi_coverage(track.bbox)  # fraction of bbox inside ROI
                inside_roi = bbox_center_in_roi(track.bbox)
                track.roi_overlap = roi_overlap
                track.inside_roi = inside_roi

                # First frame: ignore detections outside the batsman ROI unless
                # no valid ROI candidate exists at all.
                if first_frame and not inside_roi:
                    # Mark as ineligible for first-frame selection
                    track.total_score = -1.0
                    scores[track_id] = -1.0
                    continuity_scores[track_id] = 0.0
                    continue

            if is_currently_detected:
                # Full scoring for currently detected tracks
                # Position score: Gaussian prior around batting region
                track.position_score = gaussian_prior(
                    track.center, BATTING_REGION_CENTER, BATTING_REGION_SIGMA)

                # Size score: prefer taller bboxes (closer/larger person)
                if track.bbox_height >= MIN_BATSMAN_BBOX_HEIGHT:
                    track.size_score = clamp(track.bbox_height / IDEAL_BATSMAN_BBOX_HEIGHT)
                else:
                    track.size_score = 0.1  # Small penalty for too small

                # Pose score: from MediaPipe features (bbox geometry proxy)
                track.pose_score = self._compute_pose_score(track)

                # Temporal score: reward consistent presence
                if track.frames_tracked > 0:
                    track.temporal_score = clamp(track.frames_tracked / 30.0)
                else:
                    track.temporal_score = 0.0

                # Movement score: smooth, consistent movement
                track.movement_score = self._compute_movement_score(track)
            else:
                # Track is temporarily lost - use last known state for continuity only
                # Position/size/pose/temporal scores are not updated (use last values or 0)
                track.position_score = track.position_score if track.position_score > 0 else 0.0
                track.size_score = track.size_score if track.size_score > 0 else 0.0
                track.pose_score = track.pose_score if track.pose_score > 0 else 0.0
                track.temporal_score = track.temporal_score if track.temporal_score > 0 else 0.0
                track.movement_score = track.movement_score if track.movement_score > 0 else 0.0

            # Continuity score: physical-person tracking (track-ID independent)
            # This works for both detected and recently lost tracks
            track.continuity_score = compute_continuity_score(track, self.prev_batsman_state)

            # Weighted total (evidence + continuity). ROI overlap has the
            # strongest influence so the batsman ROI location always dominates.
            track.total_score = (
                ROI_OVERLAP_WEIGHT * track.roi_overlap +
                POSITION_WEIGHT * track.position_score +
                SIZE_WEIGHT * track.size_score +
                POSE_WEIGHT * track.pose_score +
                TEMPORAL_WEIGHT * track.temporal_score +
                MOVEMENT_WEIGHT * track.movement_score +
                CONTINUITY_WEIGHT * track.continuity_score
            )

            scores[track_id] = track.total_score
            continuity_scores[track_id] = track.continuity_score

        if first_frame:
            self._first_frame_processed = True

        return scores, continuity_scores
    
    def _compute_pose_score(self, track: PersonTrack) -> float:
        """Compute pose-proxy score from bbox geometry (cricket-specific heuristics).
        
        CRITICAL: This runs on tracks ALREADY filtered by ROI, but we still need
        strong rejection of any non-striker/wicketkeeper that might slip through.
        """
        score = 0.0
        
        if track.center is None:
            return 0.0
        
        cx, cy = track.center
        
        # HARD REJECTION ZONES - return very low score immediately
        # Wicketkeeper zone (behind stumps, lower portion)
        if cy > WICKETKEEPER_Y_MIN:
            return -1.0  # Hard reject
        
        # Non-striker zone (foreground, left side, lower portion)
        if cx < NON_STRIKER_X_MAX and cy > NON_STRIKER_Y_MIN:
            return -1.0  # Hard reject
        
        # Bowler zone (top of frame, running in)
        if cy < 0.15:
            return -0.5  # Soft reject
        
        # 1. Aspect ratio: batsmen at far end are typically upright but smaller
        if track.bbox_height > 0 and track.bbox_width > 0:
            aspect = track.bbox_height / track.bbox_width
            if aspect >= 1.5:
                score += 0.3
            elif aspect >= 1.2:
                score += 0.2
            elif aspect >= 1.0:
                score += 0.1
        
        # 2. Vertical position: batsman at far end is in UPPER portion (y ~ 0.2-0.5)
        if 0.15 <= cy <= 0.50:
            score += 0.3
        elif 0.10 <= cy <= 0.55:
            score += 0.2
        elif 0.05 <= cy <= 0.60:
            score += 0.1
        else:
            score -= 0.2  # Penalty for being too low (non-striker zone)
        
        # 3. Horizontal position: batsman typically center (0.3-0.65)
        if 0.30 <= cx <= 0.65:
            score += 0.25
        elif 0.25 <= cx <= 0.70:
            score += 0.15
        elif 0.20 <= cx <= 0.75:
            score += 0.05
        else:
            score -= 0.15  # Penalty for being too far left/right
        
        # 4. Size: batsman at far end is SMALLER (further from camera)
        # Don't reward large size - that's the non-striker
        if track.bbox_height >= MIN_BATSMAN_BBOX_HEIGHT:
            # Normalize to ideal height for far-end batsman
            size_norm = clamp(track.bbox_height / IDEAL_BATSMAN_BBOX_HEIGHT)
            score += 0.2 * size_norm
        else:
            score -= 0.1  # Too small
        
        # 5. Position prior: gaussian around batting region center
        pos_prior = gaussian_prior(track.center, BATTING_REGION_CENTER, BATTING_REGION_SIGMA)
        score += 0.25 * pos_prior
        
        return clamp(score)
    
    def _compute_movement_score(self, track: PersonTrack) -> float:
        """Score based on movement consistency."""
        if len(track.position_history) < 3:
            return 0.5
        
        steps = []
        for i in range(1, len(track.position_history)):
            p1 = track.position_history[i-1]
            p2 = track.position_history[i]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps.append(dist)
        
        if len(steps) < 2:
            return 0.5
        
        mean_step = np.mean(steps)
        std_step = np.std(steps)
        cv = std_step / max(mean_step, 1e-6)
        return clamp(1.0 - min(cv, 1.0))
    
    def _select_with_continuity(self, 
                                scores: Dict[int, float], 
                                continuity_scores: Dict[int, float],
                                frame_num: int) -> BatsmanSelection:
        """Select batsman with continuity-aware hysteresis."""
        
        if not scores:
            # No active tracks - check if we should maintain physical batsman state
            self.batsman_lost_frames += 1
            self.batsman_recovery_active = (self.batsman_lost_frames <= BATSMAN_MAX_LOST_FRAMES)
            
            return BatsmanSelection(
                selected_track_id=None,
                confidence=0.0,
                all_scores={},
                all_continuity_scores={},
                reason="no_active_tracks",
                frame=frame_num,
                physical_batsman_id=self.physical_batsman_id if self.batsman_recovery_active else None,
                track_id_changed=False,
                batsman_lost=not self.batsman_recovery_active,
                recovery_active=self.batsman_recovery_active
            )
        
        # Reset lost frames counter
        self.batsman_lost_frames = 0
        self.batsman_recovery_active = False
        
        # Sort by score descending
        sorted_tracks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_score = sorted_tracks[0]
        second_score = sorted_tracks[1][1] if len(sorted_tracks) > 1 else 0.0
        best_continuity = continuity_scores.get(best_id, 0.0)
        
        # Safety guard: when current track ID is lost, try to find continuity match
        if self.current_batsman_id is not None and self.current_batsman_id not in scores:
            self.batsman_lost_frames += 1
            self.batsman_recovery_active = True
            
            # Try to find the same physical person via continuity scoring
            best_continuity_match = None
            best_continuity_score = 0.0
            
            for track_id, cont_score in continuity_scores.items():
                if cont_score > best_continuity_score and cont_score >= 0.5:
                    best_continuity_match = track_id
                    best_continuity_score = cont_score
            
            # If we found a high-continuity match, select it as the same physical person
            if best_continuity_match is not None:
                match_score = scores.get(best_continuity_match, 0.0)
                track_id_changed = True
                # Same physical person, just track ID changed
                return BatsmanSelection(
                    selected_track_id=best_continuity_match,
                    confidence=scores.get(best_continuity_match, 0.0),
                    all_scores=scores,
                    all_continuity_scores=continuity_scores,
                    reason="track_id_lost_continuity_recovery",
                    frame=frame_num,
                    physical_batsman_id=self.physical_batsman_id,
                    track_id_changed=True,
                    batsman_lost=False,
                    recovery_active=True
                )
            
            # No good continuity match - check if we should wait for recovery
            if self.batsman_lost_frames <= BATSMAN_MAX_LOST_FRAMES:
                # Still in recovery window - maintain physical person state but no selection
                return BatsmanSelection(
                    selected_track_id=None,
                    confidence=0.0,
                    all_scores=scores,
                    all_continuity_scores=continuity_scores,
                    reason="track_id_lost_awaiting_recovery",
                    frame=frame_num,
                    physical_batsman_id=self.physical_batsman_id,
                    track_id_changed=False,
                    batsman_lost=False,
                    recovery_active=True
                )
            
            # Recovery window exceeded - lost for good
            return BatsmanSelection(
                selected_track_id=None,
                confidence=0.0,
                all_scores=scores,
                all_continuity_scores=continuity_scores,
                reason="locked_batsman_track_missing_recovery_exceeded",
                frame=frame_num,
                physical_batsman_id=self.physical_batsman_id,
                track_id_changed=False,
                batsman_lost=True,
                recovery_active=False
            )

        # Check confidence threshold
        if best_score < BATSMAN_CONFIDENCE_THRESHOLD:
            if self.current_batsman_id is not None:
                # Check if current track is still being detected
                current_still_detected = self.current_batsman_id in scores
                
                if current_still_detected:
                    # Current track still detected but score low - maintain lock with hysteresis
                    if self.lock_frames >= BATSMAN_LOCK_FRAMES:
                        self.current_batsman_id = None
                        self.lock_frames = 0
                        self.batsman_recovery_active = True
                        self.batsman_lost_frames = 0
                        return BatsmanSelection(
                            selected_track_id=None,
                            confidence=best_score,
                            all_scores=scores,
                            all_continuity_scores=continuity_scores,
                            reason="best_score_below_threshold_lock_released",
                            frame=frame_num,
                            physical_batsman_id=self.physical_batsman_id,
                            track_id_changed=False,
                            batsman_lost=False,
                            recovery_active=True
                        )
                    return BatsmanSelection(
                        selected_track_id=self.current_batsman_id,
                        confidence=best_score,
                        all_scores=scores,
                        all_continuity_scores=continuity_scores,
                        reason="below_threshold_maintaining_lock",
                        frame=frame_num,
                        physical_batsman_id=self.physical_batsman_id,
                        track_id_changed=False,
                        batsman_lost=False,
                        recovery_active=self.batsman_recovery_active
                    )
                else:
                    # Current track LOST - don't maintain lock on ghost track
                    # Allow selection of best available track (even if below threshold)
                    # or enter recovery mode if all scores are very low
                    self.current_batsman_id = None
                    self.lock_frames = 0
                    self.batsman_recovery_active = True
                    self.batsman_lost_frames = 0
                    
                    # If we have any tracks, select the best one (even below threshold)
                    # but mark as recovery mode
                    if scores:
                        return BatsmanSelection(
                            selected_track_id=best_id,
                            confidence=best_score,
                            all_scores=scores,
                            all_continuity_scores=continuity_scores,
                            reason="track_lost_recovering_best_available",
                            frame=frame_num,
                            physical_batsman_id=self.physical_batsman_id,
                            track_id_changed=True,
                            batsman_lost=False,
                            recovery_active=True
                        )
                    else:
                        return BatsmanSelection(
                            selected_track_id=None,
                            confidence=best_score,
                            all_scores=scores,
                            all_continuity_scores=continuity_scores,
                            reason="track_lost_no_candidates",
                            frame=frame_num,
                            physical_batsman_id=self.physical_batsman_id,
                            track_id_changed=False,
                            batsman_lost=True,
                            recovery_active=True
                        )
            
            return BatsmanSelection(
                selected_track_id=None,
                confidence=best_score,
                all_scores=scores,
                all_continuity_scores=continuity_scores,
                reason="below_threshold_no_current_track",
                frame=frame_num,
                physical_batsman_id=None,
                track_id_changed=False,
                batsman_lost=True,
                recovery_active=False
            )
        
        # Check if current batsman is still viable
        track_id_changed = False
        if self.current_batsman_id is not None:
            current_score = scores.get(self.current_batsman_id, 0.0)
            current_continuity = continuity_scores.get(self.current_batsman_id, 0.0)
            score_ratio = best_score / max(current_score, 1e-6)
            
            # Also consider continuity of current batsman
            current_is_viable = (score_ratio < BATSMAN_SWITCH_THRESHOLD or 
                                current_continuity > 0.6)  # High continuity overrides score ratio
            
            if current_is_viable:
                self.lock_frames = min(self.lock_frames + 1, BATSMAN_LOCK_FRAMES)
                return BatsmanSelection(
                    selected_track_id=self.current_batsman_id,
                    confidence=current_score,
                    all_scores=scores,
                    all_continuity_scores=continuity_scores,
                    reason="maintaining_lock_hysteresis",
                    frame=frame_num,
                    physical_batsman_id=self.physical_batsman_id,
                    track_id_changed=False,
                    batsman_lost=False,
                    recovery_active=False
                )
            else:
                # Significant difference - allow switch ONLY if minimum lock duration met
                if self.lock_frames >= MIN_LOCK_FRAMES_BEFORE_SWITCH:
                    self.lock_frames = 0
                    track_id_changed = (best_id != self.current_batsman_id)
                else:
                    self.lock_frames = min(self.lock_frames + 1, BATSMAN_LOCK_FRAMES)
                    return BatsmanSelection(
                        selected_track_id=self.current_batsman_id,
                        confidence=current_score,
                        all_scores=scores,
                        all_continuity_scores=continuity_scores,
                        reason="maintaining_lock_min_duration",
                        frame=frame_num,
                        physical_batsman_id=self.physical_batsman_id,
                        track_id_changed=False,
                        batsman_lost=False,
                        recovery_active=False
                    )
        
        # New selection or forced switch
        if best_id != self.current_batsman_id:
            self.lock_frames = 1
            track_id_changed = True
            
            # Check if this is a physical person continuation (high continuity)
            if self.prev_batsman_state is not None:
                best_continuity = continuity_scores.get(best_id, 0.0)
                if best_continuity > 0.75:
                    # Same physical person, different track ID - don't increment physical ID
                    track_id_changed = False  # It's a track ID change, not person change
                else:
                    # New physical person
                    self.physical_batsman_id += 1
            else:
                # First selection - don't count as a switch
                self.physical_batsman_id = 1
                track_id_changed = False
        else:
            self.lock_frames = min(self.lock_frames + 1, BATSMAN_LOCK_FRAMES)
        
        self.current_batsman_id = best_id
        
        # Update physical batsman state
        best_track = self.track_history.get(best_id)
        if best_track:
            self.prev_batsman_state = {
                'center': best_track.center,
                'bbox': best_track.bbox,
                'bbox_height': best_track.bbox_height,
                'bbox_width': best_track.bbox_width,
                'frame': frame_num,
            }
        
        # Check score differential
        score_diff = best_score - second_score
        if score_diff < MIN_SCORE_DIFFERENTIAL and len(sorted_tracks) > 1:
            reason = "selected_low_differential"
        else:
            reason = "selected_best_score"
        
        if track_id_changed:
            if self.prev_batsman_state and continuity_scores.get(best_id, 0.0) > 0.75:
                reason = "track_id_changed_same_batsman"
            else:
                reason = "physical_person_switched"
        
        return BatsmanSelection(
            selected_track_id=best_id,
            confidence=best_score,
            all_scores=scores,
            all_continuity_scores=continuity_scores,
            reason=reason,
            frame=frame_num,
            physical_batsman_id=self.physical_batsman_id,
            track_id_changed=track_id_changed,
            batsman_lost=False,
            recovery_active=False
        )


# ---------------------------------------------------------------------------
# Integration Helper
# ---------------------------------------------------------------------------

def create_person_track_from_yolo(track_id: int, box, conf: float, 
                                  frame_shape: Tuple[int, int]) -> PersonTrack:
    """Create PersonTrack from YOLO detection box."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    
    # Normalize to 0-1
    nx1, ny1 = x1 / w, y1 / h
    nx2, ny2 = x2 / w, y2 / h
    cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
    bh, bw = ny2 - ny1, nx2 - nx1
    
    return PersonTrack(
        track_id=track_id,
        bbox=(nx1, ny1, nx2, ny2),
        confidence=conf,
        center=(cx, cy),
        bbox_height=bh,
        bbox_width=bw
    )