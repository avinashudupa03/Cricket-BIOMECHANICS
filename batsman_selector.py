"""Batsman Selection Module

Scores YOLO-tracked persons to identify the batsman using multiple evidence sources.
Designed to be robust against wicketkeeper/non-striker confusion.
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
    BATTING_REGION_CENTER, BATTING_REGION_SIGMA,
    MIN_BATSMAN_BBOX_HEIGHT, IDEAL_BATSMAN_BBOX_HEIGHT,
    WICKETKEEPER_Y_MIN, NON_STRIKER_X_MAX, NON_STRIKER_Y_MIN,
    BATSMAN_CONFIDENCE_THRESHOLD, MIN_SCORE_DIFFERENTIAL,
    MAX_RECOVERY_FRAMES, MIN_LOCK_FRAMES_BEFORE_SWITCH,
    WICKETKEEPER_Y_MIN, NON_STRIKER_X_MAX, NON_STRIKER_Y_MIN,
    POSITION_WEIGHT, SIZE_WEIGHT, POSE_WEIGHT, TEMPORAL_WEIGHT, MOVEMENT_WEIGHT,
    BATTING_REGION_CENTER, BATTING_REGION_SIGMA,
    MIN_BATSMAN_BBOX_HEIGHT, IDEAL_BATSMAN_BBOX_HEIGHT,
    BATSMAN_CONFIDENCE_THRESHOLD, MIN_SCORE_DIFFERENTIAL,
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
    
    # Scores
    position_score: float = 0.0
    size_score: float = 0.0
    pose_score: float = 0.0
    temporal_score: float = 0.0
    movement_score: float = 0.0
    total_score: float = 0.0


@dataclass
class BatsmanSelection:
    """Result of batsman selection."""
    selected_track_id: Optional[int]
    confidence: float
    all_scores: Dict[int, float]
    reason: str
    frame: int


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


# ---------------------------------------------------------------------------
# Batsman Selector Class
# ---------------------------------------------------------------------------

class BatsmanSelector:
    """
    Selects the batsman from YOLO-tracked persons using multi-evidence scoring.
    """
    
    def __init__(self):
        self.current_batsman_id: Optional[int] = None
        self.lock_frames: int = 0
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
        
        # Score all active tracks
        scores = self._score_all_tracks()
        
        # Select batsman with hysteresis
        selection = self._select_with_hysteresis(scores, frame_num)
        
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
        """Extract cricket-relevant pose features from MediaPipe landmarks - placeholder for now."""
        # Pose data integration will be added later when integrating with MediaPipe pipeline
        # For now, we rely on YOLO bbox and position features
        pass
    
    def _score_all_tracks(self) -> Dict[int, float]:
        """Score all active tracks."""
        scores = {}
        for track_id, track in self.track_history.items():
            if track.frames_lost > BATSMAN_LOST_TOLERANCE:
                continue  # Skip lost tracks
            
            # Position score: Gaussian prior around batting region
            track.position_score = gaussian_prior(
                track.center, BATTING_REGION_CENTER, BATTING_REGION_SIGMA)
            
            # Size score: prefer taller bboxes (closer/larger person)
            if track.bbox_height >= MIN_BATSMAN_BBOX_HEIGHT:
                track.size_score = clamp(track.bbox_height / IDEAL_BATSMAN_BBOX_HEIGHT)
            else:
                track.size_score = 0.1  # Small penalty for too small
            
            # Pose score: from MediaPipe features
            track.pose_score = self._compute_pose_score(track)
            
            # Temporal score: reward consistent presence
            if track.frames_tracked > 0:
                track.temporal_score = clamp(track.frames_tracked / 30.0)
            else:
                track.temporal_score = 0.0
            
            # Movement score: smooth, consistent movement
            track.movement_score = self._compute_movement_score(track)
            
            # Weighted total
            track.total_score = (
                POSITION_WEIGHT * track.position_score +
                SIZE_WEIGHT * track.size_score +
                POSE_WEIGHT * track.pose_score +
                TEMPORAL_WEIGHT * track.temporal_score +
                MOVEMENT_WEIGHT * track.movement_score
            )
            
            scores[track_id] = track.total_score
        
        return scores
    
    def _compute_pose_score(self, track: PersonTrack) -> float:
        """Compute pose-proxy score from bbox geometry (cricket-specific heuristics).
        
        Until MediaPipe integration, use bbox geometry as pose proxy:
        - Batsman: tall bbox (full body), upright-ish, center-horizontal
        - Wicketkeeper: crouched (wider, shorter), behind stumps (y > 0.55)
        - Bowler: tall, at edge of frame
        - Non-striker: at non-striker's end (x < 0.3), often crouched
        """
        # Base score from bbox geometry
        score = 0.0
        
        # 1. Aspect ratio: batsmen are typically upright (tall, not too wide)
        if track.bbox_height > 0 and track.bbox_width > 0:
            aspect = track.bbox_height / track.bbox_width
            # Batsman: tall and upright (aspect ~1.5-3.0)
            # Wicketkeeper: crouched (aspect ~0.8-1.5)
            if aspect >= 1.8:
                score += 0.3
            elif aspect >= 1.4:
                score += 0.2
            elif aspect >= 1.0:
                score += 0.1
        
        # 2. Vertical position: batsman typically in upper-middle of frame
        if track.center is not None:
            cy = track.center[1]
            # Batsman typically at y=0.3-0.5 (upper-middle)
            if 0.25 <= cy <= 0.55:
                score += 0.25
            elif 0.20 <= cy <= 0.65:
                score += 0.15
            elif 0.15 <= cy <= 0.75:
                score += 0.05
            
            # Wicketkeeper penalty: behind stumps, crouched (y > 0.55, low aspect)
            if cy > WICKETKEEPER_Y_MIN:
                if track.bbox_height > 0 and track.bbox_width > 0:
                    aspect = track.bbox_height / track.bbox_width
                    if aspect < 1.6:  # Crouched posture
                        score -= 0.3  # Strong penalty for wicketkeeper-like pose
        
        # 3. Horizontal position: batsman typically center-right
        if track.center is not None:
            cx = track.center[0]
            # Batsman typically at x=0.4-0.6
            if 0.35 <= cx <= 0.65:
                score += 0.2
            elif 0.25 <= cx <= 0.75:
                score += 0.1
            
            # Non-striker penalty: at other end (x < 0.3), often crouched (y > 0.5)
            if cx < NON_STRIKER_X_MAX and track.center[1] > NON_STRIKER_Y_MIN:
                score -= 0.25  # Penalty for non-striker-like position
        
        # 4. Size: batsman usually larger (closer to camera)
        if track.bbox_height >= MIN_BATSMAN_BBOX_HEIGHT:
            size_norm = clamp(track.bbox_height / IDEAL_BATSMAN_BBOX_HEIGHT)
            score += 0.25 * size_norm
        
        return clamp(score)
    
    def _compute_movement_score(self, track: PersonTrack) -> float:
        """Score based on movement consistency."""
        if len(track.position_history) < 3:
            return 0.5  # Neutral for short history
        
        # Check movement smoothness (low variance in step size)
        steps = []
        for i in range(1, len(track.position_history)):
            p1 = track.position_history[i-1]
            p2 = track.position_history[i]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps.append(dist)
        
        if len(steps) < 2:
            return 0.5
        
        # Low variance = consistent movement
        mean_step = np.mean(steps)
        std_step = np.std(steps)
        cv = std_step / max(mean_step, 1e-6)  # Coefficient of variation
        
        # Lower CV = more consistent = higher score
        return clamp(1.0 - min(cv, 1.0))
    
    def _select_with_hysteresis(self, scores: Dict[int, float], frame_num: int) -> BatsmanSelection:
        """Select batsman with temporal hysteresis to prevent flickering."""
        
        if not scores:
            return BatsmanSelection(
                selected_track_id=None,
                confidence=0.0,
                all_scores={},
                reason="no_active_tracks",
                frame=frame_num
            )
        
        # Sort by score descending
        sorted_tracks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_score = sorted_tracks[0]
        second_score = sorted_tracks[1][1] if len(sorted_tracks) > 1 else 0.0
        
        # Check confidence threshold
        if best_score < BATSMAN_CONFIDENCE_THRESHOLD:
            if self.current_batsman_id is not None:
                # Check if we should release lock
                if self.lock_frames >= BATSMAN_LOCK_FRAMES:
                    self.current_batsman_id = None
                    self.lock_frames = 0
                    return BatsmanSelection(
                        selected_track_id=None,
                        confidence=best_score,
                        all_scores=scores,
                        reason="best_score_below_threshold_lock_released",
                        frame=frame_num
                    )
            return BatsmanSelection(
                selected_track_id=self.current_batsman_id,
                confidence=best_score,
                all_scores=scores,
                reason="below_threshold_maintaining_lock",
                frame=frame_num
            )
        
        # Check if current batsman is still viable
        if self.current_batsman_id is not None:
            current_score = scores.get(self.current_batsman_id, 0.0)
            score_ratio = best_score / max(current_score, 1e-6)
            
            # If current batsman is close to best, keep lock
            if score_ratio < BATSMAN_SWITCH_THRESHOLD:
                self.lock_frames = min(self.lock_frames + 1, BATSMAN_LOCK_FRAMES)
                return BatsmanSelection(
                    selected_track_id=self.current_batsman_id,
                    confidence=current_score,
                    all_scores=scores,
                    reason="maintaining_lock_hysteresis",
                    frame=frame_num
                )
            else:
                # Significant difference - allow switch ONLY if minimum lock duration met
                if self.lock_frames >= MIN_LOCK_FRAMES_BEFORE_SWITCH:
                    self.lock_frames = 0
                else:
                    # Not enough lock time yet, maintain current selection
                    self.lock_frames = min(self.lock_frames + 1, BATSMAN_LOCK_FRAMES)
                    return BatsmanSelection(
                        selected_track_id=self.current_batsman_id,
                        confidence=current_score,
                        all_scores=scores,
                        reason="maintaining_lock_min_duration",
                        frame=frame_num
                    )
        
        # New selection or forced switch
        if best_id != self.current_batsman_id:
            self.lock_frames = 1
        else:
            self.lock_frames = min(self.lock_frames + 1, BATSMAN_LOCK_FRAMES)
        
        self.current_batsman_id = best_id
        
        # Check score differential
        score_diff = best_score - second_score
        if score_diff < MIN_SCORE_DIFFERENTIAL and len(sorted_tracks) > 1:
            reason = "selected_low_differential"
        else:
            reason = "selected_best_score"
        
        return BatsmanSelection(
            selected_track_id=best_id,
            confidence=best_score,
            all_scores=scores,
            reason=reason,
            frame=frame_num
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