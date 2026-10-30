"""YOLO-guided pose extraction with batsman selection.

This module replaces the MediaPipe-only detection/tracking/selection pipeline
with a YOLO-guided approach:

  Video
    → Crop to Batsman ROI
    → YOLO person detection + tracking (ByteTrack) on ROI crop
    → Convert coordinates back to full-frame
    → Filter by ROI overlap (reject wicketkeeper, bowler, umpire, non-striker)
    → Batsman ID selection (using multi-evidence scoring)
    → Crop selected batsman with configurable padding
    → MediaPipe Pose on batsman ROI only
    → Convert ROI landmarks back to full-frame coordinates
    → Existing landmark/angle/phase pipeline

This avoids identity confusion by using YOLO's person tracking on the batting ROI
+ cricket-specific batsman scoring, and only runs MediaPipe on the selected batsman's ROI.

Output format is identical to the original PoseExtractor.analyze() for
drop-in compatibility with the rest of the pipeline.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np

import video_preprocess

# Import existing utilities from pose_extractor
from pose_extractor import (
    LM, Candidate, Trajectory, _isfinite, _finite,
    get_center, get_bbox, bbox_size, get_shape, shape_distance,
    central_prior, _shoulder_width_ratio, _leg_spread_ratio,
    _is_batsman_stance,
    NUM_POSES, MIN_POSE_DETECTION_CONFIDENCE, MIN_POSE_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE, MAX_GAP_FRAMES, MAX_MERGE_GAP_FRAMES,
    POS_GATE, ACCEPT_SCORE, FULL_BODY_HEIGHT, BATSMAN_CONFIDENCE_THRESHOLD,
    BATSMAN_LOCK_GATE, POS_GATE, ACCEPT_SCORE, MAX_GAP_FRAMES, MAX_MERGE_GAP_FRAMES,
)

# Import batsman selector and utilities
from batsman_selector import (
    BatsmanSelector,
    BATSMAN_LOCK_FRAMES, BATSMAN_SWITCH_THRESHOLD, BATSMAN_LOST_TOLERANCE,
    POSITION_WEIGHT, SIZE_WEIGHT, POSE_WEIGHT, TEMPORAL_WEIGHT, MOVEMENT_WEIGHT,
    BATSMAN_CONFIDENCE_THRESHOLD, MIN_SCORE_DIFFERENTIAL,
    BATTING_REGION_CENTER, BATTING_REGION_SIGMA,
    MIN_BATSMAN_BBOX_HEIGHT, IDEAL_BATSMAN_BBOX_HEIGHT,
    WICKETKEEPER_Y_MIN, NON_STRIKER_X_MAX, NON_STRIKER_Y_MIN,
    create_person_track_from_yolo,
)

# Import centralized YOLO config
from config_yolo import (
    ROI_NUM_POSES, ROI_MIN_POSE_DETECTION_CONFIDENCE,
    ROI_MIN_POSE_PRESENCE_CONFIDENCE, ROI_MIN_TRACKING_CONFIDENCE,
    ROI_PADDING_X, ROI_PADDING_Y, MIN_ROI_SIZE,
    MAX_RECOVERY_FRAMES, MIN_LOCK_FRAMES_BEFORE_SWITCH,
    WICKETKEEPER_Y_MIN, NON_STRIKER_X_MAX, NON_STRIKER_Y_MIN,
    YOLO_MODEL_NAME, YOLO_CONF_THRESHOLD, YOLO_PERSON_CLASS,
    YOLO_TRACKER_CONFIG, YOLO_PERSIST, YOLO_VERBOSE,
    DEBUG_MODE, DEBUG_VERBOSE, FAST_MODE,
    BATSMAN_ROI, ROI_CROP_PAD_X, ROI_CROP_PAD_Y, BATSMAN_ROI_MIN_OVERLAP,
    MIN_MEDIAPIPE_CROP_SIZE, MEDIAPIPE_CROP_EXPANSION_X, MEDIAPIPE_CROP_EXPANSION_Y,
)


@dataclass
class YOLOTrack:
    """YOLO tracking result for a single frame."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords
    center: Tuple[float, float]  # pixel coords
    confidence: float


class YOLOGuidedPoseExtractor:
    """
    YOLO-guided pose extractor with integrated batsman selection.

    Pipeline:
    1. Crop frame to batsman ROI
    2. YOLO detects/tracks people on ROI crop
    3. Convert coordinates back to full-frame
    4. Filter by ROI overlap (reject wicketkeeper, bowler, umpire, non-striker)
    5. Batsman selector scores each track and picks the batsman
    6. Crop the selected batsman with padding
    7. Run MediaPipe Pose on the cropped ROI
    8. Convert landmarks back to full-frame coordinates

    This avoids identity confusion by using YOLO's person tracking on the batting ROI
    + cricket-specific batsman scoring, and only runs MediaPipe on the selected batsman's ROI.

    Output format is identical to the original PoseExtractor.analyze() for
    drop-in compatibility with the rest of the pipeline.
    """

    def __init__(self, model_path: str, yolo_model_path: str = None):
        # MediaPipe setup (for ROI pose estimation)
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        def _make_options(running_mode, num_poses):
            return PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=running_mode,
                num_poses=num_poses,
                min_pose_detection_confidence=ROI_MIN_POSE_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=ROI_MIN_POSE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=ROI_MIN_TRACKING_CONFIDENCE,
            )

        # Single-pose landmarker for batsman ROI
        self.landmarker_roi = PoseLandmarker.create_from_options(
            _make_options(mp.tasks.vision.RunningMode.VIDEO, 1)
        )

        # YOLO model for person detection/tracking
        from ultralytics import YOLO
        yolo_path = yolo_model_path or YOLO_MODEL_NAME
        self.yolo = YOLO(yolo_path)

        # Batsman selector
        self.batsman_selector = BatsmanSelector()

        # Tracking state
        self.yolo_track_history: Dict[int, Dict] = {}
        self.current_batsman_id: Optional[int] = None
        self.batsman_lock_frames = 0

        # Batsman recovery state (for YOLO track drops)
        self.last_known_batsman_bbox = None
        self.batsman_recovery_frames = 0
        self.MAX_RECOVERY_FRAMES = MAX_RECOVERY_FRAMES

        # Stats
        self.stats = {
            "frames_processed": 0,
            "frames_with_batsman": 0,
            "frames_roi_mediapipe": 0,
            "frames_mediapipe_failed": 0,
            "yolo_detections_total": 0,
            # New physical-person tracking stats
            "track_id_changes": 0,
            "batsman_lost_frames": 0,
            "recovery_frames": 0,
            "physical_batsman_switches": 0,
        }

        # Debug video writer (initialized on demand)
        self._debug_writer = None
        self._debug_video_path = None
        self._debug_video_fps = None
        self._debug_video_size = None

    def _init_debug_writer(self, video_path, fps, frame_size):
        """Initialize debug video writer.

        Debug videos are written to the output_videos directory (never into
        input_videos) to keep original input videos strictly unchanged.
        """
        import cv2
        from pathlib import Path
        from config_yolo import OUTPUT_VIDEOS_DIR
        OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        self._debug_video_path = OUTPUT_VIDEOS_DIR / (
            Path(video_path).stem + '.debug.mp4'
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._debug_writer = cv2.VideoWriter(str(self._debug_video_path), fourcc, fps, frame_size)
        self._debug_video_fps = fps
        self._debug_video_size = frame_size
        print(f"[DEBUG] Writing debug video to: {self._debug_video_path}")

    def _close_debug_writer(self):
        """Close debug video writer."""
        if self._debug_writer is not None:
            self._debug_writer.release()
            self._debug_writer = None
            print(f"[DEBUG] Debug video saved: {self._debug_video_path}")

    def _draw_debug_frame(self, frame, yolo_tracks, selection, frame_num, total_frames):
        """Draw debug visualization: ROI, YOLO bboxes, track IDs, selected batsman."""
        h, w = frame.shape[:2]
        
        # Draw Batsman ROI rectangle (yellow)
        roi_x1 = int(BATSMAN_ROI["x1"] * w)
        roi_y1 = int(BATSMAN_ROI["y1"] * h)
        roi_x2 = int(BATSMAN_ROI["x2"] * w)
        roi_y2 = int(BATSMAN_ROI["y2"] * h)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
        cv2.putText(frame, "BATSMAN ROI", (roi_x1, roi_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw crop region (cyan dashed-ish)
        (crop_x1, crop_y1, crop_x2, crop_y2), _ = self._get_roi_crop_coords()
        cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 255, 0), 1)
        cv2.putText(frame, "YOLO CROP", (crop_x1, crop_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw ALL YOLO person detections (dimmed if not selected) and highlight
        # the selected batsman. Scores come from the batsman selector.
        for track in yolo_tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            track_id = track.track_id
            is_selected = (track.track_id == selection.selected_track_id)
            track_score = selection.all_scores.get(track_id, 0.0) if selection.all_scores else 0.0
            roi_overlap = 0.0
            try:
                tr = self.batsman_selector.track_history.get(track_id)
                if tr is not None:
                    roi_overlap = tr.roi_overlap
            except Exception:
                roi_overlap = 0.0

            if is_selected:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (60, 60, 60)
                thickness = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                f"ID:{track_id} S:{track_score:.2f} R:{roi_overlap:.2f}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Draw the selected batsman's bounding box with a "BATSMAN" label on top.
        if selection.selected_track_id is not None:
            for track in yolo_tracks:
                if track.track_id != selection.selected_track_id:
                    continue
                x1, y1, x2, y2 = map(int, track.bbox)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(
                    frame,
                    "BATSMAN",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    f"TRACK ID: {track.track_id}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
                break

        # Selected batsman info
        if selection.selected_track_id is not None:
            sel_color = self._get_track_color(selection.selected_track_id)
            cv2.putText(frame, f"SELECTED BATSMAN: YOLO ID {selection.selected_track_id} | PHYSICAL ID {selection.physical_batsman_id}", (16, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, sel_color, 2)
            cv2.putText(frame, f"Confidence: {selection.confidence:.3f} | Reason: {selection.reason}", (16, 74),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Track ID change indicator
            if selection.track_id_changed:
                cv2.putText(frame, ">>> TRACK ID CHANGED - SAME BATSMAN <<<", (16, 98),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Batsman lost / recovery
            if selection.batsman_lost:
                cv2.putText(frame, ">>> BATSMAN LOST - RECOVERY ACTIVE <<<", (16, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            elif selection.recovery_active:
                cv2.putText(frame, ">>> RECOVERY MODE - USING PREDICTION <<<", (16, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "NO BATSMAN SELECTED (below threshold)", (16, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            cv2.putText(frame, f"Best score: {max(selection.all_scores.values()) if selection.all_scores else 0:.3f}", (16, 74),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame

    @staticmethod
    def _get_track_color(track_id):
        """Generate consistent color for track ID."""
        import hashlib
        hash_val = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
        r = max((hash_val & 0xFF), 100)
        g = max(((hash_val >> 8) & 0xFF), 100)
        b = max(((hash_val >> 16) & 0xFF), 100)
        return (b, g, r)  # BGR for OpenCV

    def _get_roi_crop_coords(self):
        """Get ROI crop coordinates in pixel space.
        
        Returns:
            (crop_x1, crop_y1, crop_x2, crop_y2) in pixel coordinates of full frame
            (roi_x1, roi_y1, roi_x2, roi_y2) - the original ROI in pixel coordinates
        """
        h, w = self._last_frame_shape[:2]
        
        # Original ROI in pixel coordinates
        roi_x1 = int(BATSMAN_ROI["x1"] * w)
        roi_y1 = int(BATSMAN_ROI["y1"] * h)
        roi_x2 = int(BATSMAN_ROI["x2"] * w)
        roi_y2 = int(BATSMAN_ROI["y2"] * h)
        
        # Add padding around ROI for crop
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        pad_x = int(roi_w * ROI_CROP_PAD_X)
        pad_y = int(roi_h * ROI_CROP_PAD_Y)
        
        # Crop coordinates (clamped to frame boundaries)
        crop_x1 = max(0, roi_x1 - pad_x)
        crop_y1 = max(0, roi_y1 - pad_y)
        crop_x2 = min(w, roi_x2 + pad_x)
        crop_y2 = min(h, roi_y2 + pad_y)
        
        return (crop_x1, crop_y1, crop_x2, crop_y2), (roi_x1, roi_y1, roi_x2, roi_y2)

    def _crop_to_roi(self, frame):
        """Crop frame to batsman ROI with padding.
        
        Returns:
            roi_crop: cropped frame
            crop_offset: (crop_x1, crop_y1) - offset of crop in full frame
            roi_bbox: (roi_x1, roi_y1, roi_x2, roi_y2) - original ROI in full frame
        """
        (crop_x1, crop_y1, crop_x2, crop_y2), (roi_x1, roi_y1, roi_x2, roi_y2) = self._get_roi_crop_coords()
        
        roi_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_offset = (crop_x1, crop_y1)
        roi_bbox = (roi_x1, roi_y1, roi_x2, roi_y2)
        
        return roi_crop, crop_offset, roi_bbox

    def _convert_roi_boxes_to_full_frame(self, boxes, crop_offset):
        """Convert boxes from ROI crop coordinates to full-frame coordinates.
        
        Args:
            boxes: (N, 4) array of boxes in crop coordinates (x1, y1, x2, y2)
            crop_offset: (crop_x1, crop_y1) - offset of crop in full frame
            
        Returns:
            boxes in full-frame coordinates
        """
        crop_x1, crop_y1 = crop_offset
        if len(boxes) == 0:
            return boxes
        boxes_full = boxes.copy()
        boxes_full[:, 0] += crop_x1
        boxes_full[:, 1] += crop_y1
        boxes_full[:, 2] += crop_x1
        boxes_full[:, 3] += crop_y1
        return boxes_full

    def _filter_detections_by_roi_overlap(self, boxes_full, roi_bbox):
        """Filter detections to keep only those with sufficient ROI overlap.
        
        Args:
            boxes_full: boxes in full-frame coordinates (x1, y1, x2, y2)
            roi_bbox: (roi_x1, roi_y1, roi_x2, roi_y2) in full-frame coordinates
            
        Returns:
            keep_mask: boolean array of which boxes to keep
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bbox
        roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
        
        keep_mask = []
        for box in boxes_full:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            
            # Calculate intersection with ROI
            ix1 = max(x1, roi_x1)
            iy1 = max(y1, roi_y1)
            ix2 = min(x2, roi_x2)
            iy2 = min(y2, roi_y2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                keep_mask.append(False)
                continue
                
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            
            # Check coverage (intersection / box_area) - bbox inside ROI
            coverage = inter_area / box_area if box_area > 0 else 0
            
            # Check IoU
            iou = inter_area / (box_area + roi_area - inter_area) if (box_area + roi_area - inter_area) > 0 else 0
            
            # Keep if sufficient overlap (either coverage or IoU)
            if coverage >= BATSMAN_ROI_MIN_OVERLAP or iou >= BATSMAN_ROI_MIN_OVERLAP:
                keep_mask.append(True)
            else:
                keep_mask.append(False)
                
        return np.array(keep_mask)

    def _filter_tracks_by_roi(self, yolo_tracks):
        """Keep only YOLO tracks whose bbox sufficiently overlaps the batsman ROI.

        Detections outside the configured batsman ROI (non-striker, bowler,
        wicketkeeper, umpire) are rejected so they can never become the
        selected batsman. Returns the filtered list (preserving order).
        """
        if len(yolo_tracks) == 0:
            return yolo_tracks

        _, roi_bbox = self._get_roi_crop_coords()
        boxes_full = np.array([t.bbox for t in yolo_tracks], dtype=float)
        keep_mask = self._filter_detections_by_roi_overlap(boxes_full, roi_bbox)
        return [t for t, keep in zip(yolo_tracks, keep_mask) if bool(keep)]

    def _yolo_detect_track(self, frame, timestamp_ms):
        """Run YOLO detection + tracking on ROI CROP only.
        
        Running on the batting ROI crop:
        - Excludes non-striker, bowler, wicketkeeper, umpire from detection entirely
        - Gives stable track IDs within the batting region
        - More efficient (smaller image)
        """
        # 1. Crop frame to batsman ROI with padding
        roi_crop, crop_offset, roi_bbox = self._crop_to_roi(frame)
        crop_x1, crop_y1 = crop_offset
        
        # 2. Run YOLO tracking on the ROI CROP
        results = self.yolo.track(
            roi_crop,
            conf=YOLO_CONF_THRESHOLD,
            classes=[YOLO_PERSON_CLASS],
            tracker=YOLO_TRACKER_CONFIG,
            persist=YOLO_PERSIST,
            verbose=YOLO_VERBOSE
        )
        
        detections = results[0].boxes
        
        if detections is not None and detections.id is not None:
            track_ids = detections.id.cpu().numpy().astype(int)
            boxes_crop = detections.xyxy.cpu().numpy()  # In crop coordinates
            confs = detections.conf.cpu().numpy()
            
            # 3. Convert boxes from crop coordinates to full-frame coordinates
            boxes_full = self._convert_roi_boxes_to_full_frame(boxes_crop, crop_offset)
            
            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes_full[i]
                conf = float(confs[i])
                
                yield YOLOTrack(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2, (y1 + y2) / 2),
                    confidence=conf
                )

    def _update_yolo_tracks(self, yolo_tracks, frame_num):
        """Update YOLO track history and return tracks dict for batsman selector."""
        tracks_dict = {}
        active_ids = set()

        for track in yolo_tracks:
            active_ids.add(track.track_id)

            h, w = self._last_frame_shape[:2]
            track_obj = create_person_track_from_yolo(
                track.track_id, track.bbox, track.confidence, self._last_frame_shape
            )
            tracks_dict[track.track_id] = track_obj
            active_ids.add(track.track_id)

            if track.track_id not in self.yolo_track_history:
                self.yolo_track_history[track.track_id] = {
                    'frames': 0,
                    'first_frame': frame_num,
                    'last_frame': frame_num,
                    'position_history': [],
                }
            self.yolo_track_history[track.track_id]['frames'] += 1
            self.yolo_track_history[track.track_id]['last_frame'] = frame_num
            self.yolo_track_history[track.track_id]['position_history'].append((track.center[0]/w, track.center[1]/h))
            if len(self.yolo_track_history[track.track_id]['position_history']) > 30:
                self.yolo_track_history[track.track_id]['position_history'].pop(0)

        for track_id in list(self.yolo_track_history.keys()):
            if track_id not in active_ids:
                self.yolo_track_history[track_id]['lost_frames'] = \
                    self.yolo_track_history[track_id].get('lost_frames', 0) + 1

        return tracks_dict

    @staticmethod
    def _landmarks_to_lms(pose):
        lms = []
        for landmark in pose:
            lms.append(LM(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=getattr(landmark, "visibility", float("nan")),
                presence=getattr(landmark, "presence", float("nan")),
            ))
        return lms

    def _crop_batsman_roi(self, frame, bbox, padding_x=ROI_PADDING_X, padding_y=ROI_PADDING_Y):
        """Crop the batsman region with padding, expanding for small bboxes to meet MediaPipe minimum."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1

        # Base padding
        pad_x = int(bw * padding_x)
        pad_y = int(bh * padding_y)

        # For small bboxes, expand crop to meet MediaPipe minimum input size
        # Use expansion factors from config
        min_crop_w = MIN_MEDIAPIPE_CROP_SIZE
        min_crop_h = MIN_MEDIAPIPE_CROP_SIZE
        
        # Current crop size with base padding
        crop_w = bw + 2 * pad_x
        crop_h = bh + 2 * pad_y
        
        # If crop is too small, expand using expansion factors
        if crop_w < min_crop_w:
            extra_w = min_crop_w - crop_w
            pad_x += extra_w // 2 + (extra_w % 2)
        if crop_h < min_crop_h:
            extra_h = min_crop_h - crop_h
            pad_y += extra_h // 2 + (extra_h % 2)
        
        # Additional expansion for very small bboxes (configurable)
        if bw < 80:  # Small bbox threshold
            extra_pad_x = int(bw * MEDIAPIPE_CROP_EXPANSION_X)
            extra_pad_y = int(bh * MEDIAPIPE_CROP_EXPANSION_Y)
            pad_x += extra_pad_x
            pad_y += extra_pad_y

        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)

        roi = frame[ry1:ry2, rx1:rx2]
        return roi, (rx1, ry1, rx2, ry2)  # Return full ROI bounds for coordinate conversion

    def _roi_landmarks_to_full_frame(self, roi_landmarks, roi_offset):
        """Convert ROI-normalized landmarks back to full-frame normalized coordinates."""
        if roi_landmarks is None:
            return None

        rx1, ry1, rx2, ry2 = roi_offset
        roi_w = rx2 - rx1
        roi_h = ry2 - ry1
        full_h, full_w = self._last_frame_shape[:2]

        full_lms = []
        for lm in roi_landmarks:
            # MediaPipe returns landmarks normalized to ROI dimensions (0-1)
            # Convert to full-frame normalized coordinates
            full_x = (lm.x * roi_w + rx1) / full_w
            full_y = (lm.y * roi_h + ry1) / full_h
            full_lms.append(LM(
                x=full_x,
                y=full_y,
                z=lm.z,
                visibility=lm.visibility,
                presence=lm.presence,
            ))
        return full_lms

    def _detect_batsman_roi(self, frame, timestamp_ms, batsman_bbox):
        """Run MediaPipe on batsman ROI and return full-frame landmarks."""
        if batsman_bbox is None:
            return None

        x1, y1, x2, y2 = batsman_bbox
        if x2 <= x1 or y2 <= y1:
            return None

        # Use configurable minimum crop size
        min_w = MIN_MEDIAPIPE_CROP_SIZE
        min_h = MIN_MEDIAPIPE_CROP_SIZE
        if (x2 - x1) < min_w or (y2 - y1) < min_h:
            # Still try - _crop_batsman_roi will expand the crop
            pass

        roi, roi_offset = self._crop_batsman_roi(frame, batsman_bbox)

        if roi.size == 0:
            return None

        # Verify crop meets minimum size
        crop_h, crop_w = roi.shape[:2]
        if crop_w < min_w or crop_h < min_h:
            if DEBUG_MODE:
                print(f"[ROI MediaPipe] Crop too small: {crop_w}x{crop_h} < {min_w}x{min_h}")
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            results = self.landmarker_roi.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            if DEBUG_MODE:
                print(f"[ROI MediaPipe] Error: {e}")
            return None

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        roi_pose = results.pose_landmarks[0]
        roi_lms = self._landmarks_to_lms(roi_pose)
        full_lms = self._roi_landmarks_to_full_frame(roi_lms, roi_offset)

        return full_lms

    def _validate_pose_landmarks(self, landmarks: List) -> bool:
        """Validate MediaPipe pose landmarks before accepting.

        Rejects poses with:
        - Missing key landmarks (nose, shoulders)
        - Coordinates outside [0, 1] range (invalid crop mapping)
        - Very low visibility on key landmarks
        """
        if landmarks is None or len(landmarks) == 0:
            return False

        # Key landmark indices (MediaPipe Pose)
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        def _lm(idx):
            if idx >= len(landmarks):
                return None
            return landmarks[idx]

        nose = _lm(NOSE)
        l_shoulder = _lm(LEFT_SHOULDER)
        r_shoulder = _lm(RIGHT_SHOULDER)

        # Must have nose and at least one shoulder
        if nose is None:
            return False
        if l_shoulder is None and r_shoulder is None:
            return False

        # Check coordinates are within [0, 1] range
        for lm in [nose, l_shoulder, r_shoulder]:
            if lm is None:
                continue
            if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                return False
            if lm.visibility is not None and lm.visibility < 0.1:
                return False

        return True

    def analyze(self, video_path, fps=25.0, debug_video=False):
        """
        Main analysis entry point - compatible with original PoseExtractor.analyze()

        Args:
            video_path: Path to input video
            fps: Video FPS (auto-detected if not provided)
            debug_video: If True, write debug visualization video

        Returns:
            (frames_data, best_traj, batsman_confidence)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Initialize debug video writer if requested
        if debug_video:
            # Read first frame to get size
            ok, first_frame = cap.read()
            if ok:
                self._init_debug_writer(video_path, fps, (first_frame.shape[1], first_frame.shape[0]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            else:
                debug_video = False

        frames_data = []
        frame_index = 0

        self._last_frame_shape = None

        print()
        print("===================================")
        print("YOLO-GUIDED POSE EXTRACTION")
        print("===================================")
        print(f"Video: {video_path}")
        print(f"FPS: {fps}")
        print(f"Expected frames: {reported}")
        if debug_video:
            print(f"Debug video: ENABLED")
        print()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            self._last_frame_shape = frame.shape

            timestamp_ms = int((frame_index / max(fps, 1.0)) * 1000)

            if video_preprocess.should_enhance():
                frame = video_preprocess.enhance_frame(frame)

            # 1. YOLO detection + tracking on ROI crop
            yolo_tracks_raw = list(self._yolo_detect_track(frame, timestamp_ms))
            self.stats["yolo_detections_total"] += len(yolo_tracks_raw)

            # 1b. Filter detections to those overlapping the batsman ROI
            #     (reject non-striker / bowler / wicketkeeper outside the ROI)
            yolo_tracks = self._filter_tracks_by_roi(yolo_tracks_raw)

            # 2. Update YOLO track history and prepare for batsman selector
            yolo_tracks_dict = self._update_yolo_tracks(yolo_tracks, frame_index)

            # 3. Run batsman selector
            selection = self.batsman_selector.update(yolo_tracks_dict, frame_index)

            # 4. Get selected batsman bbox - ONLY use current detection, never stale
            batsman_bbox = None
            batsman_track = None

            if selection.selected_track_id is not None:
                for track in yolo_tracks:
                    if track.track_id == selection.selected_track_id:
                        batsman_bbox = track.bbox
                        batsman_track = track
                        self.last_known_batsman_bbox = track.bbox
                        self.batsman_recovery_frames = 0
                        break
                # If selected_track_id is not None but track not in current detections,
                # batsman_bbox stays None - no MediaPipe on stale bbox during recovery.

            # Identity switch detection (BEFORE updating current_batsman_id)
            identity_switch_detected = False
            if self.current_batsman_id is not None and \
               selection.selected_track_id is not None and \
               selection.selected_track_id != self.current_batsman_id:
                identity_switch_detected = True

            # 5. Run MediaPipe on batsman ROI
            pose_landmarks = None
            tracking_ok = False
            confidence = 0.0
            n_detected = len(yolo_tracks)
            selected_person_index = None
            batsman_center = None
            batsman_detected = False
            
            # New: physical-person tracking info
            physical_batsman_id = selection.physical_batsman_id
            track_id_changed = selection.track_id_changed
            batsman_lost = selection.batsman_lost
            recovery_active = selection.recovery_active

            if batsman_bbox is not None:
                pose_landmarks = self._detect_batsman_roi(frame, timestamp_ms, batsman_bbox)

                if pose_landmarks is not None:
                    # Validate pose landmarks before accepting
                    if not self._validate_pose_landmarks(pose_landmarks):
                        if DEBUG_MODE:
                            print(f"[POSE VALIDATION] Rejected invalid pose at frame {frame_index}")
                        pose_landmarks = None
                        self.stats["frames_mediapipe_failed"] += 1
                    else:
                        tracking_ok = True
                        confidence = selection.confidence
                        batsman_detected = True
                        self.stats["frames_roi_mediapipe"] += 1
                        self.stats["frames_with_batsman"] += 1

                        for i, track in enumerate(yolo_tracks):
                            if track.track_id == selection.selected_track_id:
                                selected_person_index = i
                                break

                        h, w = frame.shape[:2]
                        if batsman_track:
                            batsman_center = (batsman_track.center[0] / w, batsman_track.center[1] / h)
                else:
                    self.stats["frames_mediapipe_failed"] += 1

            self.current_batsman_id = selection.selected_track_id

            # Build output frame data
            frame_data = {
                "pose": pose_landmarks,
                "tracking_ok": tracking_ok,
                "confidence": confidence,
                "n_detected": n_detected,
                "selected_person_index": selected_person_index,
                "selected_batsman_track_id": selection.selected_track_id,
                "batsman_center": batsman_center,
                "identity_switch_detected": identity_switch_detected,
                "batsman_detected": batsman_detected,
                # New physical-person tracking fields
                "physical_batsman_id": physical_batsman_id,
                "track_id_changed": track_id_changed,
                "batsman_lost": batsman_lost,
                "recovery_active": recovery_active,
                "selection_reason": selection.reason,
                "continuity_scores": selection.all_continuity_scores,
            }

            frames_data.append(frame_data)

            # Write debug frame if enabled
            if debug_video and self._debug_writer is not None:
                debug_frame = frame.copy()
                debug_frame = self._draw_debug_frame(debug_frame, yolo_tracks_raw, selection, frame_index, reported)
                self._debug_writer.write(debug_frame)

            # Update stats
            self.stats["frames_processed"] += 1
            if tracking_ok:
                self.stats["frames_with_batsman"] += 1
            if track_id_changed:
                self.stats["track_id_changes"] = self.stats.get("track_id_changes", 0) + 1
            if batsman_lost:
                self.stats["batsman_lost_frames"] = self.stats.get("batsman_lost_frames", 0) + 1
            if recovery_active:
                self.stats["recovery_frames"] = self.stats.get("recovery_frames", 0) + 1
            # Track physical batsman switches (only true person switches, not track ID changes)
            if selection.reason == "physical_person_switched":
                self.stats["physical_batsman_switches"] = self.stats.get("physical_batsman_switches", 0) + 1

            frame_index += 1

        cap.release()

        # Close debug writer
        if debug_video:
            self._close_debug_writer()

        print()
        print("===================================")
        print("EXTRACTION COMPLETE")
        print("===================================")
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Frames with batsman: {self.stats['frames_with_batsman']}")
        print(f"Frames ROI MediaPipe: {self.stats['frames_roi_mediapipe']}")
        print(f"Frames MediaPipe failed: {self.stats['frames_mediapipe_failed']}")
        print(f"Total YOLO detections: {self.stats['yolo_detections_total']}")
        print()
        print("PHYSICAL-PERSON TRACKING:")
        print(f"  Track ID changes (same batsman): {self.stats.get('track_id_changes', 0)}")
        print(f"  Batsman lost frames: {self.stats.get('batsman_lost_frames', 0)}")
        print(f"  Recovery frames: {self.stats.get('recovery_frames', 0)}")
        print(f"  Physical batsman switches: {self.stats.get('physical_batsman_switches', 0)}")
        print()

        total_conf = sum(f["confidence"] for f in frames_data)
        avg_conf = total_conf / max(len(frames_data), 1)

        return frames_data, None, avg_conf

    def close(self):
        self.landmarker_roi.close()

    def draw_pose(self, frame, pose):
        """Draw pose landmarks on frame (for preview/debug)."""
        from pose_extractor import PoseExtractor as OriginalPoseExtractor
        if not hasattr(self, '_original_draw'):
            self._original_draw = OriginalPoseExtractor.__new__(OriginalPoseExtractor)
        return self._original_draw.draw_pose(frame, pose)


# Compatibility wrapper to match original PoseExtractor interface
class PoseExtractor:
    """Compatibility wrapper - uses YOLO-guided extraction internally."""

    def __init__(self, model_path):
        self._extractor = YOLOGuidedPoseExtractor(model_path)

    def analyze(self, video_path, fps=25.0):
        return self._extractor.analyze(video_path, fps)

    def draw_pose(self, frame, pose):
        from pose_extractor import PoseExtractor as OriginalPoseExtractor
        if not hasattr(self, '_original_draw'):
            self._original_draw = OriginalPoseExtractor.__new__(OriginalPoseExtractor)
        return self._original_draw.draw_pose(frame, pose)

    def close(self):
        self._extractor.close()


__all__ = ['PoseExtractor', 'YOLOGuidedPoseExtractor']