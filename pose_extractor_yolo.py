"""YOLO-guided pose extraction with batsman selection.

This module replaces the MediaPipe-only detection/tracking/selection pipeline
with a YOLO-guided approach:

  Video
    → YOLO person detection
    → YOLO tracking (ByteTrack)
    → Batsman ID selection (using multi-evidence scoring)
    → Crop selected batsman with configurable padding
    → MediaPipe Pose on batsman ROI only
    → Convert ROI landmarks back to full-frame coordinates
    → Existing landmark/angle/phase pipeline

This avoids identity confusion by using YOLO's person tracking + cricket-specific
batsman scoring, and only runs MediaPipe on the selected batsman's ROI.

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
)

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

from config import FAST_MODE

# Local constants
ROI_NUM_POSES = 1
ROI_MIN_POSE_DETECTION_CONFIDENCE = 0.3
ROI_MIN_POSE_PRESENCE_CONFIDENCE = 0.3
ROI_MIN_TRACKING_CONFIDENCE = 0.3

ROI_PADDING_X = 0.15
ROI_PADDING_Y = 0.10
MIN_ROI_SIZE = 64


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
    1. YOLO detects/tracks people on every frame
    2. Batsman selector scores each track and picks the batsman
    3. Crop the selected batsman with padding
    4. Run MediaPipe Pose on the cropped ROI
    5. Convert landmarks back to full-frame coordinates

    This avoids identity confusion by using YOLO's person tracking + cricket-specific
    batsman scoring, and only runs MediaPipe on the selected batsman's ROI.

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
        }

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
        """Crop the batsman region with padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * padding_x)
        pad_y = int(bh * padding_y)

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

        if (x2 - x1) < 64 or (y2 - y1) < 64:
            return None

        roi, roi_offset = self._crop_batsman_roi(frame, batsman_bbox)

        if roi.size == 0:
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

    def _yolo_detect_track(self, frame, timestamp_ms):
        """Run YOLO detection + tracking on a frame."""
        results = self.yolo.track(
            frame,
            conf=0.25,
            classes=[0],
            tracker="bytetrack_custom.yaml",
            persist=True,
            verbose=False
        )

        tracks = []
        detections = results[0].boxes

        if detections is not None and detections.id is not None:
            track_ids = detections.id.cpu().numpy().astype(int)
            boxes = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = boxes[i]
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

    def analyze(self, video_path, fps=25.0):
        """
        Main analysis entry point - compatible with original PoseExtractor.analyze()

        Returns:
            (frames_data, best_traj, batsman_confidence)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

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
        print()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            self._last_frame_shape = frame.shape

            timestamp_ms = int((frame_index / max(fps, 1.0)) * 1000)

            if video_preprocess.should_enhance():
                frame = video_preprocess.enhance_frame(frame)

            # 1. YOLO detection + tracking
            yolo_tracks = list(self._yolo_detect_track(frame, timestamp_ms))
            self.stats["yolo_detections_total"] += len(yolo_tracks)

            # 2. Update YOLO track history and prepare for batsman selector
            yolo_tracks_dict = self._update_yolo_tracks(yolo_tracks, frame_index)

            # 3. Run batsman selector
            selection = self.batsman_selector.update(yolo_tracks_dict, frame_index)

            # 4. Get selected batsman bbox (with recovery logic)
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

                if batsman_bbox is None and self.current_batsman_id == selection.selected_track_id \
                        and self.last_known_batsman_bbox is not None:
                    self.batsman_recovery_frames += 1
                    if self.batsman_recovery_frames <= self.MAX_RECOVERY_FRAMES:
                        batsman_bbox = self.last_known_batsman_bbox
                        class DummyTrack:
                            def __init__(self, bbox):
                                self.bbox = bbox
                                self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                                self.track_id = selection.selected_track_id
                        batsman_track = DummyTrack(batsman_bbox)
                else:
                    self.last_known_batsman_bbox = None
                    self.batsman_recovery_frames = 0

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
            identity_switch_detected = False
            batsman_detected = False

            if batsman_bbox is not None:
                pose_landmarks = self._detect_batsman_roi(frame, timestamp_ms, batsman_bbox)

                if pose_landmarks is not None:
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
            }

            frames_data.append(frame_data)

            self.stats["frames_processed"] += 1
            if tracking_ok:
                self.stats["frames_with_batsman"] += 1

            frame_index += 1

        cap.release()

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

        total_conf = sum(f["confidence"] for f in frames_data)
        avg_conf = total_conf / max(len(frames_data), 1)

        return frames_data, None, avg_conf

    def close(self):
        self.landmarker_roi.close()


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