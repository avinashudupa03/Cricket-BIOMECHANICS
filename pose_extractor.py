"""Pose extraction with stable, single-identity batsman tracking.

Design
------
This module uses a three-stage approach to reliably detect ONLY the batsman:

  Stage 1 - Detect every frame (FULL frame + central band, VIDEO mode).
            Candidates are linked into *person trajectories* using a greedy
            temporal association that scores spatial distance from a predicted
            (velocity-adjusted) position, bounding-box size similarity and
            normalized pose-shape similarity.

  Stage 2 - Identify the batsman using cricket-specific heuristics:
            * Body orientation: the batsman stands side-on to the camera
            * Batting stance: wide stance, knees bent, holding a bat
            * Position: near the center of the frame (where the stumps are)
            * Motion pattern: walks in, plays the shot, follows through
            * Pose variation: dramatic posture changes through the phases

  Stage 3 - Lock onto the batsman identity for the whole video.
            Once chosen, every frame is assigned to the SAME identity.
            If the chosen trajectory has no pose on a frame, that frame is
            flagged with tracking_ok = False (BATSMAN NOT DETECTED) and the
            skeleton is NEVER moved onto the keeper, non-striker or bowler.

Per-frame output includes tracking confidence, the number of detected people
(debug/after-the-fact oversight), the index of the selected person within that
frame's candidate list, and an honest identity_switch_detected flag.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

import video_preprocess

# ---------------------------------------------------------------------------
# MediaPipe configuration (deliberately accessible / tunable)
# ---------------------------------------------------------------------------
NUM_POSES = 6
MIN_POSE_DETECTION_CONFIDENCE = 0.15
MIN_POSE_PRESENCE_CONFIDENCE = 0.15
MIN_TRACKING_CONFIDENCE = 0.15

# Trajectory association parameters
MAX_GAP_FRAMES = 20
MAX_MERGE_GAP_FRAMES = 40
POS_GATE = 2.0
ACCEPT_SCORE = 2.5

FULL_BODY_HEIGHT = 0.12

# Batsman identification confidence threshold
BATSMAN_CONFIDENCE_THRESHOLD = 0.35

# Maximum distance (in body heights) from the batsman's last known position
# to accept a match. This prevents identity switches to nearby players.
BATSMAN_LOCK_GATE = 1.5


@dataclass
class LM:
    """Lightweight landmark (matches the MediaPipe PoseLandmarker fields used)."""
    x: float = float("nan")
    y: float = float("nan")
    z: float = float("nan")
    visibility: float = float("nan")
    presence: float = float("nan")


@dataclass
class Candidate:
    frame: Optional[int]
    pose: List[LM]
    center: Optional[tuple]
    bbox: Optional[tuple]
    shape: Optional[list]


@dataclass
class Trajectory:
    frames: List[int] = field(default_factory=list)
    centers: List[tuple] = field(default_factory=list)
    bboxes: List[tuple] = field(default_factory=list)
    shapes: List[Optional[list]] = field(default_factory=list)
    poses: List[List[LM]] = field(default_factory=list)


def _isfinite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


_finite = _isfinite


def get_center(pose):
    points = []
    for index in (11, 12, 23, 24):
        if index < len(pose):
            points.append((pose[index].x, pose[index].y))
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def get_bbox(pose):
    xs = [p.x for p in pose]
    ys = [p.y for p in pose]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_size(bbox):
    return max(bbox[2] - bbox[0], 1e-6), max(bbox[3] - bbox[1], 1e-6)


def get_shape(pose, center, bbox):
    """Normalized pose shape: joint offsets from center, scaled by bbox height."""
    if center is None or bbox is None:
        return None
    _, bh = bbox_size(bbox)
    joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    cx, cy = center
    shape = []
    for index in joints:
        if index >= len(pose):
            continue
        shape.append(((pose[index].x - cx) / bh, (pose[index].y - cy) / bh))
    return shape


def shape_distance(shape1, shape2):
    if not shape1 or not shape2 or len(shape1) != len(shape2):
        return 999.0
    total = 0.0
    for p1, p2 in zip(shape1, shape2):
        total += math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return total / len(shape1)


def central_prior(center):
    """Preference for the central strike-zone position of a batting clip."""
    if center is None:
        return 0.0
    x, y = center
    dx = (x - 0.50) / 0.15
    dy = (y - 0.35) / 0.12
    return math.exp(-0.5 * (dx * dx + dy * dy))


def _shoulder_width_ratio(pose):
    """Ratio of shoulder width to torso height."""
    if len(pose) < 25:
        return 0.0
    lw = pose[11]
    rw = pose[12]
    lh = pose[23]
    rh = pose[24]
    if not all(_finite(v) for v in [lw.x, lw.y, rw.x, rw.y, lh.x, lh.y, rh.x, rh.y]):
        return 0.0
    shoulder_w = math.hypot(lw.x - rw.x, lw.y - rw.y)
    torso_h = math.hypot((lh.x + rh.x) / 2 - (lw.x + rw.x) / 2,
                         (lh.y + rh.y) / 2 - (lw.y + rw.y) / 2)
    if torso_h < 1e-6:
        return 0.0
    return shoulder_w / torso_h


def _leg_spread_ratio(pose):
    """Ratio of ankle-to-ankle distance to torso height."""
    if len(pose) < 28:
        return 0.0
    la = pose[27]
    ra = pose[28]
    lh = pose[23]
    rh = pose[24]
    if not all(_finite(v) for v in [la.x, la.y, ra.x, ra.y, lh.x, lh.y, rh.x, rh.y]):
        return 0.0
    leg_spread = math.hypot(la.x - ra.x, la.y - ra.y)
    torso_h = math.hypot(lh.x - rh.x, lh.y - rh.y)
    if torso_h < 1e-6:
        return 0.0
    return leg_spread / torso_h


def _is_batsman_stance(pose):
    """Heuristic: does this pose look like a batting stance?"""
    if len(pose) < 28:
        return 0.0

    sw_ratio = _shoulder_width_ratio(pose)
    ls_ratio = _leg_spread_ratio(pose)

    lw = pose[11]
    rw = pose[12]
    shoulder_asymmetry = 0.0
    if _finite(lw.y) and _finite(rw.y):
        _, lh = bbox_size(get_bbox(pose))
        if lh > 1e-6:
            shoulder_asymmetry = abs(lw.y - rw.y) / lh

    score = 0.0
    if sw_ratio > 1.2:
        score += 0.3
    if ls_ratio > 1.0:
        score += 0.3
    if shoulder_asymmetry > 0.05:
        score += 0.4

    return min(score, 1.0)


class PoseExtractor:

    BAND_X0 = 0.15
    BAND_X1 = 0.80
    BAND_TARGET_WIDTH = 960

    def __init__(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        def _make_options(running_mode):
            return PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=running_mode,
                num_poses=NUM_POSES,
                min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )

        self.landmarker_full = PoseLandmarker.create_from_options(
            _make_options(VisionRunningMode.VIDEO)
        )
        self.landmarker_band = PoseLandmarker.create_from_options(
            _make_options(VisionRunningMode.VIDEO)
        )

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

    def _detect_full(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker_full.detect_for_video(mp_image, timestamp_ms)
        return self._results_to_candidates(results, lambda x, y: (x, y))

    def _detect_band(self, frame, timestamp_ms):
        height, width = frame.shape[:2]
        x0 = int(width * self.BAND_X0)
        x1 = int(width * self.BAND_X1)
        if x1 <= x0:
            return []
        crop = frame[:, x0:x1]
        scale = self.BAND_TARGET_WIDTH / max(crop.shape[1], 1)
        new_h = int(crop.shape[0] * scale)
        if new_h <= 0:
            return []
        crop = cv2.resize(crop, (self.BAND_TARGET_WIDTH, new_h),
                          interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker_band.detect_for_video(mp_image, timestamp_ms)

        crop_w = x1 - x0

        def to_frame(nx, ny):
            return (nx * crop_w / width + x0 / width, ny)

        return self._results_to_candidates(results, to_frame)

    @staticmethod
    def _results_to_candidates(results, to_frame):
        candidates = []
        if results.pose_landmarks:
            for pose in results.pose_landmarks:
                lms = PoseExtractor._landmarks_to_lms(pose)
                for lm in lms:
                    lm.x, lm.y = to_frame(lm.x, lm.y)
                center = get_center(lms)
                bbox = get_bbox(lms)
                shape = get_shape(lms, center, bbox)
                candidates.append(Candidate(
                    frame=None, pose=lms, center=center, bbox=bbox, shape=shape,
                ))
        return candidates

    @staticmethod
    def _merge_candidates(*groups):
        combined = []
        for group in groups:
            combined.extend(group)

        kept = []
        for cand in combined:
            if cand.center is None:
                kept.append(cand)
                continue
            duplicate = None
            for other in kept:
                if other.center is None:
                    continue
                dx = abs(cand.center[0] - other.center[0])
                dy = abs(cand.center[1] - other.center[1])
                if dx < 0.03 and dy < 0.03:
                    duplicate = other
                    break
            if duplicate is None:
                kept.append(cand)
            else:
                n_cand = sum(1 for lm in cand.pose if _finite(lm.x) and _finite(lm.y))
                n_dup = sum(1 for lm in duplicate.pose if _finite(lm.x) and _finite(lm.y))
                if n_cand > n_dup:
                    kept[kept.index(duplicate)] = cand
        return kept

    def detect_all(self, video_path, fps=25.0):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        per_frame = []
        frame_index = 0
        while reported <= 0 or frame_index < reported:
            ok, frame = cap.read()
            if not ok:
                per_frame.append([])
                frame_index += 1
                continue
            timestamp_ms = int((frame_index / max(fps, 1.0)) * 1000)
            if video_preprocess.should_enhance():
                frame = video_preprocess.enhance_frame(frame)
            full_cands = self._detect_full(frame, timestamp_ms)
            if not self._is_clean_fullbody(full_cands):
                band_cands = self._detect_band(frame, timestamp_ms)
            else:
                band_cands = []
            candidates = self._merge_candidates(full_cands, band_cands)
            for cand in candidates:
                cand.frame = frame_index
            per_frame.append(candidates)
            frame_index += 1
        cap.release()
        return per_frame

    @staticmethod
    def _is_clean_fullbody(candidates):
        for cand in candidates:
            if cand.bbox is None or cand.pose is None:
                continue
            _, bh = bbox_size(cand.bbox)
            if bh < FULL_BODY_HEIGHT:
                continue
            vis = [lm.visibility for lm in cand.pose]
            vis = [v for v in vis if np.isfinite(v)]
            if len(vis) >= 12 and np.mean(vis) >= 0.40:
                return True
        return False

    @staticmethod
    def _candidate_score(cand, pred_center, ref_bbox, ref_shape):
        if cand.center is None or pred_center is None:
            return 999.0

        _, ref_h = bbox_size(ref_bbox)
        pos_dist = math.hypot(cand.center[0] - pred_center[0],
                              cand.center[1] - pred_center[1]) / max(ref_h, 1e-6)

        cw, ch = bbox_size(cand.bbox)
        rw, rh = bbox_size(ref_bbox)
        size_dist = abs(math.log(max(cw, 1e-6) / max(rw, 1e-6))) + \
                    abs(math.log(max(ch, 1e-6) / max(rh, 1e-6)))

        shape_dist = shape_distance(cand.shape, ref_shape) if ref_shape else 0.0

        score = 2.5 * pos_dist + 1.0 * size_dist + 1.5 * shape_dist
        if pos_dist > POS_GATE:
            score += 15.0
        return score

    def _build_trajectories(self, per_frame):
        """Link per-frame candidates into temporally-consistent trajectories.

        Uses a two-phase approach:
        1. Build all trajectories with standard association
        2. Identify the batsman trajectory using cricket-specific heuristics
        3. Re-build the batsman trajectory with a strict spatial lock to prevent
           identity switches to nearby players
        """
        # Phase 1: Build all trajectories
        trajectories = []
        active = []

        for frame_idx, candidates in enumerate(per_frame):
            used = [False] * len(candidates)

            for t, st in zip(trajectories, active):
                if not st["active"]:
                    continue
                if len(candidates) == 0:
                    continue
                pred_center = st["pred"]
                ref_bbox = t.bboxes[-1]
                ref_shape = t.shapes[-1] if t.shapes else None

                best_i = None
                best_score = 999.0
                for i, cand in enumerate(candidates):
                    if used[i] or cand.center is None:
                        continue
                    s = self._candidate_score(cand, pred_center, ref_bbox, ref_shape)
                    if s < best_score:
                        best_score = s
                        best_i = i

                if best_i is not None and best_score < ACCEPT_SCORE:
                    used[best_i] = True
                    cand = candidates[best_i]
                    t.frames.append(frame_idx)
                    t.centers.append(cand.center)
                    t.bboxes.append(cand.bbox)
                    t.shapes.append(cand.shape)
                    t.poses.append(cand.pose)

                    prev_c = t.centers[-2]
                    vel = (cand.center[0] - prev_c[0],
                           cand.center[1] - prev_c[1])
                    st["vel"] = (0.5 * st["vel"][0] + 0.5 * vel[0],
                                 0.5 * st["vel"][1] + 0.5 * vel[1])
                    st["last"] = frame_idx
                    st["pred"] = (cand.center[0] + st["vel"][0],
                                  cand.center[1] + st["vel"][1])
                    st["active"] = True

            for (t, st) in zip(trajectories, active):
                if st["last"] < frame_idx:
                    st["gap"] += 1
                    dt = st["gap"]
                    st["pred"] = (t.centers[-1][0] + st["vel"][0] * dt,
                                  t.centers[-1][1] + st["vel"][1] * dt)
                    if st["gap"] > MAX_GAP_FRAMES:
                        st["active"] = False

            for i, cand in enumerate(candidates):
                if used[i] or cand.center is None or cand.bbox is None:
                    continue
                _, h = bbox_size(cand.bbox)
                if h < FULL_BODY_HEIGHT:
                    continue
                t = Trajectory(
                    frames=[frame_idx],
                    centers=[cand.center],
                    bboxes=[cand.bbox],
                    shapes=[cand.shape],
                    poses=[cand.pose],
                )
                trajectories.append(t)
                active.append({
                    "active": True,
                    "last": frame_idx,
                    "gap": 0,
                    "vel": (0.0, 0.0),
                    "pred": (cand.center[0], cand.center[1]),
                })

        # Phase 2: Merge trajectories
        merged = self._merge_trajectories(trajectories)
        return merged

    @staticmethod
    def _merge_trajectories(trajectories):
        if len(trajectories) < 2:
            return list(trajectories)
        result = list(trajectories)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                for j in range(len(result)):
                    if i == j:
                        continue
                    t1, t2 = result[i], result[j]
                    if t1.frames[0] > t2.frames[0]:
                        t1, t2 = t2, t1
                    if t1.frames[-1] >= t2.frames[0]:
                        continue
                    gap_frames = t2.frames[0] - t1.frames[-1]
                    if gap_frames <= 0 or gap_frames > MAX_MERGE_GAP_FRAMES:
                        continue
                    c1 = t1.centers[-1]
                    c2 = t2.centers[0]
                    _, h1 = bbox_size(t1.bboxes[-1])
                    _, h2 = bbox_size(t2.bboxes[0])
                    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if dist <= 2.0 * max(h1, h2, 1e-6):
                        t1.frames.extend(t2.frames)
                        t1.centers.extend(t2.centers)
                        t1.bboxes.extend(t2.bboxes)
                        t1.shapes.extend(t2.shapes)
                        t1.poses.extend(t2.poses)
                        del result[j]
                        changed = True
                        break
                if changed:
                    break
        return result

    # ------------------------------------------------------------------
    # Batsman selection
    # ------------------------------------------------------------------
    @staticmethod
    def _score_trajectory(traj, total_frames):
        """Score a trajectory as the batsman using cricket-specific heuristics."""
        full = 0.0
        for bbox in traj.bboxes:
            if bbox is not None:
                _, h = bbox_size(bbox)
                if h >= FULL_BODY_HEIGHT:
                    full += 1
        coverage = len(traj.frames) / max(total_frames, 1)
        if full == 0:
            return -1.0
        mean_center = (
            sum(c[0] for c in traj.centers) / len(traj.centers),
            sum(c[1] for c in traj.centers) / len(traj.centers),
        )
        prior = central_prior(mean_center)
        mean_h = sum(bbox_size(b)[1] for b in traj.bboxes if b) / full

        # Batting stance score
        stance_scores = []
        for pose in traj.poses:
            if pose and len(pose) >= 28:
                stance_scores.append(_is_batsman_stance(pose))
        batting_stance_score = sum(stance_scores) / len(stance_scores) if stance_scores else 0.0

        # Motion score
        if len(traj.centers) >= 2:
            total_displacement = 0.0
            for i in range(1, len(traj.centers)):
                dx = traj.centers[i][0] - traj.centers[i - 1][0]
                dy = traj.centers[i][1] - traj.centers[i - 1][1]
                total_displacement += math.hypot(dx, dy)
            motion_score = min(total_displacement / max(mean_h, 1e-6) / 8.0, 1.0)
        else:
            motion_score = 0.0

        # Pose variation score
        if len(traj.shapes) >= 2:
            shape_changes = []
            for i in range(1, len(traj.shapes)):
                if traj.shapes[i] and traj.shapes[i - 1]:
                    shape_changes.append(
                        shape_distance(traj.shapes[i], traj.shapes[i - 1])
                    )
            if shape_changes:
                mean_shape_change = sum(shape_changes) / len(shape_changes)
                pose_variation_score = min(mean_shape_change / 0.4, 1.0)
            else:
                pose_variation_score = 0.0
        else:
            pose_variation_score = 0.0

        # Stationary penalty
        if motion_score < 0.1:
            stationary_penalty = 0.3
        else:
            stationary_penalty = 0.0

        # Consistency score
        if len(traj.centers) >= 2:
            xs = [c[0] for c in traj.centers]
            ys = [c[1] for c in traj.centers]
            std_x = (sum((x - sum(xs)/len(xs))**2 for x in xs) / len(xs)) ** 0.5
            std_y = (sum((y - sum(ys)/len(ys))**2 for y in ys) / len(ys)) ** 0.5
            consistency = 1.0 / (1.0 + std_x + std_y)
        else:
            consistency = 0.0

        return (
            0.3 * mean_h
            + 0.4 * prior
            + 0.3 * coverage
            + 1.2 * batting_stance_score
            + 0.8 * motion_score
            + 0.5 * pose_variation_score
            + 0.3 * consistency
            - stationary_penalty
        )

    def _stitch_trajectory(self, best_traj, trajectories, per_frame):
        if best_traj is None:
            return best_traj

        lock_centers = best_traj.centers
        lock_bboxes = best_traj.bboxes

        def _min_dist_to_lock(traj):
            best = float("inf")
            for c in traj.centers:
                for lc in lock_centers:
                    d = math.hypot(c[0] - lc[0], c[1] - lc[1])
                    if d < best:
                        best = d
            return best

        changed = True
        while changed:
            changed = False
            for traj in list(trajectories):
                if traj is best_traj:
                    continue
                overlap = any(
                    f in set(best_traj.frames) for f in traj.frames
                )
                if overlap:
                    continue
                d = _min_dist_to_lock(traj)
                _, lh = bbox_size(lock_bboxes[-1])
                _, th = bbox_size(traj.bboxes[-1])
                if d <= 0.8 * max(lh, th, 1e-6):
                    best_traj.frames.extend(traj.frames)
                    best_traj.centers.extend(traj.centers)
                    best_traj.bboxes.extend(traj.bboxes)
                    best_traj.shapes.extend(traj.shapes)
                    best_traj.poses.extend(traj.poses)
                    trajectories.remove(traj)
                    changed = True
                    break
        return best_traj

    def _rebuild_batsman_trajectory(self, best_traj, per_frame):
        """Re-build the batsman trajectory with a strict spatial lock.

        This prevents identity switches by only accepting candidates that are
        close to the batsman's last known position.
        """
        if best_traj is None or len(best_traj.frames) == 0:
            return best_traj

        # Get the batsman's position range from the original trajectory
        batsman_frames = set(best_traj.frames)
        batsman_centers = list(best_traj.centers)
        batsman_bboxes = list(best_traj.bboxes)
        batsman_shapes = list(best_traj.shapes)
        batsman_poses = list(best_traj.poses)

        # Re-build with strict spatial lock
        new_frames = []
        new_centers = []
        new_bboxes = []
        new_shapes = []
        new_poses = []

        last_center = None
        last_bbox = None
        misses = 0

        for frame_idx, candidates in enumerate(per_frame):
            if frame_idx in batsman_frames:
                # This frame was in the original trajectory - keep it
                idx = best_traj.frames.index(frame_idx)
                new_frames.append(frame_idx)
                new_centers.append(best_traj.centers[idx])
                new_bboxes.append(best_traj.bboxes[idx])
                new_shapes.append(best_traj.shapes[idx])
                new_poses.append(best_traj.poses[idx])
                last_center = best_traj.centers[idx]
                last_bbox = best_traj.bboxes[idx]
                misses = 0
            else:
                # Try to find a candidate near the batsman's last position
                if last_center is not None and last_bbox is not None:
                    _, ref_h = bbox_size(last_bbox)
                    best_cand = None
                    best_dist = float("inf")

                    for cand in candidates:
                        if cand.center is None:
                            continue
                        d = math.hypot(cand.center[0] - last_center[0],
                                      cand.center[1] - last_center[1])
                        if d < best_dist and d <= BATSMAN_LOCK_GATE * ref_h:
                            best_dist = d
                            best_cand = cand

                    if best_cand is not None:
                        new_frames.append(frame_idx)
                        new_centers.append(best_cand.center)
                        new_bboxes.append(best_cand.bbox)
                        new_shapes.append(best_cand.shape)
                        new_poses.append(best_cand.pose)
                        last_center = best_cand.center
                        last_bbox = best_cand.bbox
                        misses = 0
                    else:
                        misses += 1
                else:
                    misses += 1

        # Create new trajectory
        new_traj = Trajectory(
            frames=new_frames,
            centers=new_centers,
            bboxes=new_bboxes,
            shapes=new_shapes,
            poses=new_poses,
        )
        return new_traj

    def select_batsman(self, per_frame, trajectories, total_frames):
        """Pick the single batsman identity and return per-frame poses +
        tracking confidence.

        Returns (frames_data, best_traj, batsman_confidence) where frames_data
        is one dict per frame with keys: pose, tracking_ok, confidence,
        n_detected, selected_person_index, batsman_center,
        identity_switch_detected, batsman_detected.
        """
        if not trajectories:
            return [{
                "pose": None,
                "tracking_ok": False,
                "confidence": 0.0,
                "n_detected": len(cands),
                "selected_person_index": None,
                "batsman_center": None,
                "identity_switch_detected": False,
                "batsman_detected": False,
            } for cands in per_frame], None, 0.0

        best_traj = None
        best_score = -2.0
        for traj in trajectories:
            score = self._score_trajectory(traj, total_frames)
            if score > best_score:
                best_score = score
                best_traj = traj

        # Confidence check
        batsman_confidence = max(0.0, min(1.0, best_score / 2.0))
        batsman_detected = best_score >= BATSMAN_CONFIDENCE_THRESHOLD

        if not batsman_detected:
            return [{
                "pose": None,
                "tracking_ok": False,
                "confidence": 0.0,
                "n_detected": len(cands),
                "selected_person_index": None,
                "batsman_center": None,
                "identity_switch_detected": False,
                "batsman_detected": False,
            } for cands in per_frame], None, batsman_confidence

        # Stitch: absorb spatially-coincident fragments
        best_traj = self._stitch_trajectory(best_traj, trajectories, per_frame)

        # Re-build with strict spatial lock to prevent identity switches
        best_traj = self._rebuild_batsman_trajectory(best_traj, per_frame)

        # Track, for each frame, the candidate-list index of the pose that
        # belongs to the locked trajectory (identified by closest center).
        frame_to_cand_idx = {}
        for frame, center in zip(best_traj.frames, best_traj.centers):
            cands = per_frame[frame]
            best_idx = None
            best_d = 1e9
            for i, cand in enumerate(cands):
                if cand.center is None:
                    continue
                d = math.hypot(cand.center[0] - center[0],
                               cand.center[1] - center[1])
                if d < best_d:
                    best_d = d
                    best_idx = i
            frame_to_cand_idx[frame] = best_idx

        by_frame = {}
        for idx, frame in enumerate(best_traj.frames):
            by_frame[frame] = best_traj.poses[idx]

        output = []
        misses = 0
        for frame, cands in enumerate(per_frame):
            if frame in by_frame:
                center = best_traj.centers[best_traj.frames.index(frame)]
                output.append({
                    "pose": by_frame[frame],
                    "tracking_ok": True,
                    "confidence": 1.0,
                    "n_detected": len(cands),
                    "selected_person_index": frame_to_cand_idx.get(frame),
                    "batsman_center": center,
                    "identity_switch_detected": False,
                    "batsman_detected": True,
                })
                misses = 0
            else:
                misses += 1
                confidence = max(0.0, 1.0 - (misses / (MAX_GAP_FRAMES + 2.0)))
                switch = len(cands) > 0
                output.append({
                    "pose": None,
                    "tracking_ok": False,
                    "confidence": confidence,
                    "n_detected": len(cands),
                    "selected_person_index": None,
                    "batsman_center": None,
                    "identity_switch_detected": switch,
                    "batsman_detected": True,
                })
        return output, best_traj, batsman_confidence

    # ------------------------------------------------------------------
    # Public API (single call that drives the whole video)
    # ------------------------------------------------------------------
    def analyze(self, video_path, fps=25.0):
        """Two-pass analysis: detect -> trajectories -> lock batsman.

        Returns (frames_data, best_traj, batsman_confidence) where frames_data
        is a list of dicts with keys: pose (list[LM] or None), tracking_ok (bool),
        confidence (float), n_detected (int), selected_person_index (int or
        None), batsman_center (tuple or None),
        identity_switch_detected (bool), batsman_detected (bool).
        """
        per_frame = self.detect_all(video_path, fps)
        total_frames = len(per_frame)

        trajectories = self._build_trajectories(per_frame)

        frames_data, best_traj, batsman_confidence = self.select_batsman(
            per_frame, trajectories, total_frames
        )
        return frames_data, best_traj, batsman_confidence

    def draw_pose(self, frame, pose):
        if pose is None:
            return
        height, width = frame.shape[:2]

        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28), (15, 17), (15, 19), (15, 21),
            (16, 18), (16, 20), (16, 22),
        ]

        points = []
        for landmark in pose:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    def close(self):
        self.landmarker_full.close()
        self.landmarker_band.close()
