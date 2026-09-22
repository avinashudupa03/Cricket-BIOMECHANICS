"""Pose extraction with stable, single-identity batsman tracking.

Design
------
This module deliberately avoids picking a *new* person on every frame, which
is what the previous crop-based tracker (fixed crop + hardcoded initial
target) did and which allowed the skeleton to jump to the wicketkeeper or
bowler. Instead it uses a two-pass approach over the whole video:

  Pass 1 - Detect every frame (FULL frame + central band, VIDEO mode).
           Candidates are linked into *person trajectories* using a greedy
           temporal association that scores spatial distance from a predicted
           (velocity-adjusted) position, bounding-box size similarity and
           normalized pose-shape similarity. Short occlusion gaps are bridged
           by prediction, and trajectories that are spatially continuous are
           merged, so a brief disappearance does NOT switch identity.

  Pass 2 - Select the batsman trajectory.
           The batsman is the most persistent, full-body, central person
           (the main subject of a batting clip). Once chosen, every frame is
           assigned to the SAME identity for the whole video. If the chosen
           trajectory has no pose on a frame, that frame is flagged with
           tracking_ok = False (TRACKING UNCERTAIN) and the skeleton is NEVER
           silently moved onto the keeper, non-striker or bowler.

Per-frame output includes tracking confidence, the number of detected people
(debug/after-the-fact oversight), the index of the selected person within that
frame's candidate list, and an honest identity_switch_detected flag: it is
True only for frames where the lock was withheld while *other* people were
still detected (a lock break), never for frames where a different person was
silently substituted.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import mediapipe as mp

# ---------------------------------------------------------------------------
# MediaPipe configuration (deliberately accessible / tunable)
# ---------------------------------------------------------------------------
NUM_POSES = 6
# Kept low so the batsman IS detected while crouched at stance / small in
# frame; trajectory + identity-selection machinery (not raw detection) is
# what defends against wrong-person switching.
MIN_POSE_DETECTION_CONFIDENCE = 0.20
MIN_POSE_PRESENCE_CONFIDENCE = 0.20
MIN_TRACKING_CONFIDENCE = 0.20

# Trajectory association parameters
MAX_GAP_FRAMES = 4          # bridge detection gaps within a trajectory
MAX_MERGE_GAP_FRAMES = 6    # merge spatially-continuous trajectories across longer gaps
POS_GATE = 1.3              # max allowed center displacement (in body heights) per frame
ACCEPT_SCORE = 1.1          # association score threshold

FULL_BODY_HEIGHT = 0.12     # min bbox height to consider a candidate "full body";
                            # low enough that the batsman's walk-in (small) and
                            # crease (large) segments both form trajectories so
                            # they merge into ONE identity, high enough to reject
                            # horizontal/partial crops and blob noise


@dataclass
class LM:
    """Lightweight landmark (matches the MediaPipe PoseLandmark fields used)."""
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
    """Normalized pose shape: joint offsets from center, scaled by bbox height.

    Scale-invariant and translation-invariant, so it describes the *posture*
    of a person rather than their absolute position.
    """
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
    """Soft preference for the central strike-zone position of a batting clip.

    Kept as a weak prior (0..1) - it never hard-excludes off-centre players.
    """
    if center is None:
        return 0.0
    x, y = center
    dx = (x - 0.50) / 0.18
    dy = (y - 0.34) / 0.15
    return math.exp(-0.5 * (dx * dx + dy * dy))


class PoseExtractor:

    # Central band crop used as a second detection stream. It up-scales the
    # batting region (empirically improves detection for small players) while
    # keeping the full body height so legs are never cut off.
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

        # Two independent VIDEO-mode landmarkers (each needs strictly
        # increasing timestamps on its own call stream).
        self.landmarker_full = PoseLandmarker.create_from_options(
            _make_options(VisionRunningMode.VIDEO)
        )
        self.landmarker_band = PoseLandmarker.create_from_options(
            _make_options(VisionRunningMode.VIDEO)
        )

    # ------------------------------------------------------------------
    # Detection (Pass 1)
    # ------------------------------------------------------------------
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
        """Detect within the central band crop (full height) and map the
        normalized crop coordinates back to full-frame coordinates."""
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
        """Merge candidate lists from the two detection streams.

        Keeps only geometrically *distinct* candidates, preferring the pose
        with the most defined landmarks when the same person is seen twice.
        """
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
                if dx < 0.04 and dy < 0.04:
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
        """Pass 1: run detection on every frame using two merged streams."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        per_frame = []
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int((frame_index / max(fps, 1.0)) * 1000)
            full_cands = self._detect_full(frame, timestamp_ms)
            band_cands = self._detect_band(frame, timestamp_ms)
            candidates = self._merge_candidates(full_cands, band_cands)
            for cand in candidates:
                cand.frame = frame_index
            per_frame.append(candidates)
            frame_index += 1
        cap.release()
        return per_frame

    # ------------------------------------------------------------------
    # Trajectory building
    # ------------------------------------------------------------------
    @staticmethod
    def _candidate_score(cand, pred_center, ref_bbox, ref_shape):
        """Weighted identity-consistency score (lower = more consistent).

        Combines: spatial distance from the predicted position (velocity
        adjusted), bounding-box size similarity and pose-shape similarity.
        Candidates that suddenly appear far away get a heavy penalty.
        """
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

        score = 1.6 * pos_dist + 0.8 * size_dist + 1.2 * shape_dist
        # Strong penalty for suddenly-appearing far-away candidates
        # (keeper / non-striker / crowd), preventing identity switches.
        if pos_dist > POS_GATE:
            score += 8.0
        return score

    def _build_trajectories(self, per_frame):
        """Link per-frame candidates into temporally-consistent trajectories."""
        trajectories = []  # list of Trajectory
        active = []        # parallel list of dicts with motion-model state

        for frame_idx, candidates in enumerate(per_frame):
            used = [False] * len(candidates)

            # 1) Try to continue existing active trajectories.
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

            # 2) Mark trajectories not seen this frame.
            for (t, st) in zip(trajectories, active):
                if st["last"] < frame_idx:
                    st["gap"] += 1
                    dt = st["gap"]
                    st["pred"] = (t.centers[-1][0] + st["vel"][0] * dt,
                                  t.centers[-1][1] + st["vel"][1] * dt)
                    if st["gap"] > MAX_GAP_FRAMES:
                        st["active"] = False

            # 3) Create new trajectories from unmatched full-body candidates.
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

        # 4) Merge spatially-continuous short trajectories (identity recovery).
        merged = self._merge_trajectories(trajectories)
        return merged

    @staticmethod
    def _merge_trajectories(trajectories):
        """Merge two trajectories when the second resumes where the first left
        off (gap <= MAX_MERGE_GAP_FRAMES and small centre displacement), i.e.
        the same person reappeared after a short disappearance."""
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
                    gap_frames = t2.frames[0] - t1.frames[-1]
                    if not (0 < gap_frames <= MAX_MERGE_GAP_FRAMES):
                        continue
                    c1 = t1.centers[-1]
                    c2 = t2.centers[0]
                    _, h1 = bbox_size(t1.bboxes[-1])
                    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if dist <= 1.5 * max(h1, 1e-6):
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
        """Score a trajectory as the batsman. Main subject = persists, full
        body, central. Returns a 0..~1.5 score (higher = better candidate)."""
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
        return 0.9 * mean_h + 0.7 * prior + 0.9 * coverage

    def select_batsman(self, per_frame, trajectories, total_frames):
        """Pick the single batsman identity and return per-frame poses +
        tracking confidence.

        Returns (frames_data, best_traj) where frames_data is one dict per
        frame with keys: pose, tracking_ok, confidence, n_detected,
        selected_person_index, batsman_center, identity_switch_detected.
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
            } for cands in per_frame], None

        best_traj = None
        best_score = -2.0
        for traj in trajectories:
            score = self._score_trajectory(traj, total_frames)
            if score > best_score:
                best_score = score
                best_traj = traj

        # Track, for each frame, the candidate-list index of the pose that
        # belongs to the locked trajectory (identified by closest center).
        frame_to_cand_idx = {}
        for traj_idx, (frame, center) in enumerate(zip(best_traj.frames,
                                                       best_traj.centers)):
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
                })
                misses = 0
            else:
                misses += 1
                confidence = max(0.0, 1.0 - (misses / (MAX_GAP_FRAMES + 2.0)))
                # Honest lock-break flag: other people were detected but the
                # lock was defended rather than silently re-anchored.
                switch = len(cands) > 0
                output.append({
                    "pose": None,
                    "tracking_ok": False,
                    "confidence": confidence,
                    "n_detected": len(cands),
                    "selected_person_index": None,
                    "batsman_center": None,
                    "identity_switch_detected": switch,
                })
        return output, best_traj

    # ------------------------------------------------------------------
    # Public API (single call that drives the whole video)
    # ------------------------------------------------------------------
    def analyze(self, video_path, fps=25.0):
        """Two-pass analysis: detect -> trajectories -> lock batsman.

        Returns (frames_data, best_traj) where frames_data is a list of dicts
        with keys: pose (list[LM] or None), tracking_ok (bool),
        confidence (float), n_detected (int), selected_person_index (int or
        None), batsman_center (tuple or None),
        identity_switch_detected (bool).
        """
        per_frame = self.detect_all(video_path, fps)
        total_frames = len(per_frame)

        trajectories = self._build_trajectories(per_frame)

        frames_data, best_traj = self.select_batsman(
            per_frame, trajectories, total_frames
        )
        return frames_data, best_traj

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