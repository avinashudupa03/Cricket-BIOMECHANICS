"""Shot rating engine for cricket batting analysis.

This module converts raw biomechanical shot-analysis data (the metrics
produced by ``biomechanics_analyzer.py``) into a meaningful, repeatable
0-10 rating for a single batting shot.

Design goals
------------
* Deterministic - identical input always yields an identical rating, so the
  same quality of shot scores approximately the same every time.
* Reusable - the entry point :func:`rate_shot` is a pure function over plain
  data, so it can be dropped into the Flask app, a CLI, or a batch pipeline.
* Missing-data aware - if a biomechanical metric is unavailable it is NOT
  invented; the factor is reported as unavailable and its weight is
  redistributed among the factors that do have data, while the confidence
  score reflects how much of the input was trustworthy.

Factors & weights
-----------------
The twelve factors below are evaluated on a 0-10 scale. The weights encode the
importance of each factor to overall shot quality; timing, contact quality,
balance, bat angle and footwork are weighted highest, while less critical
factors (power, follow-through, direction) carry less weight.

======  ===========================================  ======
Code    Factor                                        Weight
======  ===========================================  ======
swing       Bat swing & bat speed                     0.09
bat_angle   Bat angle at point of contact            0.11
timing      Shot timing                              0.15
backlift    Bat lift (backlift) control              0.06
contact     Quality of ball contact                  0.12
power       Power generated                          0.05
follow      Follow-through & overall technique       0.06
technique   Body position & stability                0.06
balance     Head position & balance                  0.10
footwork    Footwork & weight transfer               0.10
stride      Stride & forward movement                0.07
direction   Shot direction & placement               0.03
======  ===========================================  ======

Total weight = 1.0.

Source data
-----------
Factors fall into two tiers by data availability:

* **Coarse features** - every factor except balance/stride/backlift can be
  scored from a single row of ``biomechanics_features.csv`` produced by
  ``biomechanics_analyzer.py``.
* **Per-frame ``extra`` data** - ``backlift`` (bat-lift control),
  ``balance`` (head steadiness) and ``stride`` (forward stride / weight
  transfer) need per-frame landmark data. Use :func:`build_extra` to assemble
  the ``extra`` dict from the processed-video folder, or pass it manually.

When a factor's source data is unavailable it is NOT invented; its weight is
re-distributed among the factors that do have data (see
:func:`_effective_weights`), and the confidence score reflects how much of the
input was trustworthy. Direction/placement likewise stays unavailable unless a
``placement_score`` is supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------

# Ordered list of (code, display name, weight, short description).
FACTOR_DEFS: Tuple[Tuple[str, str, float, str], ...] = (
    ("swing", "Bat Swing & Bat Speed", 0.09,
     "Swing velocity measured from downswing duration and explosive movement."),
    ("bat_angle", "Bat Angle at Contact", 0.11,
     "Bat-face orientation approximated from the bat-arm elbow angle at impact."),
    ("timing", "Shot Timing", 0.15,
     "How cleanly the swing peaks at the moment of ball contact."),
    ("backlift", "Bat Lift Control", 0.06,
     "Height and steadiness of the bat lift during the backlift phase."),
    ("contact", "Quality of Ball Contact", 0.12,
     "Smoothness and control at the instant of impact."),
    ("power", "Power Generated", 0.05,
     "Kinetic output and momentum of the shot."),
    ("follow", "Follow-through & Technique", 0.06,
     "Control through the follow-through phase."),
    ("technique", "Body Position & Stability", 0.06,
     "Stability and symmetry of the body through the swing."),
    ("balance", "Head Position & Balance", 0.10,
     "Head stillness and balance through the swing."),
    ("footwork", "Footwork & Weight Transfer", 0.10,
     "Knee/hip mobility and transfer of weight through the shot."),
    ("stride", "Stride & Forward Movement", 0.07,
     "Forward stride length and back-foot stability through the shot."),
    ("direction", "Shot Direction & Placement", 0.03,
     "Accuracy of ball placement and shot direction."),
)

FACTOR_CODES: List[str] = [f[0] for f in FACTOR_DEFS]
FACTOR_LABELS: Dict[str, str] = {f[0]: f[1] for f in FACTOR_DEFS}
FACTOR_WEIGHTS: Dict[str, float] = {f[0]: f[2] for f in FACTOR_DEFS}
FACTOR_DESCRIPTIONS: Dict[str, str] = {f[0]: f[3] for f in FACTOR_DEFS}

# The weights must sum to 1.0 for the weighted average to be a valid 0-10.
assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-9, (
    "Factor weights must sum to 1.0"
)

# ---------------------------------------------------------------------------
# Classification bands
# ---------------------------------------------------------------------------
CLASSIFICATION: Tuple[Tuple[float, float, str], ...] = (
    (9.0, 10.01, "Excellent"),
    (8.0, 8.99, "Very Good"),
    (7.0, 7.99, "Good"),
    (6.0, 6.99, "Average"),
    (5.0, 5.99, "Below Average"),
    (0.0, 4.99, "Poor"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    """Clamp a value to the [low, high] range."""
    return max(low, min(high, value))


def _num(value, default: Optional[float] = None) -> Optional[float]:
    """Coerce a value to a float, returning ``default`` when not usable."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _norm_linear(value: Optional[float], good: float, poor: float) -> Optional[float]:
    """Map a value to 0-10 linearly within [poor, good].

    ``good`` scores 10, ``poor`` scores 0; beyond the ends clamps.
    """
    if value is None:
        return None
    if abs(good - poor) < 1e-9:
        return 10.0 if value >= good else 0.0
    score = 10.0 * (value - poor) / (good - poor)
    return _clamp(score)


def _norm_inverse(value: Optional[float], good: float, poor: float) -> Optional[float]:
    """Like :func:`_norm_linear` but decreasing (small values score higher)."""
    if value is None:
        return None
    if abs(good - poor) < 1e-9:
        return 0.0 if value >= good else 10.0
    score = 10.0 * (good - value) / (good - poor)
    return _clamp(score)


def _bell(value: Optional[float], centre: float, half_span: float) -> Optional[float]:
    """Score closeness to ``centre`` on a 0-10 scale.

    At ``centre`` the score is 10; at ``centre +/- half_span`` it is 0.
    """
    if value is None:
        return None
    return _clamp(10.0 * (1.0 - abs(value - centre) / half_span))


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _control_side(f: Dict) -> str:
    """Return the batsman's top/control hand side ("left" or "right").

    For a right-handed batsman the left arm is the top/control arm and is more
    extended (higher elbow angle) at the moment of impact; for a left-handed
    batsman the mirror is true. Detecting this from the actual elbow angles at
    impact keeps the bat-angle and contact scores correct regardless of
    batting stance, instead of blindly assuming the right arm handles the bat.
    Falls back to "right" when the signal is unavailable.
    """
    left = _num(f.get("left_elbow_angle_at_impact"))
    right = _num(f.get("right_elbow_angle_at_impact"))
    if left is None or right is None:
        return "right"
    return "left" if left >= right else "right"


def _symmetry(left: Optional[float], right: Optional[float],
              good=20.0) -> Optional[float]:
    """Score how similar left/right are (small difference scores high)."""
    if left is None or right is None:
        return None
    diff = abs(left - right)
    return _norm_inverse(diff, good, 0.0)


# ---------------------------------------------------------------------------
# Per-factor scoring
# ---------------------------------------------------------------------------
def _score_swing(f: Dict) -> Optional[float]:
    """Bat swing & bat speed - faster, more explosive swing scores higher."""
    downswing_ms = _num(f.get("downswing_duration_ms"))
    max_movement = _num(f.get("maximum_movement_score"))

    components: List[float] = []
    # Shorter downswing => quicker swing. Typical 150-450 ms.
    speed = _norm_inverse(downswing_ms, 450.0, 150.0)
    if speed is not None:
        components.append(0.65 * speed)
    # Higher peak movement => more explosive.
    if max_movement is not None:
        movement = _clamp(10.0 * (max_movement - 4.0) / 12.0)
        components.append(0.35 * movement)

    if not components:
        return None
    return _clamp(sum(components))


def _score_bat_angle(f: Dict) -> Optional[float]:
    """Bat angle at contact - control-arm elbow near a good operating zone.

    Uses the detected top/control arm (see :func:`_control_side`), which is
    the arm whose extension governs bat-face orientation at impact.
    """
    side = _control_side(f)
    control_elbow = _num(f.get(f"{side}_elbow_angle_at_impact"))
    if control_elbow is None:
        return None
    # A well-extended control arm at impact (~150-170 deg) indicates a sound
    # bat angle and a full, attacking swing rather than a cramped one.
    return _bell(control_elbow, 160.0, 45.0)


def _score_timing(f: Dict) -> Optional[float]:
    """Timing - clean, brief, peak-aligned impact scores higher."""
    impact_frames = _num(f.get("impact_frames"))
    max_movement = _num(f.get("maximum_movement_score"))
    avg_movement = _num(f.get("average_movement_score"))

    components: List[float] = []
    # Few impact frames => crisp single contact point.
    if impact_frames is not None:
        crisp = _norm_inverse(impact_frames, 8.0, 1.0)
        if crisp is not None:
            components.append(0.6 * crisp)
    # Sharp peak-to-average movement => well-timed explosive contact.
    if max_movement is not None and avg_movement and avg_movement > 0:
        ratio = max_movement / avg_movement
        components.append(0.4 * _norm_linear(ratio, 4.0, 1.0))

    if not components:
        return None
    return _clamp(sum(components))


def _score_body_stability(f: Dict) -> Optional[float]:
    """Body position & stability - symmetry and steadiness of the body."""
    scores: List[float] = []

    knee_sym = _symmetry(
        _num(f.get("left_knee_angle_at_impact")),
        _num(f.get("right_knee_angle_at_impact")), good=25.0)
    if knee_sym is not None:
        scores.append(knee_sym)

    hip_sym = _symmetry(
        _num(f.get("left_hip_angle_at_impact")),
        _num(f.get("right_hip_angle_at_impact")), good=25.0)
    if hip_sym is not None:
        scores.append(hip_sym)

    shoulder_sym = _symmetry(
        _num(f.get("left_shoulder_angle_at_impact")),
        _num(f.get("right_shoulder_angle_at_impact")), good=35.0)
    if shoulder_sym is not None:
        scores.append(shoulder_sym)

    # Low knee std through the swing => stable lower body.
    knee_std = _num(f.get("right_knee_angle_std"))
    if knee_std is not None:
        scores.append(_norm_inverse(knee_std, 40.0, 8.0))

    if not scores:
        return None
    return _clamp(sum(scores) / len(scores))


def _score_balance(f: Dict, extra: Optional[Dict] = None) -> Optional[float]:
    """Head position & balance - HEAD vs shoulders, not raw global position.

    Prefers per-frame torso-relative head offsets from ``extra["balance_data"]``
    (nose minus shoulder-midline), which cancel whole-body translation/crouch
    and isolate genuine head wobble. Falls back to absolute nose tracks
    (``extra["head_tracks"]``) when relative data is unavailable.
    """
    if extra is None:
        return None

    tracks = extra.get("balance_data") or extra.get("head_tracks")
    if not tracks:
        return None

    xs = [_num(v.get("x")) for v in tracks]
    ys = [_num(v.get("y")) for v in tracks]
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if len(xs) < 3 or len(ys) < 3:
        return None

    # A still head moves little relative to the shoulders. Vertical bob is
    # down-weighted because a crouch/straighten through the swing is normal.
    x_std = _stats_std(xs)
    y_std = _stats_std(ys)
    if extra.get("balance_data"):
        return _norm_inverse(x_std + 0.5 * y_std, 0.005, 0.03)
    return _norm_inverse(x_std + y_std, 0.05, 0.005)


def _score_footwork(f: Dict) -> Optional[float]:
    """Footwork & weight transfer - knee/hip mobility and symmetry."""
    scores: List[float] = []

    # Good leg mobility (proper knee bend) is indicated by a healthy knee range.
    left_knee_range = _range_of(f, "left", "knee")
    right_knee_range = _range_of(f, "right", "knee")
    knee_range = _mean([left_knee_range, right_knee_range])
    if knee_range is not None:
        scores.append(_norm_linear(knee_range, 90.0, 20.0))

    # Symmetric leg loading contributes to stable footwork.
    knee_sym = _symmetry(left_knee_range, right_knee_range, good=20.0)
    hip_sym = _symmetry(
        _range_of(f, "left", "hip"), _range_of(f, "right", "hip"), good=20.0)
    if knee_sym is not None:
        scores.append(knee_sym)
    if hip_sym is not None:
        scores.append(hip_sym)

    if not scores:
        return None
    return _clamp(sum(scores) / len(scores))


def _score_contact(f: Dict) -> Optional[float]:
    """Quality of ball contact - control at impact from the upper body."""
    scores: List[float] = []
    side = _control_side(f)

    # Control-arm (top hand) elbow should be smoothly controlled near full
    # extension at the moment of impact for a clean strike.
    elbow_impact = _num(f.get(f"{side}_elbow_angle_at_impact"))
    if elbow_impact is not None:
        scores.append(_bell(elbow_impact, 160.0, 50.0))

    # Control-arm shoulder angle stability at impact.
    shoulder_impact = _num(f.get(f"{side}_shoulder_angle_at_impact"))
    if shoulder_impact is not None:
        scores.append(_bell(shoulder_impact, 65.0, 45.0))

    # Low elbow variance through the swing indicates a controlled strike.
    elbow_std = _num(f.get(f"{side}_elbow_angle_std"))
    if elbow_std is not None:
        scores.append(_norm_inverse(elbow_std, 45.0, 10.0))

    if not scores:
        return None
    return _clamp(sum(scores) / len(scores))


def _score_backlift(f: Dict, extra: Optional[Dict] = None) -> Optional[float]:
    """Bat lift (backlift) control - height and steadiness of the bat lift.

    Requires ``extra["backlift"]`` assembled by :func:`build_extra`: a list of
    per-frame lift ratios ``(shoulder_y - wrist_y) / torso_len`` observed during
    the Backlift phase. A ratio of 1.0 means the hands were lifted one
    torso-length above the shoulder (a full, classical backlift).
    """
    if extra is None:
        return None
    bl = extra.get("backlift")
    if not bl:
        return None

    ratios = [_num(v) for v in bl.get("ratios", [])]
    ratios = [v for v in ratios if v is not None]
    if len(ratios) < 3:
        return None

    mean_lift = sum(ratios) / len(ratios)
    # Ideal lift sits around a full torso-length above the shoulder.
    lift = _bell(mean_lift, 1.0, 1.0)
    # A steady lift (low spread) shows control; a flailing bat drops the score.
    control = _norm_inverse(_stats_std(ratios), 0.5, 0.1)

    if lift is None:
        return None
    score = 0.6 * lift
    if control is not None:
        score += 0.4 * control
    return _clamp(score)


def _score_direction(f: Dict, extra: Optional[Dict] = None) -> Optional[float]:
    """Shot direction & placement.

    Requires ball/bat trajectory or a bat-path signal, which the pose-only
    feature set does not reliably provide. Unavailable unless supplied.
    """
    if extra is None or not extra.get("placement_score"):
        return None
    return _clamp(_num(extra.get("placement_score")))


def _score_power(f: Dict) -> Optional[float]:
    """Power generated - kinetic output and momentum."""
    scores: List[float] = []

    max_movement = _num(f.get("maximum_movement_score"))
    if max_movement is not None:
        scores.append(_clamp(10.0 * (max_movement - 4.0) / 14.0))

    # A longer follow-through indicates more momentum carried through.
    follow_ms = _num(f.get("followthrough_duration_ms"))
    if follow_ms is not None:
        scores.append(_norm_linear(follow_ms, 1500.0, 300.0))

    arm_mobility = _mean([
        _range_of(f, "left", "elbow"), _range_of(f, "right", "elbow"),
        _range_of(f, "left", "shoulder"), _range_of(f, "right", "shoulder"),
    ])
    if arm_mobility is not None:
        scores.append(_norm_linear(arm_mobility, 120.0, 30.0))

    if not scores:
        return None
    return _clamp(sum(scores) / len(scores))


def _score_follow(f: Dict) -> Optional[float]:
    """Follow-through & overall technique."""
    scores: List[float] = []

    follow_ms = _num(f.get("followthrough_duration_ms"))
    if follow_ms is not None:
        # A follow-through that is neither abrupt nor unnaturally long.
        scores.append(_bell(follow_ms, 900.0, 700.0))

    downswing_ms = _num(f.get("downswing_duration_ms"))
    if follow_ms and downswing_ms and downswing_ms > 0:
        ratio = follow_ms / downswing_ms
        scores.append(_norm_linear(ratio, 2.5, 0.5))

    # Smooth full-arm technique reflected in shoulder mobility.
    shoulder_mob = _mean([_range_of(f, "left", "shoulder"),
                          _range_of(f, "right", "shoulder")])
    if shoulder_mob is not None:
        scores.append(_norm_linear(shoulder_mob, 90.0, 20.0))

    if not scores:
        return None
    return _clamp(sum(scores) / len(scores))


def _score_stride(f: Dict, extra: Optional[Dict] = None) -> Optional[float]:
    """Stride & forward movement - front-foot stride vs. back-foot stability.

    Requires ``extra["stride"]`` from :func:`build_extra`: the front-foot
    displacement during the stride (as a multiple of torso length, so it is
    robust to subject size and camera distance) and the back-foot displacement
    (which should stay small for a well-balanced drive).
    """
    if extra is None:
        return None
    st = extra.get("stride")
    if not st:
        return None

    front = _num(st.get("front_ratio"))
    back = _num(st.get("back_ratio"))

    scored = 0.0
    used = 0.0
    if front is not None:
        # Balanced stride: roughly 0.5-2.7 torso-lengths of forward travel.
        stride = _bell(front, 1.5, 1.2)
        if stride is not None:
            scored += 0.7 * stride
            used += 0.7
    if back is not None:
        # The back foot should stay planted (small displacement scores high).
        control = _norm_inverse(back, 0.5, 0.05)
        if control is not None:
            scored += 0.3 * control
            used += 0.3

    if used <= 0:
        return None
    return _clamp(scored / used)


def _range_of(f: Dict, side: str, joint: str) -> Optional[float]:
    """Return the min-max angle range for a joint/side, if present."""
    lo = _num(f.get(f"{side}_{joint}_angle_min"))
    hi = _num(f.get(f"{side}_{joint}_angle_max"))
    if lo is None or hi is None:
        return None
    return hi - lo


def _stats_std(values: Sequence[float]) -> float:
    """Population standard deviation of a numeric sequence."""
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


_FACTOR_FUNCS = {
    "swing": lambda f, extra=None: _score_swing(f),
    "bat_angle": lambda f, extra=None: _score_bat_angle(f),
    "timing": lambda f, extra=None: _score_timing(f),
    "backlift": lambda f, extra=None: _score_backlift(f, extra),
    "contact": lambda f, extra=None: _score_contact(f),
    "power": lambda f, extra=None: _score_power(f),
    "follow": lambda f, extra=None: _score_follow(f),
    "technique": lambda f, extra=None: _score_body_stability(f),
    "balance": lambda f, extra=None: _score_balance(f, extra),
    "footwork": lambda f, extra=None: _score_footwork(f),
    "stride": lambda f, extra=None: _score_stride(f, extra),
    "direction": lambda f, extra=None: _score_direction(f, extra),
}


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------
@dataclass
class FactorScore:
    code: str
    label: str
    weight: float
    raw: float
    score: Optional[float]      # None => unavailable
    weighted: Optional[float]   # None => unavailable
    comment: str

    @property
    def available(self) -> bool:
        return self.score is not None


@dataclass
class ShotRating:
    shot_type: str = "unknown"
    rating: float = 0.0
    classification: str = "Poor"
    confidence: float = 0.0          # 0..1 reliability of the rating
    covered_weight: float = 0.0      # fraction of total factor weight with data
    factors: List[FactorScore] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    explanation: str = ""
    unavailable_metrics: List[str] = field(default_factory=list)

    @property
    def missing_metrics(self) -> List[str]:
        return [f.label for f in self.factors if not f.available]


# ---------------------------------------------------------------------------
# Effective weights (redistribute missing-factor weight)
# ---------------------------------------------------------------------------
def _effective_weights(available_codes: Sequence[str]) -> Dict[str, float]:
    """Return a normalised weight map over the available factor codes.

    When some factors are unavailable their weight is redistributed pro-rata
    across the factors that do have data, so the weighted average remains on
    the 0-10 scale. If ALL factors are unavailable an empty map is returned.
    """
    avail = [c for c in FACTOR_CODES if c in set(available_codes)]
    if not avail:
        return {}
    total_avail = sum(FACTOR_WEIGHTS[c] for c in avail)
    if total_avail <= 0:
        return {}
    return {c: FACTOR_WEIGHTS[c] / total_avail for c in avail}


# ---------------------------------------------------------------------------
# Building the per-frame "extra" dict from a processed-video folder
# ---------------------------------------------------------------------------
def build_extra(video_name: str,
                folder: Optional[Path] = None) -> Dict:
    """Assemble the optional per-frame ``extra`` data for a processed video.

    Reads ``batting_landmarks.csv`` (3D landmark positions per frame) and
    ``batting_phases.csv`` (phase windows) from the per-video output folder and
    returns the ``extra`` dict accepted by :func:`rate_shot`. This activates
    the factors that cannot be scored from a single coarse feature row:

    * ``head_tracks`` - powers *Head Position & Balance* (nose x/y per frame).
    * ``balance_data`` - torso-relative head offsets (nose minus shoulder
      midline), the preferred signal for *Head Position & Balance*.
    * ``backlift``    - powers *Bat Lift Control* (hands-y vs shoulder-y per
      frame inside the Backlift phase, scaled by torso length).
    * ``stride``      - powers *Stride & Forward Movement* (front/back ankle
      displacement from stance to impact, scaled by torso length).

    Any factor whose source file or phase is missing simply stays unavailable -
    nothing is ever guessed.
    """
    if folder is None:
        folder = BASE_DIR / "output_data" / Path(video_name).stem
    folder = Path(folder)

    extra: Dict = {}

    landmarks_file = folder / "batting_landmarks.csv"
    phases_file = folder / "batting_phases.csv"
    if not landmarks_file.exists():
        return extra

    import pandas as pd

    try:
        landmarks = pd.read_csv(landmarks_file)
    except Exception:
        return extra

    if len(landmarks) == 0:
        return extra

    # --- Head tracks: axis + torso-relative offsets for balance -----------
    head_tracks = []
    balance_data = []
    for _, r in landmarks.iterrows():
        x = _num(r.get("nose_x"))
        y = _num(r.get("nose_y"))
        if x is not None and y is not None:
            head_tracks.append({"x": x, "y": y})
            lsx = _num(r.get("left_shoulder_x"))
            rsx = _num(r.get("right_shoulder_x"))
            lsy = _num(r.get("left_shoulder_y"))
            rsy = _num(r.get("right_shoulder_y"))
            if None not in (lsx, rsx, lsy, rsy):
                # Head offset from the shoulder midline cancels whole-body
                # translation and isolates genuine head wobble.
                balance_data.append({
                    "x": x - (lsx + rsx) / 2.0,
                    "y": y - (lsy + rsy) / 2.0,
                })
    if len(head_tracks) >= 3:
        extra["head_tracks"] = head_tracks
    if len(balance_data) >= 3:
        extra["balance_data"] = balance_data

    # --- Torso-length proxy for scale-invariant normalisation -------------
    torso_rows = []
    for _, r in landmarks.iterrows():
        sy = _num(r.get("right_shoulder_y"))
        hy = _num(r.get("right_hip_y"))
        if sy is not None and hy is not None:
            torso_rows.append(abs(sy - hy))
    torso_len = _mean(torso_rows) if torso_rows else None
    if torso_len is None or torso_len <= 0:
        return extra

    try:
        phases = pd.read_csv(phases_file)
    except Exception:
        phases = None

    if phases is not None and "phase" in phases.columns and "frame" in phases.columns:
        # --- Bat lift ratios during the Backlift phase --------------------
        backlift_frames = set()
        for _f in phases.loc[phases["phase"] == "Backlift", "frame"]:
            f_ok = _num(_f)
            if f_ok is not None:
                backlift_frames.add(int(f_ok))
        ratios = []
        if backlift_frames:
            for _, r in landmarks.iterrows():
                fr = _num(r.get("frame"))
                if fr is None or int(fr) not in backlift_frames:
                    continue
                sy = _num(r.get("right_shoulder_y"))
                lsy = _num(r.get("left_shoulder_y"))
                if sy is None and lsy is not None:
                    sy = lsy
                if sy is None:
                    continue
                lwy = _num(r.get("left_wrist_y"))
                rwy = _num(r.get("right_wrist_y"))
                # The bat's height is set by the higher (top) hand; using the
                # topmost wrist keeps the lift measure correct regardless of
                # whether the batsman is right- or left-handed.
                top_wrist = None
                if lwy is not None and rwy is not None:
                    top_wrist = min(lwy, rwy)
                elif lwy is not None:
                    top_wrist = lwy
                elif rwy is not None:
                    top_wrist = rwy
                if top_wrist is not None:
                    ratios.append((sy - top_wrist) / torso_len)
        if len(ratios) >= 3:
            extra["backlift"] = {"ratios": ratios, "torso_len": torso_len}

        # --- Stride: ankle travel through the shot -------------------------
        # The feet can leave the frame during the deep stance crouch, so the
        # reference position is the first frame where each foot is tracked
        # inside the Stance phase (falling back to the first tracked frame
        # anywhere). A planted foot contributes its true (~0) displacement so
        # a still back foot is rewarded rather than ignored.
        impact_frames = []
        for _f in phases.loc[phases["phase"] == "Impact", "frame"]:
            f_ok = _num(_f)
            if f_ok is not None:
                impact_frames.append(int(f_ok))
        stance_frames = []
        for _f in phases.loc[phases["phase"] == "Stance", "frame"]:
            f_ok = _num(_f)
            if f_ok is not None:
                stance_frames.append(int(f_ok))
        if impact_frames:
            impact_rows = landmarks[
                landmarks["frame"].astype(float).isin(impact_frames)
            ]

            def _first_valid(col: str) -> Optional[float]:
                if stance_frames:
                    sub = landmarks[
                        landmarks["frame"].astype(float).isin(stance_frames)
                    ][col].dropna()
                    if len(sub):
                        return float(sub.iloc[0])
                s = landmarks[col].dropna()
                return float(s.iloc[0]) if len(s) else None

            def _median_valid(series) -> Optional[float]:
                s = pd.to_numeric(series, errors="coerce").dropna()
                return float(s.median()) if len(s) else None

            displacements = {}
            for side in ("left", "right"):
                base = f"{side}_ankle"
                sx = _first_valid(f"{base}_x")
                sz = _first_valid(f"{base}_z")
                ix = _median_valid(impact_rows[f"{base}_x"])
                iz = _median_valid(impact_rows[f"{base}_z"])
                if sx is None or ix is None:
                    continue
                if sz is not None and iz is not None:
                    dist = math.hypot(ix - sx, iz - sz)
                else:
                    # Depth not tracked; fall back to horizontal travel only.
                    dist = abs(ix - sx)
                displacements[side] = dist

            if displacements:
                front = max(displacements.values())
                back = min(displacements.values())
                extra["stride"] = {
                    "front_ratio": front / torso_len,
                    "back_ratio": back / torso_len,
                }

    return extra


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def rate_shot(features: Dict,
              extra: Optional[Dict] = None,
              shot_type: str = "unknown") -> ShotRating:
    """Rate a single batting shot from its biomechanical features.

    Parameters
    ----------
    features:
        A mapping of biomechanical feature name -> value, typically one row of
        ``biomechanics_features.csv`` produced by ``biomechanics_analyzer.py``.
    extra:
        Optional richer per-frame data that enables factors not derivable from
        the coarse features row. Use :func:`build_extra` to obtain it from a
        processed-video folder. It may contain::

            "head_tracks": [{"x": .., "y": ..}, ...]  # nose position per frame
            "backlift": {"ratios": [..]}              # bat-lift ratios/frame
            "stride": {"front_ratio": .., "back_ratio": ..}
            "placement_score": float                   # 0-10 shot placement

        If omitted (or a key is missing), the corresponding factors are marked
        unavailable. Nothing is ever invented.
    shot_type:
        Optional readable shot type label used for reporting.

    Returns
    -------
    A populated :class:`ShotRating` object.
    """
    if features is None:
        features = {}

    # 1. Score every factor (None marks an unavailable metric - never invented).
    raw_scores: Dict[str, Optional[float]] = {}
    for code in FACTOR_CODES:
        try:
            # All factor callables accept (features, extra) uniformly; a
            # raised TypeError is a real bug and must NOT be masked.
            raw_scores[code] = _FACTOR_FUNCS[code](features, extra)
        except Exception:
            raw_scores[code] = None

    available_codes = [c for c in FACTOR_CODES if raw_scores.get(c) is not None]

    # 2. Re-weight to account for missing data.
    effective_weights = _effective_weights(available_codes)

    # 3. Build factor records + weighted contribution.
    factors: List[FactorScore] = []
    for code in FACTOR_CODES:
        weight = FACTOR_WEIGHTS[code]
        eff = effective_weights.get(code, 0.0)
        score = raw_scores.get(code)
        raw = 0.0
        if score is not None:
            score = _clamp(score)
            raw = score
        weighted = (eff * score) if score is not None else None
        factors.append(FactorScore(
            code=code,
            label=FACTOR_LABELS[code],
            weight=weight,
            raw=raw,
            score=score,
            weighted=weighted,
            comment=_factor_comment(code, score, features, extra),
        ))

    # 4. Weighted final score over available factors only. The rating is the
    #    sum of each available factor's re-weighted contribution.
    uncovered = 1.0 - sum(FACTOR_WEIGHTS[c] for c in available_codes)
    covered = 1.0 - uncovered
    rating = 0.0
    if available_codes:
        rating = _clamp(
            sum(f.weighted for f in factors if f.weighted is not None)
        )

    # 5. Confidence: how reliable the rating is given the amount and quality of
    #    available input data. It is the fraction of total factor weight backed
    #    by real data, reduced further when very few factors contributed (a
    #    single-factor rating is inherently fragile).
    n_avail = len(available_codes)
    diversity = _clamp(n_avail / len(FACTOR_CODES), 0.0, 1.0)
    confidence = _clamp(covered * (0.5 + 0.5 * diversity), 0.0, 1.0)

    classification = _classify(rating)
    unavailable = [f.label for f in factors if not f.available]

    strengths, weaknesses = _build_strength_weakness(factors)

    explanation = _build_explanation(rating, classification, factors,
                                     unavailable, uncovered)

    return ShotRating(
        shot_type=shot_type,
        rating=round(rating, 1),
        classification=classification,
        confidence=round(confidence, 2),
        covered_weight=round(covered, 2),
        factors=factors,
        strengths=strengths,
        weaknesses=weaknesses,
        explanation=explanation,
        unavailable_metrics=unavailable,
    )


# ---------------------------------------------------------------------------
# Quality classification
# ---------------------------------------------------------------------------
def _classify(rating: float) -> str:
    for low, high, label in CLASSIFICATION:
        if low <= rating < high:
            return label
    return "Poor"


# ---------------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------------
def _factor_comment(code: str, score: Optional[float],
                    f: Dict, extra: Optional[Dict]) -> str:
    if score is None:
        return "Metric unavailable - no score assigned (not invented)."
    label = FACTOR_LABELS[code]
    if score >= 8.0:
        return f"Strong {label.lower()} - a real asset to the shot."
    if score >= 6.0:
        return f"Adequate {label.lower()} - acceptable but improvable."
    if score >= 4.0:
        return f"Weak {label.lower()} - detracting from the shot."
    return f"Poor {label.lower()} - a significant weakness."


def _build_strength_weakness(
        factors: List[FactorScore]
) -> Tuple[List[str], List[str]]:
    strengths = [
        f.label for f in factors
        if f.available and f.score is not None and f.score >= 7.0
    ]
    weaknesses = [
        f.label for f in factors
        if f.available and f.score is not None and f.score < 4.0
    ]
    return strengths, weaknesses


def _build_explanation(rating: float, classification: str,
                       factors: List[FactorScore],
                       unavailable: List[str], uncovered: float) -> str:
    parts = [
        f"Overall rating {rating:.1f}/10 ({classification}). "
        f"Score is the weighted average of the assessed factors."
    ]

    top = sorted(
        [f for f in factors if f.available and f.weighted is not None],
        key=lambda f: f.weighted or 0, reverse=True)
    if top:
        best = top[0]
        parts.append(
            f"The strongest contributing factor was {best.label.lower()} "
            f"(score {best.score:.1f}, weighted {best.weighted:.2f}).")
    if len(top) > 1:
        worst = top[-1]
        parts.append(
            f"The weakest contributing factor was {worst.label.lower()} "
            f"(score {worst.score:.1f}).")

    if unavailable:
        parts.append(
            "Metrics without sufficient data were not scored and are listed "
            "as unavailable: " + ", ".join(unavailable) + ".")
    if uncovered > 0.01:
        pct = uncovered * 100
        parts.append(
            f"Their weight ({pct:.0f}%) was redistributed to the factors "
            f"that had data, so the {rating:.1f} figure remains on the 0-10 "
            f"scale but carries a reduced confidence level.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Serialisation helper (dict form, handy for the web layer / CSV export)
# ---------------------------------------------------------------------------
def rating_to_dict(rating: ShotRating) -> Dict:
    """Convert a :class:`ShotRating` to a plain JSON-serialisable dict."""
    return {
        "shot_type": rating.shot_type,
        "rating": rating.rating,
        "classification": rating.classification,
        "confidence": rating.confidence,
        "covered_weight": rating.covered_weight,
        "strengths": rating.strengths,
        "weaknesses": rating.weaknesses,
        "explanation": rating.explanation,
        "unavailable_metrics": rating.unavailable_metrics,
        "factors": [
            {
                "code": f.code,
                "label": f.label,
                "weight": round(f.weight, 3),
                "score": round(f.score, 2) if f.score is not None else None,
                "weighted": round(f.weighted, 3) if f.weighted is not None else None,
                "comment": f.comment,
            }
            for f in rating.factors
        ],
    }


def save_rating_csv(rating: ShotRating,
                    video_name: str,
                    folder=None) -> str:
    """Write the rating to ``<folder>/shot_rating.csv`` and return the path.

    The CSV mirrors the JSON/dict representation so the per-video rating can be
    consumed by the dashboard, batched pipelines or external tooling.

    Parameters
    ----------
    rating:
        The rating object to persist.
    video_name:
        Logical video/clip name used to populate the first column.
    folder:
        Destination directory (defaults to ``output_data/<video_name>``).
    """
    import csv as _csv

    folder = folder or (BASE_DIR / "output_data" / video_name)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "shot_rating.csv"
    payload = rating_to_dict(rating)

    factors = payload["factors"]
    columns = ["video_name"]
    columns += list(k for k in payload if k != "factors")
    columns += [f"factor_{f['code']}_score" for f in factors]
    columns += [f"factor_{f['code']}_weighted" for f in factors]

    row: List = [video_name]
    row += [payload["shot_type"], payload["rating"], payload["classification"],
            payload["confidence"], payload["covered_weight"],
            "|".join(payload["strengths"]), "|".join(payload["weaknesses"]),
            payload["explanation"],
            "|".join(payload["unavailable_metrics"])]
    row += [f["score"] if f["score"] is not None else "" for f in factors]
    row += [f["weighted"] if f["weighted"] is not None else "" for f in factors]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(row)

    return str(path)
