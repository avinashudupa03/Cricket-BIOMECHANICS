"""Injury risk & batsman safety analysis for cricket batting.

This module screens a processed batting clip for common, detectable injury
mechanisms using the pose-derived data the pipeline already produces:

  * ``batting_phases.csv``    - smoothed joint angles + loading phase labels
  * ``batting_landmarks.csv`` - normalized (x, y, z) landmark positions

It is NOT a medical diagnosis. It flags *mechanical* patterns that sports
coaches and physios commonly associate with overload / tissue stress, so a
batsman can be advised before a niggle becomes an injury. Every finding is
always reported with:

  * which movement was seen (``metric`` + ``side``)
  * how severe the loading pattern was (mild / moderate / high)
  * how often it occurred (``frames``, ``pct_total``)
  * where in the swing it happened (``worst_phase``)
  * a plain-language explanation and a practical coaching cue

Outputs (written to ``output_data/<video_name>/``):

  * ``injury_risk.csv``          - one row per detected flag
  * ``injury_risk_summary.csv``  - overall safety score (0-10) + risk level

Usage:
    python injury_risk.py <video_name>

Missing data is handled honestly: metrics that cannot be computed are simply
not evaluated (not guessed), and the safety summary reports data coverage so
an incomplete capture is never mistaken for a clean bill of health.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Phases where the body is actually loaded (bat swinging / contact / follow).
LOADED_PHASES = {"Downswing", "Impact", "Follow-through"}

# Max-load phases: the moment the knee/trunk actually takes the bodyweight
# shock. Crouching in the *follow-through* to ground the bat is normal and
# safe, so knee/trunk-loading flags only fire here.
MAX_LOAD_PHASES = {"Downswing", "Impact"}

SEVERITY_PENALTY = {"mild": 1.0, "moderate": 2.0, "high": 4.0}

METRIC_INFO = {
    "elbow_hyperextension": (
        "Elbow hyperextension",
        "The elbow locks out past straight during the swing, placing strain "
        "on the joint capsule and the medial ligaments (valgus stress).",
    ),
    "knee_valgus": (
        "Knee collapse (valgus drift)",
        "The knee tracks inward/outward off the straight leg line under "
        "load. Repeated drift is associated with ACL strain, MCL sprains and "
        "patellar tracking problems.",
    ),
    "deep_knee_flexion": (
        "Deep knee flexion under load",
        "The knee folds far past a normal stance squat during a loaded phase "
        "(downswing / impact / follow-through), stressing the patellar "
        "tendon and quadriceps.",
    ),
    "trunk_lean": (
        "Excessive trunk lean / sway",
        "The torso leans far off vertical, overloading the lumbar spine and "
        "core stabilisers and exposing the lower back to shear.",
    ),
    "trunk_flexion": (
        "Rounded / over-flexed back",
        "The trunk folds heavily forward at the hips, risking lumbar disc "
        "and posterior-chain strain, especially when loaded at impact.",
    ),
}


def _num(value, default=None):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _sustained_mask(mask, min_run=3):
    """Keep only True runs of at least ``min_run`` frames (noise filter)."""
    n = len(mask)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _severity_for(metric, value):
    """Return (severity, coaching cue) for a measured peak/min value."""
    if metric == "elbow_hyperextension":
        if value >= 190:
            return "high", ("Stop locking the elbow; keep a soft bend at "
                            "contact and strengthen the forearm / medial elbow.")
        if value >= 184:
            return "moderate", ("Watch the elbow lock-out; aim for a controlled, "
                                "softly bent arm through the contact zone.")
        return "mild", ("Occasional elbow lock-out - keep the arm soft through "
                        "contact and build forearm control.")
    if metric == "knee_valgus":
        if value >= 0.38:
            return "high", ("Knee drifts a lot further off the leg line under load "
                            "than at stance. Work on hip/knee alignment, glute "
                            "strength and landing control.")
        if value >= 0.25:
            return "moderate", ("Notable extra knee drift under load - strengthen "
                                "hip abductors/rotators; keep the knee tracking over the foot.")
        return "mild", ("Slight extra knee drift under load - monitor knee tracking "
                        "and hip alignment.")
    if metric == "deep_knee_flexion":
        if value <= 85:
            return "high", ("Knee folds very deeply under load. Reduce stance depth "
                            "and build patellar-tendon / quadriceps resilience.")
        if value <= 95:
            return "moderate", ("Deep knee fold under load - lighten the squat depth "
                                "and balance knee/ankle mobility.")
        return "mild", ("Knee flexes a little deeper than ideal under load - watch "
                        "for patellar discomfort.")
    if metric == "trunk_lean":
        if value >= 22:
            return "high", ("Torso sways far more under load than at stance. Build "
                            "core anti-rotation strength and check stance alignment.")
        if value >= 14:
            return "moderate", ("Torso sways noticeably more under load - strengthen "
                                "the obliques/core and keep the spine tall through the swing.")
        return "mild", ("Slight extra trunk sway under load - engage the core and "
                        "keep the chest up.")
    if metric == "trunk_flexion":
        if value <= 85:
            return "high", ("Back folds heavily forward under load. Avoid repeated "
                            "heavy swings until lumbar strength/awareness improves.")
        if value <= 95:
            return "moderate", ("Too much forward trunk folding - hinge from the hips "
                                "with a braced spine rather than rounding the back.")
        return "mild", ("Mild forward rounding of the back under load - brace the "
                        "core and keep a neutral spine.")
    return None, None


class InjuryAnalyzer:
    """Compute injury-risk flags from a processed video's per-frame data."""

    def __init__(self, video_name):
        self.video_name = Path(video_name).stem
        self.folder = BASE_DIR / "output_data" / self.video_name
        self.angles_df = None
        self.landmarks_df = None
        self.flags = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self):
        angles_file = self.folder / "batting_phases.csv"
        if not angles_file.exists():
            print(f"ERROR: {angles_file} not found. Run the pipeline first.")
            return False
        self.angles_df = pd.read_csv(angles_file)

        landmarks_file = self.folder / "batting_landmarks.csv"
        if landmarks_file.exists():
            try:
                self.landmarks_df = pd.read_csv(landmarks_file)
            except Exception as exc:
                print(f"[injury_risk] could not read landmarks: {exc}")
                self.landmarks_df = None
        else:
            self.landmarks_df = None
        return True

    # ------------------------------------------------------------------
    # Small vector helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _series(df, col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    @staticmethod
    def _mask_on_phase(df, phases):
        if phases is None:
            return np.ones(len(df), dtype=bool)
        return df["phase"].isin(phases).to_numpy()

    def _build_flag(self, metric, side, frames, phases, flagged,
                    values, take_min=False):
        """Aggregate a boolean ``flagged`` mask into one flag dict row.

        ``values`` holds the underlying magnitude (angle / ratio). Severity is
        taken from a *robust* percentile of the sustained frames (p10 for
        lower-is-worse metrics, p90 for higher-is-worse) so a single 1-2 frame
        tracking spike cannot inflate the rating. Returns the dict row, or
        None when nothing survived the filters.
        """
        idx = np.flatnonzero(flagged)
        if len(idx) == 0:
            return None

        peak = (
            float(np.nanmin(values[idx])) if take_min
            else float(np.nanmax(values[idx]))
        )
        sev_value = float(np.nanpercentile(
            values[idx], 10.0 if take_min else 90.0))
        severity, cue = _severity_for(metric, sev_value)
        if severity is None:
            severity = "mild"

        sub_phases = phases[flagged]
        counts = pd.Series(sub_phases).value_counts() if len(sub_phases) else None
        worst_phase = counts.idxmax() if counts is not None and len(counts) else "across swing"

        label, description = METRIC_INFO.get(metric, (metric, ""))
        return {
            "video_name": self.video_name,
            "metric": metric,
            "metric_label": label,
            "side": side,
            "severity": severity,
            "frames": int(len(idx)),
            "pct_total": round(100.0 * len(idx) / max(len(flagged), 1), 1),
            "peak_value": round(peak, 2),
            "value_units": "deg" if metric in ("elbow_hyperextension",
                                               "deep_knee_flexion",
                                               "trunk_lean",
                                               "trunk_flexion") else "ratio",
            "from_frame": int(frames[idx[0]]),
            "to_frame": int(frames[idx[-1]]),
            "worst_phase": worst_phase,
            "description": description,
            "cue": cue,
        }

    def _stance_baseline(self, values, frame_arr, default):
        """Robust baseline (p90) of a signal over the unloaded phases.

        Used to normalize load-driven metrics: whatever lateral drift or lean
        the batsman already has while standing still (stance posture, camera
        angle, subject size) is cancelled out, and only the *extra* movement
        during the loaded phases is reported.
        """
        df = self.angles_df
        if df is None or "phase" not in df.columns:
            return default
        phase_map = dict(zip(
            self._series(df, "frame"), df["phase"].astype(str).to_numpy()))
        base_mask = np.array([
            phase_map.get(float(f), "Stance") in ("Stance", "Backlift")
            for f in frame_arr])
        vals = values[base_mask & ~np.isnan(values)]
        if len(vals) >= 3:
            return float(np.nanpercentile(vals, 90))
        # No stance window available - fall back to the global p90.
        vals = values[~np.isnan(values)]
        return float(np.nanpercentile(vals, 90)) if len(vals) else default

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------
    def detect_elbows(self):
        df = self.angles_df
        frames = df["frame"].to_numpy(dtype=float)
        phases = df["phase"].astype(str).to_numpy()
        for side in ("left", "right"):
            col = f"{side}_elbow_angle"
            if col not in df.columns:
                continue
            values = self._series(df, col)
            valid = ~np.isnan(values)
            # MediaPipe can report >200 deg when the arms cross; real elbow
            # extension plateaus well below that, so clamp the detection window.
            flagged = valid & (values >= 178.0) & (values <= 200.0)
            flagged = _sustained_mask(flagged, 3)
            row = self._build_flag("elbow_hyperextension", side, frames, phases,
                                   flagged, values)
            if row:
                self.flags.append(row)

    def detect_knee_flexion(self):
        df = self.angles_df
        frames = df["frame"].to_numpy(dtype=float)
        phases = df["phase"].astype(str).to_numpy()
        loaded = self._mask_on_phase(df, MAX_LOAD_PHASES)
        for side in ("left", "right"):
            col = f"{side}_knee_angle"
            if col not in df.columns:
                continue
            values = self._series(df, col)
            valid = ~np.isnan(values)
            # [60, 105): below 60 deg is a landmark/pose artifact (a human knee
            # cannot fold that far in a loaded bat swing), not a real load.
            flagged = valid & loaded & (values >= 60.0) & (values <= 105.0)
            flagged = _sustained_mask(flagged, 3)
            row = self._build_flag("deep_knee_flexion", side, frames, phases,
                                   flagged, values, take_min=True)
            if row:
                self.flags.append(row)

    def detect_trunk_flexion(self):
        df = self.angles_df
        frames = df["frame"].to_numpy(dtype=float)
        phases = df["phase"].astype(str).to_numpy()
        loaded = self._mask_on_phase(df, MAX_LOAD_PHASES)
        rows_cols = [("left_hip_angle", "right_hip_angle")]
        for left_col, right_col in rows_cols:
            lv = self._series(df, left_col) if left_col in df.columns else None
            rv = self._series(df, right_col) if right_col in df.columns else None
            if lv is None and rv is None:
                continue
            if lv is not None and rv is not None:
                values = np.fmin(lv, rv)          # the more flexed side
            else:
                values = lv if lv is not None else rv
            valid = ~np.isnan(values)
            # [70, 105): below 70 deg implies the torso is folded flat, which
            # is a tracking artifact rather than a real batting posture.
            flagged = valid & loaded & (values >= 70.0) & (values <= 105.0)
            flagged = _sustained_mask(flagged, 3)
            row = self._build_flag("trunk_flexion", "core", frames, phases,
                                   flagged, values, take_min=True)
            if row:
                self.flags.append(row)

    def detect_knee_valgus(self):
        if self.landmarks_df is None:
            return
        lm = self.landmarks_df
        if "frame" not in lm.columns:
            return
        frames = self._series(lm, "frame")
        phases_map = {}
        if self.angles_df is not None and "phase" in self.angles_df.columns:
            phases_map = dict(zip(
                self._series(self.angles_df, "frame"),
                self.angles_df["phase"].astype(str).to_numpy(),
            ))
        phases = np.array([phases_map.get(f, "Stance") for f in frames])

        # Reference scale: average torso length (shoulder-midline to hip-midline)
        # across the clip. Used to turn a raw lateral knee drift (small, noisy
        # at broadcast resolution) into a stable fraction of the body size.
        sm_x = (self._series(lm, "left_shoulder_x") +
                self._series(lm, "right_shoulder_x")) / 2.0
        sm_y = (self._series(lm, "left_shoulder_y") +
                self._series(lm, "right_shoulder_y")) / 2.0
        hm_y = (self._series(lm, "left_hip_y") +
                self._series(lm, "right_hip_y")) / 2.0
        torso_vals = np.abs(hm_y - sm_y)
        torso_vals = torso_vals[~np.isnan(torso_vals) & (torso_vals > 0.02)]
        reference = float(np.median(torso_vals)) if len(torso_vals) else 0.0
        if reference <= 0.02:
            return

        for side in ("left", "right"):
            hx = self._series(lm, f"{side}_hip_x")
            hy = self._series(lm, f"{side}_hip_y")
            kx = self._series(lm, f"{side}_knee_x")
            ky = self._series(lm, f"{side}_knee_y")
            ax = self._series(lm, f"{side}_ankle_x")
            ay = self._series(lm, f"{side}_ankle_y")

            # Valid only when all six coords are inside the frame (0..1) and
            # numeric - a clipped/off-screen joint gives garbage geometry.
            def _inside(a):
                return (a > 0.0) & (a < 1.0) & ~np.isnan(a)

            valid = (
                _inside(hx) & _inside(hy) & _inside(kx) & _inside(ky)
                & _inside(ax) & _inside(ay)
            )

            # Perpendicular distance of the knee from the hip->ankle line, in
            # normalized frame units, then scaled by the torso length.
            dx = ax - hx
            dy = ay - hy
            leg_len = np.hypot(dx, dy)
            line_ok = leg_len > 1e-6
            cross = (kx - hx) * dy - (ky - hy) * dx
            dist = np.where(line_ok, cross / np.maximum(leg_len, 1e-9), 0.0)
            ratio = np.where(valid & line_ok,
                             np.abs(dist) / reference, np.nan)

            # Risk = the *excess* over this subject's own stance baseline, so
            # camera angle, subject size and stance posture are cancelled out
            # and only load-driven collapse is flagged.
            baseline = self._stance_baseline(ratio, frames, 0.0)
            excess = ratio - baseline

            loaded = self._mask_on_phase(self.angles_df, MAX_LOAD_PHASES)
            lf = np.zeros(len(lm), dtype=bool)
            if len(loaded) == len(lm):
                lf = loaded
            flagged = ~np.isnan(excess) & (excess >= 0.15) & lf
            flagged = _sustained_mask(flagged, 3)
            row = self._build_flag("knee_valgus", side, frames, phases,
                                   flagged, excess)
            if row:
                self.flags.append(row)

    def detect_trunk_lean(self):
        if self.landmarks_df is None:
            return
        lm = self.landmarks_df
        if not all(c in lm.columns for c in (
                "left_shoulder_x", "right_shoulder_x",
                "left_shoulder_y", "right_shoulder_y",
                "left_hip_x", "right_hip_x",
                "left_hip_y", "right_hip_y")):
            return
        frames = self._series(lm, "frame")
        phases_map = {}
        if self.angles_df is not None and "phase" in self.angles_df.columns:
            phases_map = dict(zip(
                self._series(self.angles_df, "frame"),
                self.angles_df["phase"].astype(str).to_numpy(),
            ))
        phases = np.array([phases_map.get(f, "Stance") for f in frames])

        sx = (self._series(lm, "left_shoulder_x") +
              self._series(lm, "right_shoulder_x")) / 2.0
        sy = (self._series(lm, "left_shoulder_y") +
              self._series(lm, "right_shoulder_y")) / 2.0
        hx = (self._series(lm, "left_hip_x") +
              self._series(lm, "right_hip_x")) / 2.0
        hy = (self._series(lm, "left_hip_y") +
              self._series(lm, "right_hip_y")) / 2.0

        valid = (
            ~np.isnan(sx) & ~np.isnan(sy) & ~np.isnan(hx) & ~np.isnan(hy)
        )
        # Trunk angle from vertical in the image plane (y grows downward).
        # Require a real vertical separation (dy >= 0.05) or the lean read is
        # meaningless; cap at 65 deg which implies a tracking fail.
        dy = np.abs(hy - sy)
        dx = np.abs(hx - sx)
        ok = valid & (dy > 0.05)
        angle = np.degrees(np.arctan2(np.where(ok, dx, 0.0),
                                      np.where(ok, dy, 1.0)))
        angle = np.where(ok & (angle <= 65.0), angle, np.nan)

        # Risk = extra sideways sway during loading vs the subject's normal
        # stance lean (which includes any camera tilt).
        baseline = self._stance_baseline(angle, frames, 0.0)
        excess = angle - baseline

        loaded = self._mask_on_phase(self.angles_df, MAX_LOAD_PHASES)
        lf = np.zeros(len(lm), dtype=bool)
        if len(loaded) == len(lm):
            lf = loaded
        flagged = ~np.isnan(excess) & (excess >= 8.0) & lf
        flagged = _sustained_mask(flagged, 3)
        row = self._build_flag("trunk_lean", "core", frames, phases,
                               flagged, excess)
        if row:
            self.flags.append(row)

    # ------------------------------------------------------------------
    # Coverage + summary
    # ------------------------------------------------------------------
    def _coverage(self, total_frames):
        """Fraction of frames where any safety metric had usable data."""
        usable = 0.0
        if self.angles_df is not None:
            angle_cols = [c for c in self.angles_df.columns
                          if c.endswith("_angle")]
            if angle_cols:
                usable = self.angles_df[angle_cols].apply(
                    lambda s: pd.to_numeric(s, errors="coerce"),
                    axis=0,
                ).notna().any(axis=1).mean()
        if self.landmarks_df is not None:
            land_cols = ["left_shoulder_x", "right_shoulder_x",
                         "left_hip_x", "right_hip_x", "left_knee_x"]
            if all(c in self.landmarks_df.columns for c in land_cols):
                usable = max(
                    usable,
                    self.landmarks_df[land_cols].apply(
                        lambda s: pd.to_numeric(s, errors="coerce"),
                        axis=0,
                    ).notna().any(axis=1).mean(),
                )
        return float(usable) if total_frames else 0.0

    def summarize(self):
        total_frames = len(self.angles_df) if self.angles_df is not None else 0
        coverage = self._coverage(total_frames)

        penalty = sum(SEVERITY_PENALTY.get(f["severity"], 0.0) for f in self.flags)
        safety = round(max(0.0, 10.0 - penalty), 1)

        if safety >= 8.0:
            risk_level = "Low"
        elif safety >= 5.0:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        if coverage < 0.5:
            risk_level += "*"
        if self.flags:
            top = max(self.flags, key=lambda f: SEVERITY_PENALTY.get(
                f["severity"], 0.0))
            top_concern = f"{top['metric_label']} ({top['side']})"
        else:
            top_concern = "None - no significant mechanical flags detected"

        n_high = sum(1 for f in self.flags if f["severity"] == "high")
        n_mod = sum(1 for f in self.flags if f["severity"] == "moderate")
        parts = []
        if not self.flags:
            parts.append("No mechanically risky patterns were detected in the "
                         "pose data available for this clip.")
        else:
            parts.append(f"{len(self.flags)} flag(s): {n_high} high, "
                         f"{n_mod} moderate, "
                         f"{len(self.flags) - n_high - n_mod} mild.")
        if coverage < 0.5:
            parts.append("Note: pose tracking was limited, so some risk factors "
                         "may be under-reported.")
        else:
            parts.append(f"Pose-based risk coverage was {coverage:.0%}.")
        summary = " ".join(parts)

        return {
            "video_name": self.video_name,
            "safety_score": safety,
            "risk_level": risk_level,
            "coverage": round(coverage, 3),
            "n_flags": len(self.flags),
            "max_severity": max((f["severity"] for f in self.flags),
                                key=lambda s: {"mild": 1, "moderate": 2,
                                               "high": 3}.get(s, 0))
            if self.flags else "none",
            "top_concern": top_concern,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def analyze(self):
        self.detect_elbows()
        self.detect_knee_flexion()
        self.detect_trunk_flexion()
        self.detect_knee_valgus()
        self.detect_trunk_lean()
        return self.summarize()

    def save(self, summary):
        flags_df = pd.DataFrame(self.flags)
        flags_path = self.folder / "injury_risk.csv"
        if len(flags_df):
            flags_df.to_csv(flags_path, index=False)
        else:
            pd.DataFrame(columns=[
                "video_name", "metric", "metric_label", "side", "severity",
                "frames", "pct_total", "peak_value", "value_units",
                "from_frame", "to_frame", "worst_phase", "description", "cue",
            ]).to_csv(flags_path, index=False)

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.folder / "injury_risk_summary.csv", index=False)
        return str(flags_path)


def print_report(summary):
    print()
    print("=" * 56)
    print("INJURY RISK & BATSMAN SAFETY")
    print("=" * 56)
    print(f"Video         : {summary['video_name']}")
    print(f"Safety score  : {summary['safety_score']:.1f} / 10")
    print(f"Risk level    : {summary['risk_level']}")
    print(f"Flags         : {summary['n_flags']}")
    print(f"Top concern   : {summary['top_concern']}")
    print(f"Coverage      : {summary['coverage']:.0%}")
    print()
    print("-" * 56)
    print("DETECTED FLAGS")
    print("-" * 56)
    if not summary["n_flags"]:
        print("  No mechanically risky patterns detected.")
    else:
        # Re-open the flags to mirror what the pipeline stores.
        flags_file = BASE_DIR / "output_data" / summary["video_name"] / "injury_risk.csv"
        if flags_file.exists():
            try:
                df = pd.read_csv(flags_file)
                for _, r in df.iterrows():
                    side = f" ({r['side']})" if r["side"] != "core" else ""
                    print(f"  [{r['severity']:>8}] {r['metric_label']}{side} - "
                          f"{r['frames']} frames, peak {r['peak_value']} "
                          f"{r['value_units']} ({r['worst_phase']})")
                    print(f"           {r['cue']}")
            except Exception:
                pass
    print()
    print(f"Summary: {summary['summary']}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        print("Usage:")
        print("python injury_risk.py <video_name>")
        return 1

    video_name = Path(argv[0]).stem
    analyzer = InjuryAnalyzer(video_name)
    if not analyzer.load():
        return 1

    summary = analyzer.analyze()
    flags_path = analyzer.save(summary)

    print_report(summary)
    print()
    print(f"Flags written to  : {flags_path}")
    print(f"Summary written to: {analyzer.folder / 'injury_risk_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())