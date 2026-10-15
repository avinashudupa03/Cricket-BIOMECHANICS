"""Scientific-validity provenance labeling for all outputs.

This module classifies every value the pipeline emits into one of four
epistemically distinct categories, so that academic readers (and the
researchers themselves) cannot mistake one for another:

    MEASURED        - Directly observed from the source video with no
                      biomechanical interpretation added (frame numbers,
                      timestamps, per-pixel content, video metadata).
    ESTIMATED_2D    - Derived from a single 2D camera view through a pose
                      estimation model (MediaPipe). Angles and distances are
                      *apparent* quantities in the camera plane - they are
                      NOT clinical joint angles. The model's z output is an
                      estimate of depth, not a measurement.
    ESTIMATED_3D    - Values reconstructed/refined from the 2D estimates via
                      geometry (e.g. torso-normalised distances). Still
                      camera-projection-dependent; not marker-based motion
                      capture. No claim of clinical accuracy.
    MODEL_PREDICTION- Output of a machine-learning classifier trained on
                      this small dataset (n=9 videos). Allowed to be wrong;
                      MUST be reported with its confidence scores.
    CONFIDENCE      - A 0..1 number from a pose model or classifier. It is a
                      self-reported internal score, NOT an accuracy measure.

Classifying order (highest precedence used): MEASURED > ESTIMATED_2D >
ESTIMATED_3D > MODEL_PREDICTION > CONFIDENCE.

The pipeline MUST NOT claim clinical/professional biomechanical accuracy:
single-camera 2D pose estimation is not validated against lab-based motion
capture, and the classified shot labels are model predictions on a tiny
dataset. Reports therefore carry a standard disclaimer referencing this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_data"
REPORTS_DIR = BASE_DIR / "reports"

DISCLAIMER = (
    "SCIENTIFIC VALIDITY DISCLAIMER\n"
    "==============================\n"
    "These results are research-grade estimates from SINGLE-CAMERA 2D pose\n"
    "estimation and a classifier trained on a very small dataset. They are\n"
    "NOT clinical biomechanical measurements: no marker-based motion\n"
    "capture, calibration to world coordinates, or clinical validation was\n"
    "performed. Angles are apparent projections in the camera plane; ML\n"
    "shot labels are predictions with associated (self-reported) confidence\n"
    "scores, not ground truth. Treat all quantitative outputs as estimates\n"
    "for exploratory analysis only.\n"
)

# Per-output-file classification for the per-video CSV artefacts.
ARTEFACTS = {
    "batting_landmarks.csv": (
        "ESTIMATED_3D",
        "Raw per-frame landmark x/y coordinates are pixel-space estimates from "
        "MediaPipe (ESTIMATED_2D); the z column is a depth estimate. Short-gap "
        "interpolated rows and visibility/presence columns are MODEL-derived "
        "confidence, not measurements. Consumer math should treat every "
        "coordinate as camera-projected, NOT a true 3D position.",
    ),
    "batting_angles.csv": (
        "ESTIMATED_2D",
        "Apparent joint angles in the camera projection plane from estimated "
        "landmarks. Comparable across clips of the SAME camera angle only. Not "
        "clinical joint angles.",
    ),
    "batting_phases.csv": (
        "ESTIMATED_2D",
        "Swing phases auto-derived from estimated landmark motion with a "
        "rule-based detector. Phase labels/impact timing are algorithmic "
        "annotations and may be wrong.",
    ),
    "biomechanics_features.csv": (
        "ESTIMATED_3D",
        "Angle summaries and torso-normalised distances derived from the "
        "ESTIMATED_2D stream. FPS-corrected velocities use real timestamps. "
        "All quantities inherit the camera-projection uncertainty.",
    ),
    "output_video_model_predictions.csv": (
        "MODEL_PREDICTION",
        "Shot-type labels produced by the ML classifier plus its confidence "
        "scores. On the current n=9 dataset these are exploratory and are "
        "expected to be frequently wrong.",
    ),
}


def _class_of(path: Path):
    for name, (cls, note) in ARTEFACTS.items():
        if path.name == name:
            return cls, note
    return None, None


def write_provenance(folder: Path) -> bool:
    """Write measurement_provenance.txt documenting value categories."""
    if not folder.is_dir():
        return False
    lines = [
        "MEASUREMENT PROVENANCE (scientific validity)",
        "=" * 52,
        "",
        DISCLAIMER,
        "",
        "Value categories used in this project:",
        "  MEASURED          direct video observations (frame #, timestamp, pixels)",
        "  ESTIMATED_2D      apparent camera-plane quantity from pose estimation",
        "  ESTIMATED_3D      refined via geometry, still camera-projected",
        "  MODEL_PREDICTION  machine-learning classifier output",
        "  CONFIDENCE        0..1 self-reported score, not an accuracy measure",
        "",
        "Per-file classification in this folder:",
        "",
    ]
    for f in sorted(folder.glob("*.csv")):
        cls, note = _class_of(f)
        note = note or "MEASURED (metadata) / mixed - see individual columns"
        lines.append(f"  {f.name:<45} -> {cls}")
        lines.append(f"      {note}")
    lines.append("")
    out = folder / "measurement_provenance.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    return True


def write_report_header():
    """Write reports/scientific_validity.txt summarising the disclaimer."""
    lines = [
        DISCLAIMER,
        "",
        "Categories used across reports/CSVs:",
        "  MEASURED          direct video observations",
        "  ESTIMATED_2D      apparent camera-plane pose-estimation values",
        "  ESTIMATED_3D      geometry-derived values (camera-dependent)",
        "  MODEL_PREDICTION  classifier predictions (label + confidence)",
        "  CONFIDENCE        self-reported score, not accuracy",
        "",
        "Every per-video output folder contains measurement_provenance.txt.",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "scientific_validity.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    only = argv[0] if argv else None

    written = 0
    for folder in OUTPUT_DIR.iterdir():
        if not folder.is_dir():
            continue
        if folder.name == "ml_preprocessing":
            continue
        if only and only not in folder.name:
            continue
        if write_provenance(folder):
            written += 1
    out = write_report_header()
    print(f"Wrote provenance notes to {written} video folder(s): {OUTPUT_DIR}/<video>/measurement_provenance.txt")
    print(f"Report header: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())