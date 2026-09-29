"""Run the CSV-based analytics steps in one process.

Merges angle_calculator, phase_detector, biomechanics_analyzer and
shot_rater into a single interpreter so the expensive pandas/sklearn
imports and Python startup are paid once instead of four times.

Outputs are byte-for-byte the same files as the individual scripts:

  output_data/<video_name>/batting_angles.csv
  output_data/<video_name>/batting_phases.csv
  output_data/<video_name>/biomechanics_features.csv
  output_data/<video_name>/shot_rating.csv

Usage:
    python finalize_analysis.py <video_name> <shot_type>
"""

import sys
from pathlib import Path

import angle_calculator
import phase_detector
import biomechanics_analyzer
import shot_rater
import injury_risk
import provenance
import ml_model

BASE_DIR = Path(__file__).resolve().parent


def classify_shot(video_name, shot_type=None):
    """Run the trained shot classifier over this clip's features.

    If ``shot_type`` is provided (from upload selection), use it as the
    classification instead of running the ML model. This ensures the
    classification matches the user's label.

    Writes output_data/<video_name>/shot_classification.json.
    """
    if shot_type:
        # Use the provided shot type (from upload selection)
        import json
        from pathlib import Path
        folder = BASE_DIR / "output_data" / video_name
        folder.mkdir(parents=True, exist_ok=True)

        result = {
            "video_name": video_name,
            "shot_type": shot_type,
            "predicted": shot_type,
            "confidence": 1.0,
            "threshold": 0.0,
            "is_unknown": False,
            "reason": "user_labeled",
            "reason_label": "User-provided label from upload",
            "model_name": "user_label",
            "n_classes": 6,
            "probabilities": {shot_type: 1.0},
        }

        with open(folder / "shot_classification.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        print()
        print("=" * 56)
        print("SHOT CLASSIFICATION")
        print("=" * 56)
        print(f"Model      : user_label (from upload)")
        print(f"Predicted  : {shot_type}")
        print(f"Confidence : 1.000 (user-labeled)")
        print(f"Final class: {shot_type}")
        print(f"Reason     : User-provided label from upload")
        return result

    # Fallback to ML model if no shot_type provided
    try:
        result = ml_model.classify_video(video_name)
    except Exception as exc:                      # never break the pipeline
        print(f"[classify] skipped for {video_name}: {exc}")
        return None

    conf = result.get("confidence")
    conf_txt = "n/a" if conf is None else f"{conf:.3f}"
    print()
    print("=" * 56)
    print("SHOT CLASSIFICATION")
    print("=" * 56)
    print(f"Model      : {result.get('model_name') or 'n/a'}")
    print(f"Predicted  : {result.get('predicted') or 'n/a'}")
    print(f"Confidence : {conf_txt} (threshold {result.get('threshold')})")
    print(f"Final class: {result.get('shot_type')}")
    print(f"Reason     : {result.get('reason_label')}")
    return result


def _annotate_tracking_debug(video_name):
    """Fill the ``phase`` column of tracking_debug.csv from batting_phases.csv.

    main.py writes the tracking-debug rows with an empty phase placeholder
    (phases are not known until phase_detector runs). Once the per-frame phase
    labels exist, this joins them back onto the debug CSV so the file carries
    the full per-frame tracking + phase record in one place.
    """
    import csv

    folder = BASE_DIR / "output_data" / video_name
    debug_file = folder / "tracking_debug.csv"
    phases_file = folder / "batting_phases.csv"
    if not debug_file.exists() or not phases_file.exists():
        return

    try:
        phase_by_frame = {}
        import pandas as pd
        df = pd.read_csv(phases_file)
        for _, row in df.iterrows():
            phase_by_frame[int(row["frame"])] = str(row["phase"])

        with open(debug_file, "r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = list(reader)

        header = rows[0]
        if "phase" not in header:
            return
        phase_idx = header.index("phase")
        frame_idx = header.index("frame_number") if "frame_number" in header else 0

        for row in rows[1:]:
            if len(row) <= phase_idx:
                continue
            try:
                frame_no = int(row[frame_idx])
            except (TypeError, ValueError):
                continue
            row[phase_idx] = phase_by_frame.get(frame_no, "")

        with open(debug_file, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
    except Exception as exc:
        print(f"[annotate_tracking_debug] {exc}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python finalize_analysis.py <video_name> [shot_type]")
        return 1

    video_name = Path(sys.argv[1]).stem
    shot_type = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    steps = [
        ("angle_calculator.py", lambda: angle_calculator.main([video_name])),
        ("phase_detector.py", lambda: phase_detector.main([video_name])),
        ("biomechanics_analyzer.py", lambda: biomechanics_analyzer.main([video_name])),
        ("shot_rater.py", lambda: shot_rater.main([
            video_name, "--shot-type", shot_type])),
        ("injury_risk.py", lambda: injury_risk.main([video_name])),
    ]

    for token, run in steps:
        print()
        print("=" * 56)
        print(f"RUNNING: {token}")
        print("=" * 56)
        run()

    # Fill the phase column of tracking_debug.csv now that phases exist.
    _annotate_tracking_debug(video_name)

    # Shot classification: use uploaded shot_type if provided, else ML model.
    classify_shot(video_name, shot_type)

    provenance.write_provenance(BASE_DIR / "output_data" / video_name)
    provenance.write_report_header()

    print()
    print("=" * 56)
    print("FINALIZE COMPLETE")
    print("=" * 56)
    print(f"Video: {video_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())