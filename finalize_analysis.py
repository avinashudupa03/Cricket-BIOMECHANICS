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


def classify_shot(video_name):
    """Run the trained shot classifier over this clip's features.

    Writes output_data/<video_name>/shot_classification.json. The result is
    the model's own prediction plus a confidence; a clip the model is not
    confident about is reported as Unknown rather than forced into a named
    shot. Classification never fails the pipeline - if the model is missing
    or the features are unusable the clip is simply reported as Unknown.
    """
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

    # Model-based shot classification (Unknown when not confident).
    classify_shot(video_name)

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