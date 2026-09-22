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

BASE_DIR = Path(__file__).resolve().parent


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