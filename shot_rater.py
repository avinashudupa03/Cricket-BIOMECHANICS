"""Command-line shot rater for cricket batting analysis.

Rates every shot for a processed video using the biomechanical features
already extracted by ``biomechanics_analyzer.py`` (a separate, reusable step)
and writes the result to ``output_data/<video_name>/shot_rating.csv``.

Usage:
    python shot_rater.py <video_name> [--shot-type TYPE] [--json]

Example:
    python shot_rater.py defence --shot-type defence
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import shot_rating as sr

BASE_DIR = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rate a batting shot (0-10).")
    parser.add_argument("video_name", help="Name of the processed video folder.")
    parser.add_argument("--shot-type", default="unknown",
                        help="Human-readable shot type label (e.g. defence).")
    parser.add_argument("--json", action="store_true",
                        help="Print the rating as JSON instead of a report.")
    args = parser.parse_args(argv)

    video_name = Path(args.video_name).stem
    folder = BASE_DIR / "output_data" / video_name
    features_file = folder / "biomechanics_features.csv"

    if not features_file.exists():
        print(f"ERROR: {features_file} not found. "
              f"Run the pipeline first (process_video.py).")
        return 1

    row = pd.read_csv(features_file).iloc[0].to_dict()
    extra = sr.build_extra(video_name)
    rating = sr.rate_shot(row, extra=extra, shot_type=args.shot_type)
    payload = sr.rating_to_dict(rating)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(rating)

    # Persist the rating alongside the other per-video outputs.
    out = sr.save_rating_csv(rating, video_name, folder=folder)
    print(f"\nRating saved to: {out}")

    return 0


def print_report(rating: sr.ShotRating) -> None:
    print()
    print("=" * 56)
    print("SHOT RATING")
    print("=" * 56)
    print(f"Shot type         : {rating.shot_type or 'unknown'}")
    print(f"Final rating      : {rating.rating:.1f} / 10")
    print(f"Classification    : {rating.classification}")
    print(f"Confidence        : {rating.confidence:.0%}")
    print(f"Data coverage     : {rating.covered_weight:.0%} of factor weight")
    print()
    print("-" * 56)
    print("FACTOR BREAKDOWN (score / effective weight)")
    print("-" * 56)
    for f in rating.factors:
        score = f"{f.score:.1f}" if f.score is not None else "n/a"
        weighted = f"{f.weighted:.2f}" if f.weighted is not None else "n/a"
        print(f"  {f.label:<34} {score:>5}  w={weighted}")
    print()
    if rating.strengths:
        print("STRENGTHS:")
        for s in rating.strengths:
            print(f"  + {s}")
    print()
    if rating.weaknesses:
        print("WEAKNESSES:")
        for w in rating.weaknesses:
            print(f"  - {w}")
    print()
    print("EXPLANATION:")
    print(f"  {rating.explanation}")
    print()
    if rating.unavailable_metrics:
        print("UNAVAILABLE METRICS (not invented):")
        for m in rating.unavailable_metrics:
            print(f"  ! {m}")


if __name__ == "__main__":
    sys.exit(main())
