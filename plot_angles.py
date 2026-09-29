"""Plot joint angle time-series charts for a processed video.

Reads ``output_data/<video_name>/batting_angles.csv`` and saves one PNG
per joint angle column into the same folder.

Usage:
    python plot_angles.py <video_name> [--show]

The ``--show`` flag opens interactive plot windows (useful on a desktop).
By default plots are only saved to disk so the script is safe to run in
headless environments (Docker, CI).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; --show overrides via plt.show()
import matplotlib.pyplot as plt

from config import OUTPUT_DATA


ANGLE_COLUMNS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot joint angle time-series charts.")
    parser.add_argument("video_name", help="Processed video name (folder in output_data/)")
    parser.add_argument("--show", action="store_true", help="Display interactive plot windows")
    args = parser.parse_args(argv)

    video_name = Path(args.video_name).stem
    input_file = OUTPUT_DATA / video_name / "batting_angles.csv"

    if not input_file.exists():
        print(f"ERROR: {input_file} not found.")
        print(f"Make sure '{video_name}' has been processed (run finalize_analysis.py first).")
        return 1

    df = pd.read_csv(input_file)

    if "timestamp_ms" not in df.columns:
        print("ERROR: timestamp_ms column not found in batting_angles.csv")
        return 1

    output_dir = input_file.parent
    saved_files = []

    for column in ANGLE_COLUMNS:
        if column not in df.columns:
            print(f"  [SKIP] {column} not in data")
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp_ms"], df[column])
        plt.xlabel("Time (ms)")
        plt.ylabel("Angle (degrees)")
        plt.title(column.replace("_", " ").title())
        plt.grid(True)

        output_file = output_dir / f"{column}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        saved_files.append(output_file)
        print(f"  Saved: {output_file}")

        if args.show:
            plt.show()
        else:
            plt.close(fig)

    print()
    print(f"Angle plots saved for {video_name}: {len(saved_files)} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
