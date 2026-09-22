import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import OUTPUT_DATA


INPUT_FILE = OUTPUT_DATA / "batting_angles.csv"


def main():

    if not INPUT_FILE.exists():
        print("ERROR: batting_angles.csv not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    angle_columns = [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_knee_angle",
        "right_knee_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
        "left_hip_angle",
        "right_hip_angle"
    ]

    for column in angle_columns:

        plt.figure(figsize=(10, 5))

        plt.plot(
            df["timestamp_ms"],
            df[column]
        )

        plt.xlabel("Time (ms)")
        plt.ylabel("Angle (degrees)")
        plt.title(column.replace("_", " ").title())

        plt.grid(True)

        output_file = (
            OUTPUT_DATA /
            f"{column}.png"
        )

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.show()

        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
