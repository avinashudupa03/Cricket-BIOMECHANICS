import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:
        print("Usage:")
        print("python biomechanics_analyzer.py <video_name>")
        print()
        print("Example:")
        print("python biomechanics_analyzer.py defence")
        return

    video_name = Path(argv[0]).stem

    folder = BASE_DIR / "output_data" / video_name

    input_file = folder / "batting_phases.csv"
    output_file = folder / "biomechanics_features.csv"

    if not input_file.exists():
        print(f"ERROR: {input_file} not found.")
        return

    df = pd.read_csv(input_file)

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

    features = {}

    # -----------------------------------------
    # Angle statistics
    # -----------------------------------------

    for column in angle_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        features[f"{column}_min"] = values.min()
        features[f"{column}_max"] = values.max()
        features[f"{column}_mean"] = values.mean()
        features[f"{column}_std"] = values.std()

    # -----------------------------------------
    # Phase frame counts
    # -----------------------------------------

    for phase, count in df["phase"].value_counts().items():

        key = phase.lower().replace("-", "").replace(" ", "_")

        features[f"{key}_frames"] = int(count)

    # -----------------------------------------
    # Impact information
    # -----------------------------------------

    impact = df[
        df["phase"] == "Impact"
    ]

    if len(impact) > 0:

        impact_frame = impact.iloc[
            len(impact) // 2
        ]

        features["impact_frame"] = (
            impact_frame["frame"]
        )

        features["impact_time_ms"] = (
            impact_frame["timestamp_ms"]
        )

        for column in angle_columns:

            features[
                f"{column}_at_impact"
            ] = impact_frame[column]

    # -----------------------------------------
    # Downswing information
    # -----------------------------------------

    downswing = df[
        df["phase"] == "Downswing"
    ]

    if len(downswing) > 0:

        features["downswing_start_frame"] = (
            downswing.iloc[0]["frame"]
        )

        features["downswing_end_frame"] = (
            downswing.iloc[-1]["frame"]
        )

        features["downswing_duration_ms"] = (
            downswing.iloc[-1]["timestamp_ms"]
            -
            downswing.iloc[0]["timestamp_ms"]
        )

    # -----------------------------------------
    # Follow-through information
    # -----------------------------------------

    follow = df[
        df["phase"] == "Follow-through"
    ]

    if len(follow) > 0:

        features["followthrough_duration_ms"] = (
            follow.iloc[-1]["timestamp_ms"]
            -
            follow.iloc[0]["timestamp_ms"]
        )

    # -----------------------------------------
    # Movement
    # -----------------------------------------

    movement = pd.to_numeric(
        df["movement_smooth"],
        errors="coerce"
    ).dropna()

    if len(movement) > 0:

        features["maximum_movement_score"] = (
            movement.max()
        )

        features["average_movement_score"] = (
            movement.mean()
        )

    # -----------------------------------------
    # Save
    # -----------------------------------------

    features_df = pd.DataFrame([features])

    features_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("===================================")
    print("BIOMECHANICS FEATURE EXTRACTION")
    print("===================================")
    print(f"Video: {video_name}")
    print(f"Features generated: {len(features_df.columns)}")
    print(f"Output: {output_file}")
    print()
    print(features_df.T.to_string())


if __name__ == "__main__":
    main()
