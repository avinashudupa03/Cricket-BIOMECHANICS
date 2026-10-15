import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def smooth(series, window=5):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    series = series.interpolate(
        limit_direction="both"
    )

    return series.rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean()


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:

        print("Usage:")
        print("python phase_detector.py <video_name>")
        return

    video_name = Path(argv[0]).stem

    folder = BASE_DIR / "output_data" / video_name

    input_file = folder / "batting_angles.csv"
    output_file = folder / "batting_phases.csv"

    if not input_file.exists():

        print(f"ERROR: {input_file} not found.")
        return

    df = pd.read_csv(input_file)

    elbow = [
        "left_elbow_angle",
        "right_elbow_angle"
    ]

    shoulder = [
        "left_shoulder_angle",
        "right_shoulder_angle"
    ]

    hip = [
        "left_hip_angle",
        "right_hip_angle"
    ]

    knee = [
        "left_knee_angle",
        "right_knee_angle"
    ]

    all_angles = (
        elbow +
        shoulder +
        hip +
        knee
    )

    for column in all_angles:

        df[column] = smooth(
            df[column]
        )

    df["elbow_movement"] = (
        df[elbow]
        .diff()
        .abs()
        .mean(axis=1)
        .fillna(0)
    )

    df["shoulder_movement"] = (
        df[shoulder]
        .diff()
        .abs()
        .mean(axis=1)
        .fillna(0)
    )

    df["hip_movement"] = (
        df[hip]
        .diff()
        .abs()
        .mean(axis=1)
        .fillna(0)
    )

    df["knee_movement"] = (
        df[knee]
        .diff()
        .abs()
        .mean(axis=1)
        .fillna(0)
    )

    df["movement_score"] = (
        df["elbow_movement"] * 0.35 +
        df["shoulder_movement"] * 0.30 +
        df["hip_movement"] * 0.20 +
        df["knee_movement"] * 0.15
    )

    df["movement_smooth"] = (
        df["movement_score"]
        .rolling(
            3,
            center=True,
            min_periods=1
        )
        .mean()
    )

    peak_frame = int(
        df["movement_smooth"].idxmax()
    )

    total_frames = len(df)

    # ------------------------------------------------------------------
    # Robust impact detection. A single global movement peak is noisy: it
    # can be dragged to the clip edges by the batsman walking to/from the
    # crease, or by a speck of keypoint jitter. The bat (lead hand) reaches
    # its lowest vertical point very near ball contact, so when a reliable
    # lead-wrist-height signal exists we cross-check the movement peak
    # against the wrist-height minimum and trust the wrist cue when the two
    # agree (within a tolerance window). Individual frames are therefore
    # never allowed to misplace the phases on their own.
    # ------------------------------------------------------------------
    imp_from_wrist = False
    impact_frame = peak_frame
    if "lead_wrist_height" in df.columns:
        wh = pd.to_numeric(df["lead_wrist_height"], errors="coerce")
        wh = wh.rolling(
            3, center=True, min_periods=1
        ).mean()
        wh = wh.dropna()
        if len(wh):
            wrist_min_frame = int(wh.idxmin())
            # Tolerance window: wrist minimum must sit near the movement peak
            # to be trusted. Kept generous (rel. to clip length) so real
            # contacts are not rejected, tight enough to reject edge noise.
            window = max(3, int(total_frames * 0.20))
            if abs(wrist_min_frame - peak_frame) <= window:
                impact_frame = wrist_min_frame
                imp_from_wrist = True

    # Never let a spurious edge event place impact inside the first/last
    # tiny slice unless the clip is very short (then phases are coarse anyway).
    edge_margin = max(2, int(total_frames * 0.05))
    if total_frames > 2 * edge_margin + 1:
        lo = edge_margin
        hi = total_frames - 1 - edge_margin
        if impact_frame < lo:
            impact_frame = lo
        if impact_frame > hi:
            impact_frame = hi

    # Adaptive phase windows: size the loading phases by *actual* framerate
    # (targeting typical swing durations in seconds), then clamp to a fraction
    # of the clip so very short recordings degrade gracefully instead of
    # producing absurdly large windows. Falls back to a 30fps assumption when
    # the timestamp column is missing or constant.
    fps = 30.0
    if "timestamp_ms" in df.columns:
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").dropna()
        if len(ts) >= 2:
            diffs = ts.diff().dropna()
            diffs = diffs[diffs > 0]
            if len(diffs):
                fps = 1000.0 / float(diffs.median())

    def _window(seconds, min_frames, max_fraction):
        n = max(min_frames, int(round(fps * seconds)))
        return min(n, max(min_frames, int(total_frames * max_fraction)))

    downswing_len = _window(0.30, 3, 0.25)
    backlift_len = _window(0.45, 4, 0.35)

    impact_start = max(
        0,
        impact_frame - 1
    )

    impact_end = min(
        total_frames - 1,
        impact_frame + 1
    )

    downswing_start = max(
        0,
        impact_start - downswing_len
    )

    backlift_start = max(
        0,
        downswing_start - backlift_len
    )

    phases = []

    for frame in range(total_frames):

        if frame < backlift_start:
            phase = "Stance"

        elif frame < downswing_start:
            phase = "Backlift"

        elif frame < impact_start:
            phase = "Downswing"

        elif frame <= impact_end:
            phase = "Impact"

        else:
            phase = "Follow-through"

        phases.append(phase)

    df["phase"] = phases

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print("===================================")
    print("PHASE DETECTION COMPLETE")
    print("===================================")
    print(f"Video: {video_name}")
    print(f"Frames analysed: {total_frames}")
    print(f"Peak movement frame: {peak_frame}")
    print(f"Impact frame: {impact_frame} "
          f"({'from wrist-height minimum' if imp_from_wrist else 'from movement peak'})")
    print(
        f"Impact time: "
        f"{df.loc[impact_frame, 'timestamp_ms']} ms"
    )

    print()
    print("Phase distribution:")
    print(df["phase"].value_counts())

    print()
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
