import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def calculate_angle(a, b, c):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if a.size == 0 or b.size == 0 or c.size == 0:
        return np.nan

    vector1 = a - b
    vector2 = c - b

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return np.nan

    if np.isnan(vector1).any() or np.isnan(vector2).any():
        return np.nan

    cosine_angle = np.dot(
        vector1,
        vector2
    ) / (norm1 * norm2)

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine_angle)
    )


def get_point(row, name):

    try:
        x = float(row[f"{name}_x"])
        y = float(row[f"{name}_y"])
        z = float(row[f"{name}_z"])
    except (TypeError, ValueError):
        return np.full(3, np.nan)

    return [x, y, z]


def _point_array(df, name):
    """Extract the (N, 3) landmark array for a joint from the dataframe.

    Missing columns or non-numeric cells become NaN, so the downstream
    angle math silently drops the frame instead of crashing the run.
    """
    columns = [f"{name}_x", f"{name}_y", f"{name}_z"]
    out = np.full((len(df), 3), np.nan)
    for idx, col in enumerate(columns):
        if col in df.columns:
            out[:, idx] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return out


def _vectorized_angle(a, b, c):
    """Angle at point b formed by a-b-c, for whole (N,3) arrays at once."""
    a = np.ascontiguousarray(a, dtype=float)
    b = np.ascontiguousarray(b, dtype=float)
    c = np.ascontiguousarray(c, dtype=float)
    n = len(a)

    vec1 = a - b
    vec2 = c - b
    norm1 = np.sqrt(np.einsum("ij,ij->i", vec1, vec1))
    norm2 = np.sqrt(np.einsum("ij,ij->i", vec2, vec2))

    nonzero = (norm1 > 1e-9) & (norm2 > 1e-9)
    cos_angle = np.divide(
        np.einsum("ij,ij->i", vec1, vec2),
        norm1 * norm2,
        out=np.full(n, np.nan),
        where=nonzero,
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 1:

        print("Usage:")
        print("python angle_calculator.py <video_name>")
        print()
        print("Example:")
        print("python angle_calculator.py defence")
        return

    video_name = Path(argv[0]).stem

    folder = BASE_DIR / "output_data" / video_name

    input_file = folder / "batting_landmarks.csv"
    output_file = folder / "batting_angles.csv"

    if not input_file.exists():

        print(
            f"ERROR: {input_file} not found."
        )
        return

    df = pd.read_csv(input_file)

    # Vectorised column-wise computation (no per-row Python loop), so long
    # clips with thousands of frames go ~50-100x faster than iterrows.
    body = {
        "left": {name: _point_array(df, f"left_{name}")
                 for name in ("shoulder", "elbow", "wrist",
                              "hip", "knee", "ankle")},
        "right": {name: _point_array(df, f"right_{name}")
                  for name in ("shoulder", "elbow", "wrist",
                               "hip", "knee", "ankle")},
    }

    result = {
        "frame": df["frame"] if "frame" in df.columns else
                 pd.Series(np.arange(len(df))),
        "timestamp_ms": df["timestamp_ms"] if "timestamp_ms" in df.columns
                        else np.full(len(df), np.nan),
    }

    for side in ("left", "right"):
        P = body[side]
        result[f"{side}_elbow_angle"] = _vectorized_angle(
            P["shoulder"], P["elbow"], P["wrist"])
        result[f"{side}_knee_angle"] = _vectorized_angle(
            P["hip"], P["knee"], P["ankle"])
        result[f"{side}_shoulder_angle"] = _vectorized_angle(
            P["hip"], P["shoulder"], P["elbow"])
        result[f"{side}_hip_angle"] = _vectorized_angle(
            P["shoulder"], P["hip"], P["knee"])

    angles_df = pd.DataFrame(result)

    angles_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("===================================")
    print("ANGLE CALCULATION COMPLETE")
    print("===================================")
    print(f"Video: {video_name}")
    print(f"Frames analysed: {len(angles_df)}")
    print(f"Output: {output_file}")

    return angles_df


if __name__ == "__main__":
    main()
