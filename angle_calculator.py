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


def visibility_mask(df, name, min_visibility):
    """Per-frame boolean mask of *reliable* detections for a landmark.

    A landmark counts as reliable only when visibility data is available and
    the MediaPipe per-landmark confidence is >= ``min_visibility``. Landmarks
    that are missing entirely (no visibility column) are NOT excluded - the
    mask is only applied to filter the low-confidence detections that the
    model itself marked as unreliable.
    """
    vis_col = f"{name}_visibility"
    if vis_col not in df.columns:
        return None
    vis = pd.to_numeric(df[vis_col], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(vis) & (vis >= min_visibility)


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

    MIN_VISIBILITY = 0.30

    # Vectorised column-wise computation (no per-row Python loop), so long
    # clips with thousands of frames go ~50-100x faster than iterrows.
    body = {}
    for side in ("left", "right"):
        cols = {}
        for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            full_name = f"{side}_{name}"
            pts = _point_array(df, full_name)
            mask = visibility_mask(df, full_name, MIN_VISIBILITY)
            if mask is not None:
                # Zero-out landmarks the pose model itself flagged as
                # unreliable so they never poison angle/feature math.
                pts = np.where(mask[:, None], pts, np.nan)
            cols[name] = pts
        body[side] = cols

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

    # ------------------------------------------------------------------
    # Derived, scale-normalised signals (normalise by torso length so the
    # values are comparable across subjects, camera distances and body sizes).
    # ------------------------------------------------------------------
    L, R = body["left"], body["right"]

    def _mid(P, joint, Q, joint2):
        a = P [joint]
        b = Q[joint2]
        return np.nanmean(np.stack([a, b], axis=-1), axis=-1)

    mid_shoulder = _mid(L, "shoulder", R, "shoulder")
    mid_hip = _mid(L, "hip", R, "hip")
    nose = _point_array(df, "nose")

    def _vec_norm(v):
        return np.sqrt(np.nansum(v * v, axis=1))

    torso_len = _vec_norm(mid_shoulder - mid_hip)
    # Rolling median of torso length stabilises the per-frame normalisation
    # scale against single-frame jitter. Unavailable frames stay NaN.
    torso_len = pd.Series(torso_len).rolling(
        window=5, center=True, min_periods=1).median().to_numpy()
    safe = np.where(np.abs(torso_len) > 1e-9, torso_len, np.nan)

    # Lead (top) hand side: the wrist with the smaller y (higher up). A
    # missing wrist on one side falls back to the other.
    Lw = L["wrist"]; Rw = R["wrist"]
    L_valid = np.isfinite(Lw[:, 1]); R_valid = np.isfinite(Rw[:, 1])
    use_left = np.full(len(df), True)
    both = L_valid & R_valid
    use_left[both] = Lw[both, 1] < Rw[both, 1]
    use_left[~L_valid & R_valid] = False
    use_left[L_valid & ~R_valid] = True
    use_left[~L_valid & ~R_valid] = True

    lead_wrist = np.where(use_left[:, None], Lw, Rw)
    lead_elbow = np.where(use_left[:, None], L["elbow"], R["elbow"])
    lead_shoulder = np.where(use_left[:, None], L["shoulder"], R["shoulder"])

    # Wrist height above the hip midline, in torso-length units.
    result["lead_wrist_height"] = (
        (mid_hip[:, 1] - lead_wrist[:, 1]) / safe
    )
    # Top wrist horizontal position relative to hip midline (torso units).
    result["lead_wrist_lateral"] = (
        (lead_wrist[:, 0] - mid_hip[:, 0]) / safe
    )
    # Lead-arm elbow angle (the control/top arm governs bat-face orientation).
    result["lead_elbow_angle"] = _vectorized_angle(
        lead_shoulder, lead_elbow, lead_wrist)

    # Torso rotation: angle between the shoulder line and the hip line in the
    # image plane. A large value means the upper trunk has rotated relative to
    # the lower trunk (side-on / across-the-line movement).
    def _line_angle(p, q):
        dx = q[:, 0] - p[:, 0]
        dy = q[:, 1] - p[:, 1]
        out = np.full(len(df), np.nan)
        finite = np.isfinite(dx) & np.isfinite(dy) & ((np.abs(dx) + np.abs(dy)) > 1e-9)
        out[finite] = np.degrees(np.arctan2(dy[finite], dx[finite]))
        return out

    shoulder_line = _line_angle(L["shoulder"], R["shoulder"])
    hip_line = _line_angle(L["hip"], R["hip"])
    diff = np.abs(shoulder_line - hip_line)
    diff = np.where(diff > 180.0, 360.0 - diff, diff)
    result["torso_rotation"] = diff

    # Head position relative to the hip midline (torso units): vertical
    # crouch + horizontal lean, robust to body translation in the frame.
    result["head_vertical"] = (nose[:, 1] - mid_hip[:, 1]) / safe
    result["head_lateral"] = (nose[:, 0] - mid_hip[:, 0]) / safe

    # Wrist-to-hip separation (torso units): how far the hands are from the
    # body, an input to bat-angle/front-foot posture measures.
    result["lead_wrist_to_hip"] = (
        _vec_norm(lead_wrist - mid_hip) / safe
    )

    # Front/back foot horizontal travel relative to the hip midline.
    for side in ("left", "right"):
        foot = body[side]["ankle"]
        result[f"{side}_ankle_to_hip_lateral"] = (
            (foot[:, 0] - mid_hip[:, 0]) / safe
        )

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
