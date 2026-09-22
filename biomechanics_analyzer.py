import pandas as pd
import numpy as np
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
    # Temporal & scale-normalised features
    # -----------------------------------------
    # These capture *how the body moves* (velocities / accelerations and
    # torso-normalised relative positions) rather than only static angle
    # summaries. Units:
    #   * angles          -> degrees / s (vel) and degrees / s^2 (accel)
    #   * torso-normalised signals (lead_wrist_height, torso_rotation...)
    #     -> per second for velocities; dimensionless for positions.
    ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
    n = len(df)
    if n >= 3:
        is_swing = df["phase"].ne("Stance").to_numpy() if "phase" in df.columns \
            else np.ones(n, dtype=bool)
        # Windows between two consecutive swing frames (velocities need pairs).
        in_swing_pair = (is_swing[:-1] & is_swing[1:])
        dts = np.diff(ts)
        denom = np.where(np.abs(dts) > 1e-6, dts, np.nan)

        def _fvalues(col):
            if col not in df.columns:
                return None
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                vel = np.diff(vals) / denom
                # Acceleration: velocity derivative over the midpoint interval
                # (length n-2, matching diff(vel)).
                dt2 = (ts[2:] - ts[:-2]) / 2.0
                accel = np.diff(vel) / np.where(np.abs(dt2) > 1e-6, dt2, np.nan)
            return vals, vel, accel

        velocity_cols = [c for c in angle_columns if c in df.columns]
        for c in ("lead_elbow_angle", "torso_rotation",
                  "lead_wrist_height", "head_vertical"):
            if c in df.columns and c not in velocity_cols:
                velocity_cols.append(c)

        for col in velocity_cols:
            parsed = _fvalues(col)
            if parsed is None:
                continue
            _, vel, accel = parsed
            v_swing = np.abs(vel[in_swing_pair])
            a_swing = np.abs(accel[in_swing_pair[:-1]])
            v_swing = v_swing[np.isfinite(v_swing)]
            a_swing = a_swing[np.isfinite(a_swing)]
            if v_swing.size >= 2:
                features[f"{col}_swing_mean_abs_vel_dps"] = float(v_swing.mean())
                features[f"{col}_swing_max_abs_vel_dps"] = float(v_swing.max())
            if a_swing.size >= 2:
                features[f"{col}_swing_max_abs_accel_dps2"] = float(a_swing.max())

        # Bat-speed proxy: peak downward rate of the lead wrist (normalised
        # to torso lengths / second) inside Downswing + Impact.
        if "phase" in df.columns and "lead_wrist_height" in df.columns:
            fast = df["phase"].isin(["Downswing", "Impact"]).to_numpy()
            fast_pair = fast[:-1] & fast[1:]
            vals = pd.to_numeric(df["lead_wrist_height"],
                                 errors="coerce").to_numpy(dtype=float)
            vel = np.diff(vals) / denom
            v = np.abs(vel[fast_pair])
            v = v[np.isfinite(v)]
            if v.size >= 2:
                features["bat_speed_torso_per_s"] = float(v.max())
                features["bat_speed_mean_torso_per_s"] = float(v.mean())

        # Lead-wrist travel / lift profile (torso units) across the swing.
        lwh = pd.to_numeric(df.get("lead_wrist_height"),
                            errors="coerce") if "lead_wrist_height" in df.columns else None
        if lwh is not None:
            swing_lwh = lwh.where(pd.Series(is_swing))
            valid = swing_lwh.dropna()
            if valid.size >= 3:
                features["lead_wrist_height_min_swing"] = float(valid.min())
                features["lead_wrist_height_max_swing"] = float(valid.max())
                features["lead_wrist_height_range_swing"] = float(valid.max() - valid.min())
                travel = valid.diff().abs().sum()
                if np.isfinite(travel):
                    features["lead_wrist_total_travel_torso"] = float(travel)

        # Torso-rotation mobility through the swing.
        if "torso_rotation" in df.columns:
            tr = pd.to_numeric(df["torso_rotation"], errors="coerce")
            swing_tr = tr.where(pd.Series(is_swing)).dropna()
            if swing_tr.size >= 3:
                features["torso_rotation_range_swing"] = float(
                    swing_tr.max() - swing_tr.min())
                features["torso_rotation_std_swing"] = float(swing_tr.std())

        # Head vertical excursion (crouch / rise) through the swing.
        if "head_vertical" in df.columns:
            hv = pd.to_numeric(df["head_vertical"], errors="coerce")
            swing_hv = hv.where(pd.Series(is_swing)).dropna()
            if swing_hv.size >= 3:
                features["head_vertical_range_swing"] = float(
                    swing_hv.max() - swing_hv.min())

        # Stance width (medial-lateral ankle separation, torso units).
        if "left_ankle_to_hip_lateral" in df.columns and \
                "right_ankle_to_hip_lateral" in df.columns:
            ldx = pd.to_numeric(df["left_ankle_to_hip_lateral"], errors="coerce")
            rdx = pd.to_numeric(df["right_ankle_to_hip_lateral"], errors="coerce")
            sep = (ldx - rdx).abs()
            stance = sep.where(pd.Series(is_swing)).dropna()
            if stance.size >= 3:
                features["ankle_separation_median_swing"] = float(stance.median())
                features["ankle_separation_max_swing"] = float(stance.max())

    # -----------------------------------------
    # Impact-time relational features (torso-normalised) + stride proxy
    # -----------------------------------------
    if len(impact) > 0:
        impact_row = impact.iloc[len(impact) // 2]
        for col in ("lead_wrist_height", "torso_rotation",
                    "head_vertical", "head_lateral", "lead_wrist_to_hip",
                    "lead_elbow_angle"):
            if col in impact_row.index and pd.notna(impact_row[col]):
                features[f"{col}_at_impact"] = float(impact_row[col])
        for side in ("left", "right"):
            col = f"{side}_ankle_to_hip_lateral_at_impact"
            src = f"{side}_ankle_to_hip_lateral"
            if src in impact_row.index and pd.notna(impact_row[src]):
                features[col] = float(impact_row[src])
        if "left_ankle_to_hip_lateral_at_impact" in features and \
                "right_ankle_to_hip_lateral_at_impact" in features:
            features["ankle_separation_at_impact"] = abs(
                features["left_ankle_to_hip_lateral_at_impact"]
                - features["right_ankle_to_hip_lateral_at_impact"])

        # Forward stride proxy: maximum medial-lateral ankle displacement
        # from stance to impact (torso units) for either foot.
        from_pos = {}
        stance_present = df[df["phase"] == "Stance"] if "phase" in df.columns else df.iloc[0:1]
        for side in ("left", "right"):
            col = f"{side}_ankle_to_hip_lateral"
            if col not in df.columns:
                continue
            base = pd.to_numeric(stance_present[col], errors="coerce").dropna()
            imp = pd.to_numeric(impact[col], errors="coerce").dropna()
            if len(base) and len(imp):
                from_pos[side] = abs(float(imp.median()) - float(base.median()))
        if from_pos:
            features["stride_length_torso"] = max(from_pos.values())
            features["backfoot_stability_torso"] = min(from_pos.values())

    # -----------------------------------------
    # Tracking-quality descriptors (informational, NOT used for classification
    # - train_models drops quality_* columns). They let evaluation report how
    # reliability varies with video conditions (detection coverage, confidence,
    # interpolation rate) without teaching the classifier a shortcut.
    # -----------------------------------------
    landmarks_file = folder / "batting_landmarks.csv"
    if landmarks_file.exists():
        try:
            ldf = pd.read_csv(landmarks_file)
            if len(ldf) and "tracking_ok" in ldf.columns:
                n_frames = int(len(ldf))
                n_tracked = int(pd.to_numeric(ldf["tracking_ok"],
                                              errors="coerce").fillna(0).sum())
                features["quality_frame_count"] = n_frames
                features["quality_tracking_coverage"] = (
                    n_tracked / n_frames if n_frames else 0.0)
                conf = pd.to_numeric(ldf.get("tracking_confidence"),
                                     errors="coerce").dropna()
                features["quality_mean_tracking_confidence"] = (
                    float(conf.mean()) if len(conf) else 0.0)
                if "interpolated" in ldf.columns:
                    features["quality_interpolated_frames"] = int(
                        pd.to_numeric(ldf["interpolated"],
                                      errors="coerce").fillna(0).sum())
        except Exception:
            pass

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
