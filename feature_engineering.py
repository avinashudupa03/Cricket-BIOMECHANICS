import numpy as np
import pandas as pd
from pathlib import Path

NON_NUMERIC_COLUMNS = {"video_name", "shot_type"}

# Standard biomechanics features that already exist per joint.
JOINTS = ["elbow", "knee", "shoulder", "hip"]


def _pair_cols(joint, metric):
    return [f"left_{joint}_{metric}", f"right_{joint}_{metric}"]


def _safe_div(a, b):
    """Divide a by b, returning NaN where b is (near) zero."""
    return np.where(np.abs(b) > 1e-9, a / b, np.nan)


def engineer(df):
    df = df.copy()

    helpers = {}

    # ------------------------------------------------------------------
    # 1. Symmetry differences: |left - right| for each joint metric.
    #    Small values => symmetric loading; large values => asymmetry.
    # ------------------------------------------------------------------
    for joint in JOINTS:
        # At impact
        l, r = _pair_cols(joint, "angle_at_impact")
        helpers[f"{joint}_symmetry_diff_at_impact"] = (
            (df[l] - df[r]).abs()
        )
        # Mean over the whole stroke
        l, r = _pair_cols(joint, "angle_mean")
        helpers[f"{joint}_symmetry_diff_mean"] = (
            (df[l] - df[r]).abs()
        )

    # ------------------------------------------------------------------
    # 2. Angle ranges per joint (max - min), reflecting mobility of that
    #    joint through the swing. A larger range => greater range of motion.
    # ------------------------------------------------------------------
    for joint in JOINTS:
        for side in ["left", "right"]:
            cmin = f"{side}_{joint}_angle_min"
            cmax = f"{side}_{joint}_angle_max"
            helpers[f"{side}_{joint}_angle_range"] = (
                df[cmax] - df[cmin]
            )

    # Bilateral (mean of left/right) range as a combined mobility measure.
    for joint in JOINTS:
        l = f"left_{joint}_angle_range"
        r = f"right_{joint}_angle_range"
        helpers[f"{joint}_angle_range_bilateral"] = (
            helpers[l] + helpers[r]
        ) / 2.0

    # ------------------------------------------------------------------
    # 3. Aggregate limb mobility (sum of ranges) - a simple, interpretable
    #    measure of how much the arm/leg "travels" during the swing.
    # ------------------------------------------------------------------
    helpers["left_arm_mobility"] = (
        helpers["left_elbow_angle_range"]
        + helpers["left_shoulder_angle_range"]
    )
    helpers["right_arm_mobility"] = (
        helpers["right_elbow_angle_range"]
        + helpers["right_shoulder_angle_range"]
    )
    helpers["left_leg_mobility"] = (
        helpers["left_knee_angle_range"]
        + helpers["left_hip_angle_range"]
    )
    helpers["right_leg_mobility"] = (
        helpers["right_knee_angle_range"]
        + helpers["right_hip_angle_range"]
    )
    helpers["arm_mobility_bilateral"] = (
        helpers["left_arm_mobility"] + helpers["right_arm_mobility"]
    ) / 2.0
    helpers["leg_mobility_bilateral"] = (
        helpers["left_leg_mobility"] + helpers["right_leg_mobility"]
    ) / 2.0

    # ------------------------------------------------------------------
    # 4. Movement intensity / explosion measures. A higher maximum relative
    #    to average smooth movement indicates a sharper, more explosive
    #    (sudden) movement profile.
    # ------------------------------------------------------------------
    helpers["movement_peak_to_average_ratio"] = _safe_div(
        df["maximum_movement_score"],
        df["average_movement_score"],
    )
    helpers["movement_intensity"] = (
        df["maximum_movement_score"] * df["average_movement_score"]
    )

    # ------------------------------------------------------------------
    # 5. Normalised phase duration ratios. Guard division by zero safely
    #    (returns NaN) and only define ratios where the denominator is
    #    meaningful (i.e. the phase was actually detected).
    # ------------------------------------------------------------------
    helpers["downswing_to_followthrough_ratio"] = _safe_div(
        df["downswing_duration_ms"],
        df["followthrough_duration_ms"],
    )
    helpers["impact_to_downswing_frames_ratio"] = _safe_div(
        df["impact_frames"],
        df["downswing_frames"],
    )
    helpers["downswing_frames_ratio"] = _safe_div(
        df["downswing_frames"],
        df["downswing_frames"] + df["followthrough_frames"],
    )
    helpers["followthrough_share"] = _safe_div(
        df["followthrough_frames"],
        df["downswing_frames"] + df["followthrough_frames"],
    )

    # ------------------------------------------------------------------
    # 6. Joint "bilateral mean" angles at impact - a compact representation
    #    of the mid-line body posture at the instant of contact.
    # ------------------------------------------------------------------
    for joint in JOINTS:
        l = f"left_{joint}_angle_at_impact"
        r = f"right_{joint}_angle_at_impact"
        helpers[f"{joint}_mean_at_impact"] = (df[l] + df[r]) / 2.0

    # ------------------------------------------------------------------
    # 6b. Temporal intensity features (derived from the swing velocity /
    #     acceleration signals added by biomechanics_analyzer).
    # ------------------------------------------------------------------
    def _col(name):
        return name if name in df.columns else None

    lel = _col("left_elbow_swing_max_abs_vel_dps")
    rel = _col("right_elbow_swing_max_abs_vel_dps")
    if lel and rel:
        helpers["elbow_vel_symmetry_diff"] = (df[lel] - df[rel]).abs()
        helpers["elbow_vel_bilateral_mean"] = (df[lel] + df[rel]) / 2.0
    if lel and _col("left_shoulder_swing_max_abs_vel_dps"):
        helpers["left_arm_vel_elbow_to_shoulder"] = _safe_div(
            df[lel], df["left_shoulder_swing_max_abs_vel_dps"])
    if rel and _col("right_shoulder_swing_max_abs_vel_dps"):
        helpers["right_arm_vel_elbow_to_shoulder"] = _safe_div(
            df[rel], df["right_shoulder_swing_max_abs_vel_dps"])

    if _col("bat_speed_torso_per_s"):
        helpers["bat_speed_explosiveness"] = df["bat_speed_torso_per_s"]
        if _col("bat_speed_mean_torso_per_s"):
            helpers["bat_speed_peak_to_mean_ratio"] = _safe_div(
                df["bat_speed_torso_per_s"], df["bat_speed_mean_torso_per_s"])

    if _col("lead_elbow_swing_max_abs_vel_dps"):
        helpers["lead_arm_swing_vel"] = df["lead_elbow_swing_max_abs_vel_dps"]
        if _col("bat_speed_torso_per_s"):
            helpers["bat_speed_to_lead_elbow_vel"] = _safe_div(
                df["bat_speed_torso_per_s"], df["lead_elbow_swing_max_abs_vel_dps"])

    # Impact posture relative to the swing range: is the impact posture a
    # distinct, deliberate one or does it sit inside the swing's normal range?
    if _col("lead_wrist_height_at_impact") and _col("lead_wrist_height_range_swing"):
        helpers["lead_wrist_impact_relative"] = _safe_div(
            df["lead_wrist_height_at_impact"], df["lead_wrist_height_range_swing"])
    if _col("torso_rotation_at_impact") and _col("torso_rotation_range_swing"):
        helpers["torso_rotation_impact_relative"] = _safe_div(
            df["torso_rotation_at_impact"], df["torso_rotation_range_swing"])
    if _col("stride_length_torso") and _col("ankle_separation_median_swing"):
        helpers["stride_to_stance_width_ratio"] = _safe_div(
            df["stride_length_torso"], df["ankle_separation_median_swing"])
    if _col("stride_length_torso") and _col("backfoot_stability_torso"):
        helpers["stride_forward_to_backfoot_ratio"] = _safe_div(
            df["stride_length_torso"] + 1e-9,
            df["backfoot_stability_torso"] + 1e-9)
    if _col("head_vertical_range_swing") and _col("torso_rotation_at_impact"):
        helpers["head_crouch_to_trunk_rotation"] = _safe_div(
            df["head_vertical_range_swing"], df["torso_rotation_at_impact"])

    # ------------------------------------------------------------------
    # 7. Convert every helper into a clean numeric Series.
    # ------------------------------------------------------------------
    engineered_cols = {}
    for name, series in helpers.items():
        engineered_cols[name] = pd.to_numeric(
            series, errors="coerce"
        )

    for name, series in engineered_cols.items():
        df[name] = series

    # Validate every non-metadata column is numeric.
    non_numeric = [
        c for c in df.columns
        if c not in NON_NUMERIC_COLUMNS and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(
            "The following feature columns are not numeric: "
            f"{non_numeric}"
        )

    return df


def main():
    output_root = Path("output_data")
    source = output_root / "cricket_biomechanics_dataset_cleaned.csv"
    target = output_root / "cricket_biomechanics_dataset_engineered.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {source}\n"
            "Run data_validation.py first."
        )

    df = pd.read_csv(source)

    print()
    print("===================================")
    print("FEATURE ENGINEERING")
    print("===================================")
    print(f"Input: {source}")
    print(f"Rows: {len(df)}, Base columns: {len(df.columns)}")

    before = set(df.columns)
    engineered = engineer(df)
    after = set(engineered.columns)
    new_cols = sorted(after - before)

    print(f"New engineered features added: {len(new_cols)}")
    for c in new_cols:
        print(f"    + {c}")

    engineered.to_csv(target, index=False)

    print()
    print(f"Total columns: {len(engineered.columns)}")
    print(f"Metadata kept: "
          f"{sorted(NON_NUMERIC_COLUMNS.intersection(engineered.columns))}")
    print(f"Saved: {target}")

    n_cols_with_missing = int(engineered.isna().any().sum())
    if n_cols_with_missing:
        print()
        print("NOTE: Columns with NaN (from division-by-zero guards or "
              "missing phases):")
        miss = engineered.isna().sum()
        miss = miss[miss > 0]
        for c, count in miss.items():
            print(f"    {c}: {count} NaN(s)")


if __name__ == "__main__":
    main()
