import numpy as np
import pandas as pd
from pathlib import Path


SHOT_TYPES = {"cut", "defence", "drive", "flick", "pull shot", "unknown"}

# Non-numeric (metadata/label) columns that are intentionally kept as strings.
NON_NUMERIC_COLUMNS = {"video_name", "shot_type"}


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run build_dataset.py first to create it."
        )
    return pd.read_csv(path)


def main():

    output_root = Path("output_data")
    dataset_file = output_root / "cricket_biomechanics_dataset.csv"
    cleaned_file = output_root / "cricket_biomechanics_dataset_cleaned.csv"

    df = load_dataset(dataset_file)

    print()
    print("===================================")
    print("DATA VALIDATION REPORT")
    print("===================================")
    print(f"Input: {dataset_file}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print()

    issues = 0

    def report(title, count):
        nonlocal issues
        if count > 0:
            issues += 1
            print(f"  [ISSUE] {title}: {count}")

    # ------------------------------------------------------------------
    # 1. Missing values
    # ------------------------------------------------------------------
    print(f"[1] Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("    None found.")
    else:
        total_missing = int(missing.sum())
        print(f"    {total_missing} missing cell(s) across "
              f"{len(missing)} column(s):")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"      - {col}: {count} ({pct:.1f}%)")
        report("missing values", total_missing)

    # ------------------------------------------------------------------
    # 2. Duplicate rows
    # ------------------------------------------------------------------
    print(f"[2] Duplicate rows")
    dup_rows = int(df.duplicated().sum())
    if dup_rows == 0:
        print("    None found.")
    else:
        print(f"    Found {dup_rows} fully duplicated row(s).")
        report("duplicate rows", dup_rows)

    # ------------------------------------------------------------------
    # 3. Duplicate video names
    # ------------------------------------------------------------------
    print(f"[3] Duplicate video names")
    dup_videos = df["video_name"].duplicated().sum()
    if dup_videos == 0:
        print("    None found.")
    else:
        print(f"    Found {dup_videos} duplicate video_name(s).")
        report("duplicate video names", dup_videos)

    # ------------------------------------------------------------------
    # 4. Non-numeric / invalid values in numerical columns
    # ------------------------------------------------------------------
    print(f"[4] Non-numeric values in numerical columns")
    numeric_cols = [
        c for c in df.columns if c not in NON_NUMERIC_COLUMNS
    ]
    non_numeric_report = []
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & converted.isna()
        if bad.any():
            non_numeric_report.append((col, int(bad.sum())))
    if not non_numeric_report:
        print("    None found.")
    else:
        for col, count in non_numeric_report:
            print(f"      - {col}: {count} non-numeric value(s)")
        report("non-numeric values", len(non_numeric_report))

    # ------------------------------------------------------------------
    # 5. Infinite values
    # ------------------------------------------------------------------
    print(f"[5] Infinite values")
    inf_numeric = df[numeric_cols].apply(
        lambda s: np.isinf(pd.to_numeric(s, errors="coerce")).sum()
    )
    inf_total = int(inf_numeric.sum())
    if inf_total == 0:
        print("    None found.")
    else:
        inf_cols = inf_numeric[inf_numeric > 0]
        for col, count in inf_cols.items():
            print(f"      - {col}: {count} infinite value(s)")
        report("infinite values", inf_total)

    # ------------------------------------------------------------------
    # 6. Invalid shot_type labels
    # ------------------------------------------------------------------
    print(f"[6] Invalid shot_type labels")
    valid_shot = df["shot_type"].isin(SHOT_TYPES)
    invalid_shot = df["shot_type"].dropna()
    invalid_count = int((~valid_shot).sum())
    if invalid_count == 0:
        print("    All labels are within the expected set: "
              f"{sorted(SHOT_TYPES)}")
    else:
        bad_labels = invalid_shot[~invalid_shot.isin(SHOT_TYPES)].unique()
        print(f"    Invalid labels found: {list(bad_labels)}")
        report("invalid shot_type labels", invalid_count)

    # ------------------------------------------------------------------
    # 7. Frame-count columns stored as float but semantically integer
    # ------------------------------------------------------------------
    print(f"[7] Frame counts stored as floats (cosmetic)")
    frame_cols = [
        c for c in numeric_cols
        if c.endswith("_frames") or c.endswith("_frame")
    ]
    float_frame_cols = [
        c for c in frame_cols if pd.api.types.is_float_dtype(df[c])
    ]
    if not float_frame_cols:
        print("    None found.")
    else:
        for col in float_frame_cols:
            print(f"      - {col}: float (may contain NaN)")

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    print()
    print("===================================")
    print("CLEANING")
    print("===================================")

    cleaned = df.copy()

    drop_rows = 0
    drop_cols = []

    # Drop fully duplicated rows (exact duplicates are redundant).
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    drop_rows += before - len(cleaned)

    if dup_rows > 0:
        print(f"    Removed {dup_rows} fully duplicated row(s).")

    # Drop duplicate video_name rows (keep the first occurrence).
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset="video_name", keep="first")
    drop_rows += before - len(cleaned)

    if dup_videos > 0:
        print(f"    Removed {dup_videos} duplicate video_name row(s).")

    # Drop fully-empty columns (all values missing) - useless and
    # non-informative, so removing is lossless rather than data loss.
    empty_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
    cleaned = cleaned.drop(columns=empty_cols)
    drop_cols.extend(empty_cols)
    if empty_cols:
        print(f"    Dropped fully empty column(s): {empty_cols}")

    # For the remaining small number of missing values (frame counts like
    # backlift_frames / stance_frames), fill with 0. These are count features:
    # a missing value means that phase was not detected for that video, i.e.
    # the phase did not occur (count = 0). This is an informed, interpretable
    # imputation rather than arbitrary deletion.
    frame_cols_to_fill = [
        c for c in frame_cols if c in cleaned.columns
    ]
    for col in frame_cols_to_fill:
        n_na = int(cleaned[col].isna().sum())
        if n_na > 0:
            cleaned[col] = cleaned[col].fillna(0)
            cleaned[col] = cleaned[col].astype("Int64")
            print(
                f"    Filled {n_na} missing value(s) in '{col}' with 0 "
                f"(phase not detected -> count 0) and cast to integer."
            )

    # Rate features where 0 is a legitimate "nothing measured" value (no valid
    # swing frames -> no swing speed). 0 is a meaningful absence here.
    zero_fill_allowlist = {
        "bat_speed_torso_per_s",
        "bat_speed_mean_torso_per_s",
    }

    # Caps throughput of per-column missing data. Features missing MORE than
    # this fraction of the (already tiny) dataset cannot be estimated
    # reliably - median/zero imputation would fabricate biomechanics values
    # the tracker never saw. Such columns are dropped instead of faked.
    MAX_MISSING_FRACTION = 0.30

    columns_to_drop = []
    for col in cleaned.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        n_na = int(cleaned[col].isna().sum())
        if n_na == 0:
            continue
        frac = n_na / len(cleaned)
        if frac > MAX_MISSING_FRACTION:
            columns_to_drop.append(col)
    cleaned = cleaned.drop(columns=columns_to_drop)
    if columns_to_drop:
        drop_cols.extend(columns_to_drop)
        print(
            f"    Dropped {len(columns_to_drop)} column(s) missing in more "
            f"than {MAX_MISSING_FRACTION:.0%} of samples (unreliable to "
            f"estimate at n={len(cleaned)}):"
        )
        for col in columns_to_drop:
            print(f"      - {col}")

    # Replace remaining missing numeric values: median of the observed values
    # for continuous posture/angle features (never a fabricated 0, which would
    # read as "wrist at ground level" or "joint angle 0 deg"), and 0 only for
    # the explicit count/rate allowlist above. Record which cells were filled
    # so the imputation is transparent.
    filled_cells = 0
    filled_details = []
    for col in cleaned.columns:
        if col in NON_NUMERIC_COLUMNS:
            continue
        n_na = int(cleaned[col].isna().sum())
        if n_na == 0:
            continue
        if col in zero_fill_allowlist:
            fill_value = 0
            reason = "no measurable value (0 is meaningful absence)"
        else:
            fill_value = cleaned[col].median()
            reason = "median of observed values"
        before_ids = cleaned.index[cleaned[col].isna()].tolist()
        cleaned[col] = cleaned[col].fillna(fill_value)
        filled_cells += n_na
        filled_details.append((col, n_na, fill_value, reason, before_ids))
        print(
            f"    Filled {n_na} missing value(s) in '{col}' with "
            f"{fill_value:.4g} ({reason})."
        )

    if filled_details:
        np.save(
            output_root / "imputation_log.npy",
            np.array(
                [(c, n, f"{v:.4g}", r) for c, n, v, r, _ in filled_details],
                dtype=object,
            ),
            allow_pickle=True,
        )
        print(f"    Saved imputation log to {output_root / 'imputation_log.npy'}")

    # Replace infinite values with NaN then fill with 0.
    numeric_cols = [c for c in cleaned.columns if c not in NON_NUMERIC_COLUMNS]
    inf_found = int(cleaned[numeric_cols].apply(
        lambda s: np.isinf(pd.to_numeric(s, errors="coerce")).sum()
    ).sum())
    if inf_found > 0:
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.fillna(0)
        print(f"    Replaced {inf_found} infinite value(s) with NaN, then 0.")

    cleaned = cleaned.reset_index(drop=True)

    print()
    print("    Cleaning summary:")
    print(f"      Rows removed: {drop_rows}")
    print(f"      Columns removed: {len(drop_cols)}")
    print(f"      Original shape: {df.shape}")
    print(f"      Cleaned shape: {cleaned.shape}")

    cleaned.to_csv(cleaned_file, index=False)
    print()
    print(f"Saved cleaned dataset: {cleaned_file}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    print("===================================")
    print("FINAL OVERVIEW")
    print("===================================")
    print(f"Rows: {len(cleaned)}")
    print(f"Columns: {len(cleaned.columns)}")
    print(f"Shot distribution:")
    print(cleaned["shot_type"].value_counts())
    print()

    if issues == 0:
        print("Dataset passed basic validation (clean).")
    else:
        print(f"Validation identified {issues} category/categories "
              f"of concern that were addressed above.")

    print()
    print(f"NOTE: This dataset contains only {len(cleaned)} samples after")
    print("deduplication. The small sample size limits any downstream")
    print("statistical analysis and machine learning.")


if __name__ == "__main__":
    main()
