import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

REPORTS_EDA = Path("reports") / "eda"

NON_NUMERIC_COLUMNS = {"video_name", "shot_type"}

# Important biomechanics features to focus the analysis on.
IMPORTANT_FEATURES = [
    "left_elbow_angle_at_impact",
    "right_elbow_angle_at_impact",
    "left_knee_angle_at_impact",
    "right_knee_angle_at_impact",
    "left_shoulder_angle_at_impact",
    "right_shoulder_angle_at_impact",
    "left_hip_angle_at_impact",
    "right_hip_angle_at_impact",
    "downswing_duration_ms",
    "maximum_movement_score",
]


def ensure_dirs():
    REPORTS_EDA.mkdir(parents=True, exist_ok=True)


def savefig(fig, name, tight=True):
    path = REPORTS_EDA / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def plot_shot_distribution(df):
    print("[EDA] Shot type distribution")
    counts = df["shot_type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(counts)))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Shot Type Distribution")
    ax.set_xlabel("Shot Type")
    ax.set_ylabel("Number of Videos")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.02, str(int(v)), ha="center")
    ax.set_ylim(0, counts.max() + 1)
    savefig(fig, "shot_type_distribution.png")


def plot_correlation_heatmap(df):
    print("[EDA] Feature correlation heatmap")
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] == 0:
        print("    No numeric columns to correlate.")
        return
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.columns, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.4, label="Correlation")
    ax.set_title("Feature Correlation Heatmap "
                 f"({len(numeric.columns)} features)")
    savefig(fig, "feature_correlation_heatmap.png")


def plot_boxplots(df, features):
    print("[EDA] Boxplots of important features across shot types")
    shot_types = sorted(df["shot_type"].unique())
    n = len(features)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = [df.loc[df["shot_type"] == s, feature].dropna().values
                for s in shot_types]
        bp = ax.boxplot(data, positions=range(1, len(shot_types) + 1))
        ax.set_xticks(range(1, len(shot_types) + 1))
        ax.set_xticklabels(shot_types, rotation=45, ha="right")
        ax.set_title(feature)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    savefig(fig, "feature_boxplots_by_shot_type.png")


def plot_distributions(df, features):
    print("[EDA] Distribution plots of important features")
    n = len(features)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        ax.hist(data.values, bins="auto", color="steelblue",
                edgecolor="white", alpha=0.8)
        ax.set_title(f"{feature}\nn={len(data)}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    savefig(fig, "feature_distributions.png")


def save_summary_statistics(df):
    print("[EDA] Feature summary statistics")
    numeric = df.select_dtypes(include=np.number)
    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    summary_file = REPORTS_EDA / "feature_summary_statistics.csv"
    summary.to_csv(summary_file)
    print(f"    Saved: {summary_file}")
    return summary


def build_summary_txt(df, summary, shot_types):
    print("[EDA] Writing summary.txt")

    total = len(df)
    n_shot = df["shot_type"].nunique()
    distribution = df["shot_type"].value_counts().sort_index()

    # Highest-average-difference feature across shot types (simple proxy).
    numeric = df.select_dtypes(include=np.number)
    grouped_means = df[numeric.columns.tolist()].groupby(
        df["shot_type"]
    ).mean()
    spread = grouped_means.max() - grouped_means.min()
    top_varying = spread.sort_values(ascending=False).head(8)

    lines = []
    lines.append("=" * 70)
    lines.append("EXPLORATORY DATA ANALYSIS - SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset: cricket_biomechanics_dataset_cleaned.csv")
    lines.append(f"Videos (rows): {total}")
    lines.append(f"Shot classes: {n_shot} "
                 f"({', '.join(sorted(shot_types))})")
    lines.append("")
    lines.append("IMPORTANT LIMITATION:")
    lines.append(f"  This dataset currently contains only {total} samples.")
    lines.append("  Any statistical patterns reported here are EXPLORATORY and")
    lines.append("  NOT sufficient for strong generalization. Conclusions are")
    lines.append("  descriptive of these specific videos only.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SHOT DISTRIBUTION")
    lines.append("-" * 70)
    for shot, count in distribution.items():
        lines.append(f"  {shot}: {count}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("FEATURES WITH LARGEST MEAN DIFFERENCE ACROSS SHOT TYPES")
    lines.append(f"  (exploratory; may reflect chance with {total} samples)")
    lines.append("-" * 70)
    for feat, diff in top_varying.items():
        lines.append(f"  {feat}: mean spread = {diff:.2f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SUMMARY OF SUMMARY STATISTICS (selected columns)")
    lines.append("-" * 70)
    focus = [c for c in IMPORTANT_FEATURES if c in summary.index]
    shown = summary.loc[focus,
                        ["mean", "std", "min", "max", "missing"]]
    lines.append(shown.round(2).to_string())
    lines.append("")
    lines.append("-" * 70)
    lines.append("INTERPRETATION NOTE")
    lines.append("-" * 70)
    lines.append("  Correlation heatmap, boxplots and distributions were")
    lines.append("  generated in this folder (see .png files).")
    lines.append(f"  Given n={total}, rely on descriptive inspection only; do not")
    lines.append("  draw inferential conclusions or claim a reliable model.")
    lines.append("  Adding more videos and re-running `python process_all.py`")
    lines.append("  then `python build_dataset.py` will allow a more robust")
    lines.append("  analysis.")

    text = "\n".join(lines)
    summary_file = REPORTS_EDA / "summary.txt"
    summary_file.write_text(text, encoding="utf-8")
    print(f"    Saved: {summary_file}")


def main():
    ensure_dirs()

    dataset_file = Path("output_data") / "cricket_biomechanics_dataset_cleaned.csv"
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {dataset_file}\n"
            "Run data_validation.py first."
        )

    df = pd.read_csv(dataset_file)

    # Ensure only usable important features are plotted.
    features = [c for c in IMPORTANT_FEATURES if c in df.columns]

    print()
    print("===================================")
    print("EXPLORATORY DATA ANALYSIS")
    print("===================================")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Output folder: {REPORTS_EDA.resolve()}")
    print()

    plot_shot_distribution(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df, features)
    plot_distributions(df, features)
    summary = save_summary_statistics(df)
    build_summary_txt(df, summary, sorted(df["shot_type"].unique()))

    print()
    print("===================================")
    print("EDA COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()
