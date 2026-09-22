import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

RANDOM_SEED = 42
REPORTS_DIR = Path("reports")
META_COLUMNS = {"video_name"}

# Number of top features to display in the chart.
TOP_N = 15


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    source = Path("output_data") / "cricket_biomechanics_dataset_engineered.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found: {source}\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(source)

    feature_columns = [
        c for c in df.columns
        if c not in META_COLUMNS and c != "shot_type"
    ]
    X_raw = df[feature_columns].astype(float)

    # Drop constant (zero-variance) features (no predictive value).
    constant_cols = [c for c in feature_columns if X_raw[c].nunique() <= 1]
    if constant_cols:
        feature_columns = [c for c in feature_columns if c not in constant_cols]
        X_raw = df[feature_columns].astype(float)

    y = LabelEncoder().fit_transform(df["shot_type"].astype(str))

    print()
    print("===================================")
    print("FEATURE IMPORTANCE")
    print("===================================")
    print(f"Features considered: {len(feature_columns)}")
    print(f"Samples: {len(df)}")

    # ------------------------------------------------------------------
    # RandomForest is used here as the interpretability model because it
    # natively provides feature_importances_ (unlike the selected best
    # LogisticRegression, which has no native feature importance) and it can
    # be used safely on small multi-class data.
    # ------------------------------------------------------------------
    print()
    print("NOTE: The best classifier selected in train_models.py was")
    print("LogisticRegression, which does not expose native feature")
    print("importance. A RandomForest is therefore used here as the")
    print("interpretability model to rank the biomechanics features.")
    print()

    pipeline = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED, n_jobs=1,
    )

    # Impute (median) + scale before fitting the RandomForest. Tree models
    # are scale-invariant but imputation is still needed.
    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X_raw)

    pipeline.fit(X_clean, y)
    importances = pipeline.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    csv_out = REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # ------------------------------------------------------------------
    # Plot top features.
    # ------------------------------------------------------------------
    top = importance_df.head(TOP_N).iloc[::-1]  # ascending for horizontal bar
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(top))))
    ax.barh(top["feature"], top["importance"], color="steelblue")
    ax.set_xlabel("Feature importance (RandomForest)")
    ax.set_ylabel("Biomechanics feature")
    ax.set_title(f"Top {len(top)} Biomechanics Features for Shot "
                 f"Classification\n(RandomForest, n={len(df)} videos)")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    png_out = REPORTS_DIR / "feature_importance.png"
    fig.savefig(png_out, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_out}")

    print()
    print("Top 10 features by importance:")
    print(importance_df.head(10).to_string(index=False))

    print()
    print("IMPORTANT: With this many samples ("
          f"{len(df)}), these importance values are")
    print("unstable and largely exploratory. They should be treated as a")
    print("rough, illustrative ranking - not as reliable evidence about the")
    print("true biomechanical drivers of shot type. Recompute after adding")
    print("more videos.")
    print()
    print("===================================")
    print("FEATURE IMPORTANCE COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()
