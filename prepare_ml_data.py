import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

META_COLUMNS = {"video_name"}

PREPROCESS_DIR = Path("output_data") / "ml_preprocessing"


def main():

    source = Path("output_data") / "cricket_biomechanics_dataset_engineered.csv"

    if not source.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found: {source}\n"
            "Run feature_engineering.py first."
        )

    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source)

    print()
    print("===================================")
    print("MACHINE LEARNING DATA PREPARATION")
    print("===================================")
    print(f"Input: {source}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print()

    # ------------------------------------------------------------------
    # Label encoding of target
    # ------------------------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["shot_type"].astype(str))
    class_names = list(label_encoder.classes_)

    print(f"Target variable: shot_type -> {len(class_names)} classes")
    print(f"Class encoding:")
    for idx, name in enumerate(class_names):
        count = int((df["shot_type"].astype(str) == name).sum())
        print(f"    {name} -> {idx} (n={count})")

    # ------------------------------------------------------------------
    # Remove metadata columns
    # ------------------------------------------------------------------
    feature_columns = [
        c for c in df.columns
        if c not in META_COLUMNS
        and c not in {"shot_type"}
        and not c.startswith("quality_")
    ]

    X_raw = df[feature_columns]

    # Drop constant (zero-variance) features - they carry no discriminative
    # information and can break some feature-selection/scaling routines.
    constant_cols = [c for c in feature_columns if X_raw[c].nunique() <= 1]
    if constant_cols:
        print(f"Dropping constant (zero-variance) features: {constant_cols}")
        feature_columns = [c for c in feature_columns if c not in constant_cols]
        X_raw = df[feature_columns]

    # ------------------------------------------------------------------
    # Handle missing values with median imputation (robust to outliers,
    # suitable for the small dataset; median preserves scale reasonably).
    # ------------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    n_imputed = int(pd.DataFrame(X_imputed, columns=feature_columns)
                    .isna().sum().sum())
    print()
    print(f"Missing values imputed: {n_imputed} cell(s) using median "
          f"imputation.")

    # ------------------------------------------------------------------
    # Scale features (StandardScaler) - tree models are scale-invariant,
    # but distance/length-sensitive models (SVM, kNN, Logistic Regression)
    # require scaling.
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print(f"Features: {len(feature_columns)}")
    print(f"X shape: {X_scaled.shape}, y shape: {y.shape}")
    print()

    # ------------------------------------------------------------------
    # Save preprocessing objects and data arrays
    # ------------------------------------------------------------------
    np.save(PREPROCESS_DIR / "X.npy", X_scaled)
    np.save(PREPROCESS_DIR / "y.npy", y)
    np.save(PREPROCESS_DIR / "video_names.npy", df["video_name"].astype(str).values)

    with open(PREPROCESS_DIR / "feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(feature_columns))

    objects = {
        "label_encoder": label_encoder,
        "imputer": imputer,
        "scaler": scaler,
        "class_names": class_names,
        "feature_columns": feature_columns,
    }
    with open(PREPROCESS_DIR / "preprocessing_objects.pkl", "wb") as f:
        pickle.dump(objects, f)

    print("Saved to:")
    print(f"    X.npy, y.npy, video_names.npy (scaled, imputed arrays)")
    print(f"    feature_names.txt")
    print(f"    preprocessing_objects.pkl (label_encoder, imputer, scaler)")

    print()
    print("IMPORTANT: The sample size is small, so X and y are prepared so")
    print("the later scripts can work, but any model trained on this data")
    print("cannot be interpreted as representing real-world performance.")


if __name__ == "__main__":
    main()
