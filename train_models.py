import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RANDOM_SEED = 42

# Only keep this many top features (by ANOVA F) inside each model pipeline.
# Kept low to avoid overfitting the pathological 9-sample, ~89-feature case.
FEATURE_SELECTION_K = 5

META_COLUMNS = {"video_name"}

PREPROCESS_DIR = Path("output_data") / "ml_preprocessing"
REPORTS_DIR = Path("reports")

MODEL_DEFS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=1,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=2,
        weights="distance",
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
}


def load_data():
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

    # Drop constant (zero-variance) features: no discriminative value, and
    # they can break f_classif / produce misleading results.
    constant_cols = [
        c for c in feature_columns if X_raw[c].nunique() <= 1
    ]
    if constant_cols:
        feature_columns = [
            c for c in feature_columns if c not in constant_cols
        ]
        X_raw = df[feature_columns].astype(float)

    y_raw = df["shot_type"].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return df, X_raw, y, label_encoder, feature_columns


def build_pipeline(model):
    # SelectKBest is embedded inside the pipeline so that feature selection
    # happens on the training fold only (no leakage from the held-out sample).
    # With far more features (89) than samples (9), aggressive feature
    # selection is methodologically necessary to reduce overfitting.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=FEATURE_SELECTION_K)),
        ("clf", model),
    ])


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    df, X_raw, y, label_encoder, feature_columns = load_data()
    class_names = list(label_encoder.classes_)
    n_samples = len(df)
    n_classes = len(class_names)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("===================================")
    print("MACHINE LEARNING MODEL TRAINING")
    print("===================================")
    print(f"Samples (videos):        {n_samples}")
    print(f"Classes (shot types):    {n_classes} -> {class_names}")
    print(f"Features:                {len(feature_columns)}")
    print(f"Class distribution:      {dict(df['shot_type'].value_counts())}")
    print()

    print("EVALUATION STRATEGY")
    print("-------------------")
    print(f"  Dataset is very small ({n_samples} samples, 4 classes). A normal")
    print("  train/test split is NOT statistically valid because the")
    print("  smallest class has only a handful of samples - a stratified split")
    print("  would leave almost no test data and results would be pure noise.")
    print("  => Using Leave-One-Out Cross-Validation (LOOCV): for each of")
    print(f"     the {n_samples} samples, the model is trained on the other")
    print("     samples and tested on the held-out sample.")
    print(f"  => To counter the high feature/sample ratio, only the top")
    print(f"     {FEATURE_SELECTION_K} features (ANOVA F) are selected inside")
    print(f"     each pipeline. Feature selection is fit on the training")
    print(f"     fold only (no leakage from the held-out sample).")
    print()

    # Check whether a plain train/test split is valid at this size.
    min_class_count = df["shot_type"].value_counts().min()
    # For a stratified 70/30 split we would need at least ~2 in the test and
    # ~3 in the train per class; this is clearly not satisfied with counts
    # of 2-3 per class.
    if min_class_count < 4:
        print("  NOTE: minimum class count =", min_class_count, "< 4,")
        print("        so a reliable stratified split is not possible.")
        print("        Defaulting to LOOCV throughout.")
    print()

    loo = LeaveOneOut()

    results = []
    predictions = {}
    models = {}

    for name, base_model in MODEL_DEFS.items():
        print(f"Training {name} ...")
        pipeline = build_pipeline(base_model)

        pred = cross_val_predict(pipeline, X_raw, y, cv=loo)
        predictions[name] = pred

        acc = accuracy_score(y, pred)
        precision = precision_score(
            y, pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y, pred, average="macro", zero_division=0
        )
        f1 = f1_score(y, pred, average="macro", zero_division=0)

        models[name] = pipeline
        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
        })
        print(f"    accuracy={acc:.3f}  precision={precision:.3f}  "
              f"recall={recall:.3f}  f1={f1:.3f}")

    comparison = pd.DataFrame(results).set_index("model")
    comparison_file = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_file)
    print()
    print(f"Saved model comparison: {comparison_file}")

    print()
    print(f"Model comparison (LOOCV on {n_samples} samples):")
    print(comparison.to_string())

    # ------------------------------------------------------------------
    # Select the best model by macro F1 (a robust summary for imbalanced,
    # small, multi-class problems).
    # ------------------------------------------------------------------
    best_name = comparison["f1_macro"].idxmax()
    best_score = comparison.loc[best_name, "f1_macro"]

    print()
    print(f"Best model: {best_name} (macro F1 = {best_score})")

    # ------------------------------------------------------------------
    # Save best model + preprocessing objects for later evaluation.
    # ------------------------------------------------------------------
    best_pipeline = models[best_name]
    # Note: the pipeline contains imputer + scaler, so it is self-contained.
    objects = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "feature_columns": feature_columns,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "metric_used": "macro F1",
    }
    best_file = PREPROCESS_DIR / "best_model.pkl"
    with open(best_file, "wb") as f:
        pickle.dump(objects, f)
    print(f"Saved best model + preprocessing: {best_file}")

    # Save LOOCV predictions for later reference.
    with open(PREPROCESS_DIR / "loo_predictions.pkl", "wb") as f:
        pickle.dump({
            "y_true": y,
            "class_names": class_names,
            "video_names": df["video_name"].astype(str).values,
            "predictions": predictions,
        }, f)

    print()
    print("=" * 70)
    print("HONEST LIMITATIONS (IMPORTANT)")
    print("=" * 70)
    print(f"  - These results are computed on just {n_samples} videos.")
    print("  - Even with aggressive feature selection (top "
          f"{FEATURE_SELECTION_K} features), this sample size is far too few")
    print("    a reliable 4-class classifier over unseen videos.")
    print("  - The numbers above must NOT be interpreted as the model's")
    print("    true real-world accuracy/precision/recall/F1.")
    print("  - They only describe how the model behaved on these specific")
    print("    videos during leave-one-out evaluation.")
    print("  - Add many more videos, then re-run: process_all.py ->",
          "build_dataset.py -> prepare_ml_data.py -> this script.")
    print("=" * 70)


if __name__ == "__main__":
    main()
