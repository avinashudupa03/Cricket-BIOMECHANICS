import os
import inspect
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
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
FEATURE_SELECTION_K = 6

# A prediction is only trusted when the model assigns it at least this
# posterior probability; anything below is reported as LOW CONFIDENCE rather
# than a confident guess (the model prefers to defer than be wrong).
CONFIDENCE_THRESHOLD = 0.55

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
        probability=True,
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
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=2,
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
        # quality_* descriptors describe recording/video quality (coverage,
        # confidence). They are informational and excluded from classification
        # so the model cannot use "how well we tracked it" as a shortcut.
        and not c.startswith("quality_")
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

    sample_weight = None
    if "quality_tracking_coverage" in df.columns:
        cov = pd.to_numeric(df["quality_tracking_coverage"], errors="coerce")
        cov = cov.fillna(0.0).clip(lower=0.0, upper=1.0)
        # Poorly-tracked videos (low coverage) contribute noisy, imputed
        # features, so they should influence the classifier less than clean
        # videos. Weight ranges 0.5 (0% coverage) -> 1.0 (100% coverage);
        # no sample is ever zeroed out - we down-weight, not discard.
        sample_weight = 0.5 + 0.5 * cov.values

    return df, X_raw, y, label_encoder, feature_columns, sample_weight


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


def loocv_fit_predict(pipeline, X_raw, y, sample_weight, n_classes):
    """Run leave-one-out manually so that sample weights can be passed to
    the classifier (cross_val_predict does not support fit_params when
    using a Pipeline with per-step weights). Weights come from tracking
    quality and apply to the training fold only - the held-out sample is
    always predicted with the classifier's single-sample output."""
    final_est = pipeline.steps[-1][1]
    supports_weights = "sample_weight" in inspect.signature(
        final_est.fit).parameters

    n = len(y)
    pred = np.empty(n, dtype=int)
    proba = np.zeros((n, n_classes), dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_raw):
        fit_kwargs = {}
        if supports_weights and sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = sample_weight[train_idx]
        model = pipeline.fit(
            X_raw.iloc[train_idx] if hasattr(X_raw, "iloc") else X_raw[train_idx],
            y[train_idx],
            **fit_kwargs,
        )
        pred[test_idx] = model.predict(
            X_raw.iloc[test_idx] if hasattr(X_raw, "iloc") else X_raw[test_idx])
        try:
            p = model.predict_proba(
                X_raw.iloc[test_idx] if hasattr(X_raw, "iloc") else X_raw[test_idx])[0]
            # Align columns to full class order (predict_proba may omit unseen
            # classes in a tiny training fold).
            for cls_idx, cls in enumerate(model.classes_):
                proba[test_idx[0], cls_idx] = p[cls]
        except Exception:
            proba[test_idx[0], int(pred[test_idx[0]])] = 1.0
    return pred, proba


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    df, X_raw, y, label_encoder, feature_columns, sample_weight = load_data()
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

    results = []
    predictions = {}
    probabilities = {}   # max predicted probability per sample (confidence)
    proba_raw = {}       # full (n_samples, n_classes) posterior matrices
    models = {}

    for name, base_model in MODEL_DEFS.items():
        print(f"Training {name} ...")
        pipeline = build_pipeline(base_model)

        pred, proba = loocv_fit_predict(
            pipeline, X_raw, y, sample_weight, n_classes)
        predictions[name] = pred

        # Posterior confidence: the probability the model assigned to its
        # chosen class. SVC now runs with probability=True; all other models
        # expose predict_proba natively.
        try:
            if hasattr(proba, "shape") and proba.ndim == 2:
                proba_raw[name] = proba
                conf = np.array([
                    p[int(c)] if int(c) < len(p) else max(p)
                    for p, c in zip(proba, pred)
                ])
            else:
                proba_raw[name] = None
                conf = np.ones(len(pred))
        except Exception:
            proba_raw[name] = None
            conf = np.ones(len(pred))
        probabilities[name] = conf
        uncertain = int((conf < CONFIDENCE_THRESHOLD).sum())

        acc = accuracy_score(y, pred)
        precision = precision_score(
            y, pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y, pred, average="macro", zero_division=0
        )
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y, pred, average="weighted", zero_division=0)

        models[name] = pipeline
        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
            "f1_weighted": round(f1_weighted, 4),
            "mean_confidence": round(float(conf.mean()), 4),
            "low_confidence_count": uncertain,
        })
        print(f"    accuracy={acc:.3f}  precision={precision:.3f}  "
              f"recall={recall:.3f}  f1={f1:.3f}  "
              f"mean_conf={conf.mean():.3f}  "
              f"low_conf={uncertain}")

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
    # small, multi-class problems). When several models tie on macro F1,
    # break the tie by higher mean confidence (a model that is more often
    # sure of its answer is preferable to an equally-accurate one that is
    # not), then by fewer low-confidence calls.
    # ------------------------------------------------------------------
    best_score = comparison["f1_macro"].max()
    tied = comparison.index[comparison["f1_macro"] == best_score].tolist()
    if len(tied) > 1:
        tie_order = comparison.loc[tied].sort_values(
            by=["mean_confidence", "low_confidence_count"],
            ascending=[False, True],
        )
        best_name = tie_order.index[0]
        print()
        print(f"  NOTE: models tied on macro F1 = {best_score}: {tied}")
        print(f"        tie broken on mean confidence -> {best_name}")
    else:
        best_name = comparison["f1_macro"].idxmax()

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
            "confidence": probabilities,
            "probas_raw": proba_raw,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
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
