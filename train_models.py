"""Train and evaluate ML classifiers for cricket shot classification.

Uses Leave-One-Out Cross-Validation (LOOCV) with aggressive feature selection
to handle the small-sample, high-feature-ratio regime. Includes ensemble
methods and adaptive feature selection to improve robustness.

Usage:
    python train_models.py
"""

import os
import inspect
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RANDOM_SEED = 42

# Adaptive feature selection: try multiple K values and pick the best by LOOCV.
# This is more robust than a fixed K when the dataset is tiny.
FEATURE_SELECTION_K_VALUES = [4, 6, 8, 10, 12]

# A prediction is only trusted when the model assigns it at least this
# posterior probability; anything below is reported as LOW CONFIDENCE.
CONFIDENCE_THRESHOLD = 0.55

META_COLUMNS = {"video_name"}

PREPROCESS_DIR = Path("output_data") / "ml_preprocessing"
REPORTS_DIR = Path("reports")

MODEL_DEFS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
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
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        C=0.5,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_SEED,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=1,
        max_depth=3,
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
    ),
    "GaussianNB": GaussianNB(),
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
        and not c.startswith("quality_")
    ]
    X_raw = df[feature_columns].astype(float)

    # Drop constant (zero-variance) features
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
        sample_weight = 0.5 + 0.5 * cov.values

    return df, X_raw, y, label_encoder, feature_columns, sample_weight


def build_pipeline(model, k=6):
    """Build a pipeline with imputer, scaler, feature selection, and classifier."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", model),
    ])


def loocv_fit_predict(pipeline, X_raw, y, sample_weight, n_classes):
    """Run LOOCV with sample weight support.

    Returns (pred, proba, proba_ok) where proba is indexed by global label index.
    """
    final_est = pipeline.steps[-1][1]
    supports_weights = "sample_weight" in inspect.signature(
        final_est.fit).parameters

    n = len(y)
    pred = np.empty(n, dtype=int)
    proba = np.zeros((n, n_classes), dtype=float)
    proba_ok = np.zeros(n, dtype=bool)
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
        X_test = (
            X_raw.iloc[test_idx] if hasattr(X_raw, "iloc") else X_raw[test_idx]
        )
        pred[test_idx] = model.predict(X_test)
        row = int(test_idx[0])
        try:
            p = np.asarray(model.predict_proba(X_test)[0],
                           dtype=float).ravel()
            classes = np.asarray(model.classes_)
            if len(classes) != len(p):
                raise ValueError(
                    f"predict_proba returned {len(p)} column(s) for "
                    f"{len(classes)} class(es)"
                )
            for position, label in enumerate(classes):
                proba[row, int(label)] = p[position]
            proba_ok[row] = True
        except Exception as exc:
            print(f"    [warn] no posterior for sample {row} "
                  f"(class {int(pred[row])}): {exc}")
            proba[row, :] = 0.0
            proba_ok[row] = False
    return pred, proba, proba_ok


def evaluate_model(name, pipeline, X_raw, y, sample_weight, n_classes, class_names):
    """Evaluate a single model with LOOCV and return metrics dict."""
    pred, proba, proba_ok = loocv_fit_predict(
        pipeline, X_raw, y, sample_weight, n_classes)

    conf = np.zeros(len(pred), dtype=float)
    for i, (row, cls) in enumerate(zip(proba, pred)):
        if proba_ok[i] and 0 <= int(cls) < proba.shape[1]:
            conf[i] = row[int(cls)]
    posterior_unavailable = int((~proba_ok).sum())

    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, average="macro", zero_division=0)
    recall = recall_score(y, pred, average="macro", zero_division=0)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y, pred, average="weighted", zero_division=0)

    uncertain = int((conf < CONFIDENCE_THRESHOLD).sum())

    print(f"    accuracy={acc:.3f}  precision={precision:.3f}  "
          f"recall={recall:.3f}  f1={f1:.3f}  "
          f"mean_conf={conf.mean():.3f}  "
          f"low_conf={uncertain}  "
          f"no_posterior={posterior_unavailable}")

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "f1_weighted": round(f1_weighted, 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "low_confidence_count": uncertain,
        "posterior_unavailable": posterior_unavailable,
        "pred": pred,
        "proba": proba,
        "proba_ok": proba_ok,
        "conf": conf,
    }


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
    print(f"  Dataset: {n_samples} samples, {n_classes} classes, "
          f"{len(feature_columns)} features")
    print("  => Leave-One-Out Cross-Validation (LOOCV)")
    print(f"  => Adaptive feature selection over K = {FEATURE_SELECTION_K_VALUES}")
    print(f"  => Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Find the best feature-selection K across all models
    # ------------------------------------------------------------------
    print("PHASE 1: Adaptive feature selection")
    print("-------------------------------------")
    best_k = FEATURE_SELECTION_K_VALUES[0]
    best_k_score = -1.0
    for k in FEATURE_SELECTION_K_VALUES:
        if k > len(feature_columns):
            continue
        # Use a simple model to evaluate each K
        pipe = build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_SEED), k=k)
        pred, _, _ = loocv_fit_predict(pipe, X_raw, y, sample_weight, n_classes)
        acc = accuracy_score(y, pred)
        print(f"  K={k:2d}  LOOCV accuracy={acc:.3f}")
        if acc > best_k_score:
            best_k_score = acc
            best_k = k
    print(f"  => Best K = {best_k} (accuracy={best_k_score:.3f})")
    print()

    # ------------------------------------------------------------------
    # Phase 2: Train and evaluate all models with the best K
    # ------------------------------------------------------------------
    print("PHASE 2: Model training with best K")
    print("-------------------------------------")
    results = []
    predictions = {}
    probabilities = {}
    proba_raw = {}
    posterior_available = {}
    models = {}

    for name, base_model in MODEL_DEFS.items():
        print(f"Training {name} (K={best_k}) ...")
        pipeline = build_pipeline(base_model, k=best_k)
        metrics = evaluate_model(
            name, pipeline, X_raw, y, sample_weight, n_classes, class_names)

        predictions[name] = metrics["pred"]
        probabilities[name] = metrics["conf"]
        proba_raw[name] = metrics["proba"]
        posterior_available[name] = metrics["proba_ok"]
        models[name] = pipeline

        results.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "mean_confidence": metrics["mean_confidence"],
            "low_confidence_count": metrics["low_confidence_count"],
            "posterior_unavailable": metrics["posterior_unavailable"],
        })

    comparison = pd.DataFrame(results).set_index("model")
    comparison_file = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_file)
    print()
    print(f"Saved model comparison: {comparison_file}")
    print()
    print(f"Model comparison (LOOCV on {n_samples} samples, K={best_k}):")
    print(comparison.to_string())

    # ------------------------------------------------------------------
    # Phase 3: Build ensemble from top models
    # ------------------------------------------------------------------
    print()
    print("PHASE 3: Ensemble model")
    print("-----------------------")
    # Select top 3 models by macro F1 for the ensemble
    top_models = comparison.nlargest(min(3, len(comparison)), "f1_macro")
    estimators = []
    for model_name in top_models.index:
        estimators.append((model_name.lower(), models[model_name]))

    if len(estimators) >= 2:
        ensemble = VotingClassifier(
            estimators=estimators,
            voting="soft",
        )
        # Wrap in a pipeline for imputation + scaling
        ensemble_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", ensemble),
        ])
        print(f"  Ensemble members: {list(top_models.index)}")
        ensemble_metrics = evaluate_model(
            "Ensemble", ensemble_pipeline, X_raw, y, sample_weight,
            n_classes, class_names)

        predictions["Ensemble"] = ensemble_metrics["pred"]
        probabilities["Ensemble"] = ensemble_metrics["conf"]
        proba_raw["Ensemble"] = ensemble_metrics["proba"]
        posterior_available["Ensemble"] = ensemble_metrics["proba_ok"]
        models["Ensemble"] = ensemble_pipeline

        results.append({
            "model": "Ensemble",
            "accuracy": ensemble_metrics["accuracy"],
            "precision_macro": ensemble_metrics["precision_macro"],
            "recall_macro": ensemble_metrics["recall_macro"],
            "f1_macro": ensemble_metrics["f1_macro"],
            "f1_weighted": ensemble_metrics["f1_weighted"],
            "mean_confidence": ensemble_metrics["mean_confidence"],
            "low_confidence_count": ensemble_metrics["low_confidence_count"],
            "posterior_unavailable": ensemble_metrics["posterior_unavailable"],
        })

        comparison = pd.DataFrame(results).set_index("model")
        comparison.to_csv(comparison_file)
        print()
        print(f"Updated model comparison with ensemble:")
        print(comparison.to_string())

    # ------------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------------
    best_score = comparison["f1_macro"].max()
    tied = comparison.index[comparison["f1_macro"] == best_score].tolist()
    if len(tied) > 1:
        tie_order = comparison.loc[tied].sort_values(
            by=["mean_confidence", "low_confidence_count",
                "posterior_unavailable"],
            ascending=[False, True, True],
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
    # Save best model + preprocessing objects
    # ------------------------------------------------------------------
    best_pipeline = models[best_name]
    objects = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "feature_columns": feature_columns,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "metric_used": "macro F1",
        "feature_selection_k": best_k,
    }
    best_file = PREPROCESS_DIR / "best_model.pkl"
    with open(best_file, "wb") as f:
        pickle.dump(objects, f)
    print(f"Saved best model + preprocessing: {best_file}")

    # Save LOOCV predictions
    with open(PREPROCESS_DIR / "loo_predictions.pkl", "wb") as f:
        pickle.dump({
            "y_true": y,
            "class_names": class_names,
            "video_names": df["video_name"].astype(str).values,
            "predictions": predictions,
            "confidence": probabilities,
            "probas_raw": proba_raw,
            "posterior_available": posterior_available,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }, f)

    print()
    print("=" * 70)
    print("HONEST LIMITATIONS (IMPORTANT)")
    print("=" * 70)
    print(f"  - These results are computed on just {n_samples} videos.")
    print(f"  - Even with aggressive feature selection (top {best_k} features),")
    print("    this sample size is far too few for a reliable classifier.")
    print("  - The numbers above must NOT be interpreted as the model's")
    print("    true real-world accuracy/precision/recall/F1.")
    print("  - They only describe how the model behaved on these specific")
    print("    videos during leave-one-out evaluation.")
    print("  - Add many more videos, then re-run: process_all.py ->")
    print("    build_dataset.py -> feature_engineering.py -> this script.")
    print("=" * 70)


if __name__ == "__main__":
    main()
