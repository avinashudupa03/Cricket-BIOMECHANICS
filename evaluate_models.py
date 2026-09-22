import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

PREPROCESS_DIR = Path("output_data") / "ml_preprocessing"
REPORTS_DIR = Path("reports")


def load_artifacts():
    best_file = PREPROCESS_DIR / "best_model.pkl"
    loo_file = PREPROCESS_DIR / "loo_predictions.pkl"

    if not best_file.exists() or not loo_file.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run train_models.py first."
        )

    with open(best_file, "rb") as f:
        import pickle
        best = pickle.load(f)

    with open(loo_file, "rb") as f:
        loo = pickle.load(f)

    return best, loo


def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (LOOCV)\n{model_name} - n={len(y_true)}")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out}")
    return cm


def save_classification_report(y_true, y_pred, class_names, model_name):
    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=class_names, zero_division=0,
    )
    text = (
        "CLASSIFICATION REPORT\n"
        "=====================\n"
        f"Model : {model_name}\n"
        f"Evaluation: Leave-One-Out CV on n={len(y_true)} samples\n"
        f"Classes: {class_names}\n"
        f"Accuracy on LOO predictions: "
        f"{accuracy_score(y_true, y_pred):.4f}\n\n"
        f"{report}\n"
        "NOTE: This sample size is small, so the report is exploratory and NOT\n"
        "indicative of real-world performance.\n"
    )
    out = REPORTS_DIR / "classification_report.txt"
    out.write_text(text, encoding="utf-8")
    print(f"    Saved: {out}")
    return text


def save_model_summary(best, y_true, y_pred, class_names, cm, report_text):
    n = best["n_samples"]
    n_classes = best["n_classes"]
    acc = accuracy_score(y_true, y_pred)

    lines = []
    lines.append("=" * 72)
    lines.append("MODEL RESULTS SUMMARY (RESEARCH ORIENTED)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Task            : Cricket batting shot classification")
    lines.append(f"Target          : shot_type "
                 f"(--- one of {class_names})")
    lines.append(f"Dataset size    : {n} videos (samples)")
    lines.append(f"Shot classes    : {n_classes} "
                 f"({', '.join(class_names)})")
    lines.append(f"Evaluation method: Leave-One-Out Cross-Validation (LOOCV)")
    lines.append(f"Best model      : {best['model_name']} "
                 f"(selected by macro F1)")
    lines.append(f"LOOCV accuracy  : {acc:.4f} "
                 f"({int((y_true == y_pred).sum())}/{n} correct)")
    lines.append("")
    lines.append("-" * 72)
    lines.append("LIMITATIONS OF THE SMALL DATASET")
    lines.append("-" * 72)
    lines.append(f"  * Only {n} unique samples across 4 classes means the "
                 f"model cannot be")
    lines.append("    relied upon to generalise to unseen videos.")
    lines.append("  * Results hover around chance level (~25% for 4 balanced")
    lines.append("    classes); any apparent pattern is likely coincidence.")
    lines.append("  * Feature/sample ratio is extremely high even after")
    lines.append("    feature selection, so models overfit: high training")
    lines.append("    accuracy with poor out-of-sample behaviour is expected.")
    lines.append("  * Report metrics (accuracy / macro-F1) describe how the")
    lines.append("    model behaved on THESE videos only.")
    lines.append("  * MORE VIDEOS ARE REQUIRED for reliable generalisation.")
    lines.append("    Suggested minimum per class for a first usable model:")
    lines.append("    ~15-30+ per class, with a held-out test set.")
    lines.append("")
    lines.append("-" * 72)
    lines.append("HOW TO EXTEND")
    lines.append("-" * 72)
    lines.append("  1) Add new videos under input_videos/<shot_type>/")
    lines.append("  2) Run:  python process_all.py")
    lines.append("  3) Run:  python build_dataset.py")
    lines.append("  4) Re-run the data->ML pipeline (see below).")
    lines.append("")
    lines.append("Full pipeline after adding videos:")
    lines.append("  1. python process_all.py")
    lines.append("  2. python build_dataset.py")
    lines.append("  3. python data_validation.py")
    lines.append("  4. python eda.py")
    lines.append("  5. python feature_engineering.py")
    lines.append("  6. python prepare_ml_data.py")
    lines.append("  7. python train_models.py")
    lines.append("  8. python evaluate_models.py")
    lines.append("")
    lines.append("Confusion matrix & classification report are also saved in")
    lines.append("this folder (confusion_matrix.png, classification_report.txt).")
    lines.append("")
    lines.append("=" * 72)

    text = "\n".join(lines)
    out = REPORTS_DIR / "model_results_summary.txt"
    out.write_text(text, encoding="utf-8")
    print(f"    Saved: {out}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best, loo = load_artifacts()
    model_name = best["model_name"]
    class_names = best["class_names"]
    n = best["n_samples"]

    print()
    print("===================================")
    print("MODEL EVALUATION")
    print("===================================")
    print(f"Best model: {model_name}")
    print(f"Samples: {n}, Classes: {class_names}")
    print(f"Using Leave-One-Out predictions from training.\n")

    y_true = loo["y_true"]
    predictions = loo["predictions"]
    y_pred = np.asarray(predictions[model_name])

    cm = save_confusion_matrix(y_true, y_pred, class_names, model_name)
    report_text = save_classification_report(
        y_true, y_pred, class_names, model_name,
    )
    save_model_summary(
        best, y_true, y_pred, class_names, cm, report_text,
    )

    print()
    print("===================================")
    print("EVALUATION COMPLETE")
    print("===================================")
    for name in ["confusion_matrix.png", "classification_report.txt",
                 "model_results_summary.txt"]:
        print(f"    {REPORTS_DIR / name}")


if __name__ == "__main__":
    main()
