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
    precision_score,
    recall_score,
    f1_score,
)

PREPROCESS_DIR = Path("output_data") / "ml_preprocessing"
REPORTS_DIR = Path("reports")
DATASET_CSV = Path("output_data") / "cricket_biomechanics_dataset_engineered.csv"


def load_artifacts():
    best_file = PREPROCESS_DIR / "best_model.pkl"
    loo_file = PREPROCESS_DIR / "loo_predictions.pkl"

    if not best_file.exists() or not loo_file.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run train_models.py first."
        )

    import pickle
    with open(best_file, "rb") as f:
        best = pickle.load(f)

    with open(loo_file, "rb") as f:
        loo = pickle.load(f)

    return best, loo


VALIDITY_HEADER = (
    "\n"
    "SCIENTIFIC VALIDITY\n"
    "-------------------\n"
    "  * ML shot labels (MODEL_PREDICTION) are classifier outputs computed on\n"
    "    n=9 videos. They carry self-reported CONFIDENCE scores that are NOT\n"
    "    accuracy measures and are expected to be frequently wrong.\n"
    "  * All biomechanical quantities feeding the classifier are ESTIMATED_2D/\n"
    "    ESTIMATED_3D values from single-camera pose estimation - NOT clinical\n"
    "    measurements. See reports/scientific_validity.txt and each video\n"
    "    folder's measurement_provenance.txt for the full classification.\n"
)


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


def _top2_accuracy(proba, y_true, class_names):
    """Proportion of samples whose true class is in the top-2 predicted."""
    if proba is None or proba.ndim != 2:
        return None
    top2 = np.argsort(-proba, axis=1)[:, :2]
    return float((top2 == y_true[:, None]).any(axis=1).mean())


BODYPART_FEATURE_MAP = [
    ("hand/wrist", ("wrist", "bat")),
    ("elbow", ("elbow",)),
    ("shoulder", ("shoulder",)),
    ("hip", ("hip",)),
    ("knee", ("knee",)),
    ("ankle/leg", ("ankle", "leg")),
    ("torso", ("torso",)),
    ("head", ("head",)),
    ("other", ("",)),
]


def _bodypart(feature):
    for _ in ("left_", "right_", "lead_", "trail_"):
        feature = feature.replace(_, "")
    for part, keys in BODYPART_FEATURE_MAP:
        if any(k in feature for k in keys):
            return part
    return "other"


def save_quality_stratified(loo, y_true, y_pred):
    """Per-bucket metrics broken down by source-video tracking quality.

    Classifier performance on a real deployment depends heavily on how
    reliable the pose signal was. Videos whose pose tracking was weak
    (low coverage / low mean keypoint confidence) are expected to be the
    hard cases. This section reports accuracy, macro F1 and sample counts
    inside each quality bucket so improvements to the pipeline can be
    traced to genuinely difficult footage rather than lucky samples.
    """
    lines = []
    quality = loo.get("quality", {})
    video_names = loo["video_names"]
    threshold = loo.get("confidence_threshold", 0.55)

    conf_src = quality.get("mean_tracking_confidence")
    cov_src = quality.get("tracking_coverage")
    if conf_src is None or cov_src is None:
        try:
            df = pd.read_csv(DATASET_CSV)
            df = df.set_index("video_name")
            conf_src = [float(df.loc[str(v), "quality_mean_tracking_confidence"])
                        for v in video_names]
            cov_src = [float(df.loc[str(v), "quality_tracking_coverage"])
                       for v in video_names]
        except Exception:
            lines.append("(video-quality breakdown unavailable: engineered "
                         "dataset not found)")
            return "\n".join(lines)
    try:
        conf_src = [float(x) for x in conf_src]
        cov_src = [float(x) for x in cov_src]
    except Exception:
        return "\n".join(lines)

    lines.append("Performance by video quality")
    lines.append("-" * 68)
    lines.append("Bucketed on the SOURCE video's pose-tracking quality (not on")
    lines.append("model confidence). High/med/low use fixed 0.5/0.2 cut-offs.")
    lines.append("")
    rows = []
    for i, name in enumerate(video_names):
        qual = "high"
        if conf_src[i] < 0.2 or cov_src[i] < 0.2:
            qual = "low"
        elif conf_src[i] < 0.5 or cov_src[i] < 0.5:
            qual = "med"
        rows.append((qual, int(y_true[i]), int(y_pred[i])))

    for qual in ("high", "med", "low"):
        bucket = [r for r in rows if r[0] == qual]
        n = len(bucket)
        if not n:
            lines.append(f"  {qual:<5}: n=0")
            continue
        acc = accuracy_score([r[1] for r in bucket], [r[2] for r in bucket])
        f1 = f1_score([r[1] for r in bucket], [r[2] for r in bucket],
                      average="macro", zero_division=0)
        low_c = sum(1 for i, r in enumerate(rows) if r[0] == qual
                    and conf_src[i] < threshold)
        lines.append(f"  {qual:<5}: n={n:<3} accuracy={acc:.3f} "
                     f"macro-F1={f1:.3f} low-model-conf={low_c}")
    if not any(r[0] == "low" for r in rows):
        lines.append("  (no 'low' quality videos in this dataset yet)")
    lines.append("")
    lines.append("  Camera-angle breakdown is NOT reported: the dataset does")
    lines.append("  not record camera position. Horizontal flip is not used")
    lines.append("  (reverse hands change bat side and shot meaning).")
    return "\n".join(lines)


def load_landmark_visibility(video_name):
    """Mean per-region landmark visibility for one video (0..1)."""
    region_keys = {
        "head": ["nose", "eye", "ear", "face"],
        "shoulder": ["shoulder"],
        "elbow": ["elbow"],
        "wrist/hand": ["wrist", "index", "pinky", "thumb"],
        "hip": ["hip"],
        "knee": ["knee"],
        "ankle/foot": ["ankle", "heel", "foot"],
    }
    path = Path("output_data") / str(video_name) / "batting_landmarks.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    means = {}
    for region, keys in region_keys.items():
        cols = [c for c in df.columns
                if "visibility" in c and any(k in c for k in keys)]
        if cols:
            vals = df[cols].to_numpy(dtype=float).ravel()
            vals = vals[~np.isnan(vals)]
            if vals.size:
                means[region] = float(vals.mean())
    return means


def append_deep_dive(lines, best, loo, y_true, y_pred):
    """Append overfitting, class-sufficiency and landmark-contribution data."""
    class_names = best["class_names"]
    video_names = loo["video_names"]

    # --- 1) Class support and per-class performance -----------------------
    lines.append("Class-level data sufficiency")
    lines.append("-" * 60)
    per_class = {}
    for i in range(len(class_names)):
        mask = y_true == i
        n = int(mask.sum())
        hits = int((mask & (y_pred == y_true)).sum())
        per_class[class_names[i]] = (n, hits)
    for cls, (n, hits) in per_class.items():
        lines.append(f"  {cls:<12}: n={n}  correct={hits} "
                     f"({hits / n:.2f} if n>0)" if n else
                     f"  {cls:<12}: n=0  (POLICY: no such class)")
    lines.append("  Classes with fewer than ~5 samples cannot support a")
    lines.append("  reliable decision boundary - treat per-class numbers as")
    lines.append("  anecdotal, not as evidence of class difficulty.")
    lines.append("")

    # --- 2) Overfitting check: in-sample (train) vs LOOCV performance -----
    lines.append("Overfitting check")
    lines.append("-" * 60)
    try:
        df = pd.read_csv(DATASET_CSV)
        feature_columns = best.get("feature_columns") or [
            c for c in df.columns
            if c not in {"video_name", "shot_type"}
            and not c.startswith("quality_")]
        X_raw = df[feature_columns].astype(float)
        pipe = best["pipeline"]
        pipe.fit(X_raw, loo["y_true"])
        train_acc = float((pipe.predict(X_raw) == loo["y_true"]).mean())
        loo_acc = float(np.mean(y_true == y_pred))
        gap = train_acc - loo_acc
        lines.append(f"  In-sample (train) accuracy : {train_acc:.4f}")
        lines.append(f"  LOOCV (hold-out) accuracy  : {loo_acc:.4f}")
        lines.append(f"  Gap (overfit indicator)    : {gap:+.4f}")
        if gap > 0.25:
            lines.append("  => STRONG overfitting: the model memorises the training")
            lines.append("     videos and does not generalise. Expected at this dataset")
            lines.append("     size; improvements need MORE DATA, not a bigger model.")
        elif gap > 0.05:
            lines.append("  => Mild overfitting.")
        else:
            lines.append("  => No statistically meaningful generalisation is possible")
            lines.append("     with 9 samples - the gap itself is noise.")
    except Exception as exc:
        lines.append(f"  (overfitting check failed: {exc})")
    lines.append("")

    # --- 3) Which pose regions were poorly tracked / correlate with errors -
    lines.append("Landmark / tracking contribution to errors")
    lines.append("-" * 60)
    reg_rows = []
    for i, name in enumerate(video_names):
        vis = load_landmark_visibility(name)
        if vis is None:
            continue
        reg_rows.append((name, int(y_true[i]) == int(y_pred[i]), vis))
    if not reg_rows:
        lines.append("  (batting_landmarks.csv not available per video)")
    else:
        regions = sorted({k for _, _, v in reg_rows for k in v})
        lines.append(f"  {'region':<12}{'avg(match)':>12}{'avg(mismatch)':>15}"
                     f"{'delta':>9}   n_match/n_mis")
        for r in regions:
            # Not every clip exposes the same landmark regions (short videos
            # can drop e.g. ankle/foot), so a region present in one video may
            # be absent in another. Skip the missing entries instead of
            # raising KeyError.
            m = [v[r] for _, ok, v in reg_rows if ok and r in v]
            x = [v[r] for _, ok, v in reg_rows if not ok and r in v]
            if not m or not x:
                continue
            am, ax = sum(m) / len(m), sum(x) / len(x)
            lines.append(f"  {r:<12}{am:>11.3f}{ax:>14.3f}{am - ax:>8.3f}   "
                         f"{len(m)}/{len(x)}")
        lines.append("  Regions with a large negative delta are poorly tracked")
        lines.append("  on misclassified videos - improving that tracking")
        lines.append("  (better preprocessing / camera) is a concrete target.")
    lines.append("")

    # --- 4) Class imbalance -------------------------------------------------
    counts = {class_names[i]: int((loo["y_true"] == i).sum())
              for i in range(len(class_names))}
    mx, mn = max(counts.values()), min(counts.values())
    lines.append("Class imbalance")
    lines.append("-" * 60)
    lines.append(f"  Class counts        : {counts}")
    lines.append(f"  Imbalance ratio     : {mx / mn:.2f}:1 (max/min)")
    if mx / mn >= 2.0:
        lines.append("  => Noticeable imbalance; the weakest classes need more")
        lines.append("     videos before the model can be trusted on them.")
    else:
        lines.append("  => Mild imbalance; not the main accuracy driver here.")
    lines.append("")
    return lines


def save_error_analysis(best, loo, y_true, y_pred, conf, model_name):
    class_names = best["class_names"]
    video_names = loo["video_names"]
    threshold = loo.get("confidence_threshold", 0.55)

    cm = confusion_matrix(y_true, y_pred)
    lines = []
    lines.append("=" * 72)
    lines.append("ERROR ANALYSIS")
    lines.append("=" * 72)
    lines.append(f"Model: {model_name} (LOOCV, n={len(y_true)})")
    lines.append("")
    lines.append(VALIDITY_HEADER)
    lines.append(f"Per-sample predictions (low confidence -> UNKNOWN)")
    lines.append("-" * 40)
    lines.append(f"{'video':<14}{'true':<12}{'pred':<12}{'conf':>7}  correct")
    lines.append("-" * 40)
    for i, name in enumerate(video_names):
        tru = class_names[int(y_true[i])]
        c = float(conf[i])
        if c < threshold:
            pre = "UNKNOWN"
            correct = "deferred"
        else:
            pre = class_names[int(y_pred[i])]
            correct = "YES" if y_true[i] == y_pred[i] else "no"
        lines.append(f"{str(name):<14}{tru:<12}{pre:<12}{c:>6.2f}   {correct}")
    lines.append("")

    # After deferring low-confidence samples, only confident predictions are
    # counted as classified. Report the classified-only accuracy separately.
    n_classified = int((conf >= threshold).sum())
    if n_classified > 0:
        ok = y_true[conf >= threshold] == y_pred[conf >= threshold]
        classified_acc = float(ok.mean())
        lines.append(
            f"Of {len(y_true)} videos, {n_classified} were classified with "
            f"confidence >= {threshold:.2f};")
        lines.append(
            f"accuracy on those classified-only videos: "
            f"{classified_acc:.4f} "
            f"(the rest are reported as UNKNOWN / deferred, not wrong).")
        lines.append("")

    # Class pairs most commonly confused (off-diagonal cells tied to the
    # dominant confusion in each row).
    lines.append("Most-confused class pairs (corrections needed per pair)")
    lines.append("-" * 60)
    n_cls = len(class_names)
    confusion_tuples = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j and cm[i, j] > 0:
                confusion_tuples.append((cm[i, j], class_names[i],
                                         class_names[j]))
    confusion_tuples.sort(reverse=True)
    if confusion_tuples:
        for count, tru, pre in confusion_tuples:
            lines.append(f"  {tru:<12} -> {pre:<12} ({count} sample(s))")
    else:
        lines.append("  (no cross-class confusions)")
    lines.append("")

    # Uncertainty report.
    lines.append("Confidence / uncertainty report")
    lines.append("-" * 60)
    lines.append(f"Confidence threshold        : {threshold:.2f}")
    low_mask = conf < threshold
    lines.append(f"Below threshold (LOW CONF.) : {low_mask.sum()} / "
                 f"{len(conf)} sample(s)")
    if low_mask.any():
        wrong_low = int((low_mask & (y_pred != y_true)).sum())
        right_low = int((low_mask & (y_pred == y_true)).sum())
        lines.append(f"  ...of which  wrong: {wrong_low}   right: {right_low}")
        lines.append("  LOW-CONFIDENCE predictions should be surfaced as")
        lines.append("  'unknown/low confidence' rather than a forced class.")
    lines.append("")
    if len(conf):
        conf_sorted = np.sort(conf)
        lines.append("Confidence distribution")
        for pct in (0.0, 0.25, 0.5, 0.75, 1.0):
            idx = min(int(pct * (len(conf_sorted) - 1)), len(conf_sorted) - 1)
            lines.append(f"  {pct * 100:3.0f}% quantile : {conf_sorted[idx]:.3f}")
        lines.append(f"  mean       : {conf.mean():.3f}")
    lines.append("")

    append_deep_dive(lines, best, loo, y_true, y_pred)

    lines.append("Why misclassifications happen (hypotheses to test)")
    lines.append("-" * 60)
    lines.append("  * Several shot types share very similar body kinematics")
    lines.append("    (drive vs flick vs cut all extend the lead arm through")
    lines.append("    the ball), so static angle summaries overlap heavily.")
    lines.append("  * Classes with few samples (2/class) cannot support a")
    lines.append("    robust decision boundary - support is simply too low.")
    lines.append("  * Camera angle / distance differences change the apparent")
    lines.append("    motion even for the same shot (frame-normalised, not")
    lines.append("    world-normalised, coordinates).")
    lines.append("  * Only LOOCV on THIS dataset is measured; no held-out set")
    lines.append("    exists yet, so none of these numbers are real-world")
    lines.append("    performance.")

    out = REPORTS_DIR / "error_analysis.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    Saved: {out}")


def save_model_summary(best, loo, y_true, y_pred, conf, metrics, class_names,
                       cm, model_name, quality_text=""):
    n = best["n_samples"]
    n_classes = best["n_classes"]
    acc = metrics["accuracy"]
    threshold = loo.get("confidence_threshold", 0.55)
    low_count = int((conf < threshold).sum())

    lines = []
    lines.append("=" * 72)
    lines.append("MODEL RESULTS SUMMARY (RESEARCH ORIENTED)")
    lines.append("=" * 72)
    lines.append("")
    lines.append(VALIDITY_HEADER)
    lines.append(f"Task            : Cricket batting shot classification")
    lines.append(f"Target          : shot_type "
                 f"(--- one of {class_names})")
    lines.append(f"Dataset size    : {n} videos (samples)")
    lines.append(f"Shot classes    : {n_classes} "
                 f"({', '.join(class_names)})")
    lines.append(f"Evaluation method: Leave-One-Out Cross-Validation (LOOCV)")
    lines.append(f"Best model      : {model_name} "
                 f"(selected by macro F1)")
    lines.append(f"LOOCV accuracy  : {acc:.4f} "
                 f"({int((y_true == y_pred).sum())}/{n} correct)")
    lines.append("")
    lines.append("Detailed metrics (LOOCV)")
    lines.append("-" * 40)
    lines.append(f"  macro precision   : {metrics['precision_macro']:.4f}")
    lines.append(f"  macro recall      : {metrics['recall_macro']:.4f}")
    lines.append(f"  macro F1          : {metrics['f1_macro']:.4f}")
    lines.append(f"  weighted F1       : {metrics['f1_weighted']:.4f}")
    lines.append(f"  top-2 accuracy    : "
                 f"{metrics['top2_accuracy']:.4f}"
                 if metrics["top2_accuracy"] is not None
                 else f"  top-2 accuracy    : (not available)")
    lines.append(f"  mean confidence   : {conf.mean():.4f}"
                 if len(conf) else "  mean confidence   : (n/a)")
    lines.append(f"  low-confidence    : {low_count} sample(s) below "
                 f"{threshold:.2f}")
    lines.append("")
    if quality_text.strip():
        lines.append(quality_text)
        lines.append("")
    lines.append("-" * 72)
    lines.append("LIMITATIONS OF THE SMALL DATASET")
    lines.append("-" * 72)
    lines.append(f"  * Only {n} unique samples across {n_classes} classes means the")
    lines.append("    model cannot be relied upon to generalise to unseen videos.")
    lines.append("  * Results hover around chance level and any apparent pattern")
    lines.append("    may be coincidence.")
    lines.append("  * Feature/sample ratio is extremely high even after feature")
    lines.append("    selection, so models overfit.")
    lines.append("  * Reported accuracy/F1 describe how the model behaved on THESE")
    lines.append("    videos only (LOOCV) - NOT real-world performance.")
    lines.append("  * MORE VIDEOS ARE REQUIRED for reliable generalisation.")
    lines.append("  * Deep learning (LSTM/GRU/temporal-CNN/Transformer) is NOT")
    lines.append("    appropriate here: with only 9 videos a single fold would")
    lines.append("    train on 8 clips, far too few to learn a temporal")
    lines.append("    representation. The tabular models compared (RandomForest,")
    lines.append("    SVM, KNN, LogisticRegression, GradientBoosting) are the")
    lines.append("    right complexity for n=9.")
    lines.append("")
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
    lines.append("Error / confidence details: error_analysis.txt")
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

    y_true = np.asarray(loo["y_true"])
    predictions = loo["predictions"]
    y_pred = np.asarray(predictions[model_name])

    conf = np.asarray(loo.get("confidence", {}).get(model_name,
                       np.ones(len(y_true))))
    proba = loo.get("probas_raw", {}).get(model_name)

    n_cls = len(class_names)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": _top2_accuracy(proba, y_true, class_names),
    }

    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Macro P/R/F1  : {metrics['precision_macro']:.4f} / "
          f"{metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}")
    print(f"Weighted F1   : {metrics['f1_weighted']:.4f}")
    if metrics["top2_accuracy"] is not None:
        print(f"Top-2 accuracy: {metrics['top2_accuracy']:.4f}")
    print(f"Mean confidence: {conf.mean():.4f}")

    cm = save_confusion_matrix(y_true, y_pred, class_names, model_name)
    report_text = save_classification_report(
        y_true, y_pred, class_names, model_name,
    )
    save_error_analysis(best, loo, y_true, y_pred, conf, model_name)
    quality_text = save_quality_stratified(loo, y_true, y_pred)
    save_model_summary(
        best, loo, y_true, y_pred, conf, metrics, class_names,
        cm, model_name, quality_text,
    )
    print()
    print(f"Confusion matrix ({cm.shape[0]}x{cm.shape[1]}):")
    print(cm)

    print()
    print("===================================")
    print("EVALUATION COMPLETE")
    print("===================================")
    for name in ["confusion_matrix.png", "classification_report.txt",
                 "error_analysis.txt", "model_results_summary.txt"]:
        print(f"    {REPORTS_DIR / name}")


if __name__ == "__main__":
    main()