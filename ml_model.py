"""Shot-type classification for a single analysed clip.

This is the *inference* half of the project's shot-classification system.
``train_models.py`` fits the classifiers under leave-one-out CV and persists
the winner to ``output_data/ml_preprocessing/best_model.pkl``; this module
loads that artefact and turns one clip's ``biomechanics_features.csv`` into a
shot label plus a confidence score.

Why "Unknown" is not a hard-coded shot
--------------------------------------
A prediction is only reported as a named shot when the model actually supports
it *and* the model was confident enough about it. Otherwise the clip is
deferred to ``unknown``. The deferral is driven purely by the model's own
posterior distribution and the project's existing
``train_models.CONFIDENCE_THRESHOLD`` - no shot type is ever forced, and
``unknown`` is never used as a stand-in for a particular shot.

Reasons a clip can end up Unknown (all reported in ``reason``):

``model_unavailable``     no trained model / artefact missing / unreadable
``feature_error``         the clip's features could not be engineered
``model_unknown``         the model's own top class is already ``unknown``
``unsupported_class``     top class is not one of the supported shot types
``low_confidence``        top class is supported but below the threshold

Usage:
    python ml_model.py <video_name> [<video_name> ...]

Writes ``output_data/<video_name>/shot_classification.json`` for each clip and
prints a short human-readable summary.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import feature_engineering
from train_models import CONFIDENCE_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "output_data" / "ml_preprocessing" / "best_model.pkl"
OUTPUT_DATA = BASE_DIR / "output_data"

UNKNOWN = "unknown"

# The canonical shot vocabulary. Must stay in sync with app.SHOT_TYPES and
# data_validation.SHOT_TYPES; a model class outside this set is never shown
# as a named shot.
SUPPORTED_SHOT_TYPES = (
    "cut",
    "defence",
    "drive",
    "flick",
    "pull shot",
    UNKNOWN,
)

REASON_LABELS = {
    "ok": "Confident match",
    "model_unavailable": "No trained classifier available",
    "feature_error": "Could not build features for this clip",
    "model_unknown": "Classifier returned Unknown",
    "unsupported_class": "Predicted class is not a supported shot",
    "low_confidence": "Confidence below threshold",
}


def _unknown_result(reason, model_name=None, n_classes=0, confidence=None,
                    probabilities=None, predicted=None):
    """Build the canonical Unknown result payload."""
    return {
        "shot_type": UNKNOWN,
        "predicted": predicted,
        "confidence": confidence,
        "threshold": CONFIDENCE_THRESHOLD,
        "is_unknown": True,
        "reason": reason,
        "reason_label": REASON_LABELS.get(reason, reason),
        "model_name": model_name,
        "n_classes": n_classes,
        "probabilities": probabilities or {},
    }


class ShotClassifier:
    """Loads the persisted LOOCV winner and classifies engineered features."""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.model_name = None
        self.class_names = []
        self.feature_columns = []
        self._pipeline = None
        self.load_error = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            self.load_error = (
                f"Model artefact not found: {self.model_path}. "
                "Run train_models.py to create it."
            )
            return
        try:
            with open(self.model_path, "rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:                      # unreadable / stale pickle
            self.load_error = f"Could not load {self.model_path.name}: {exc}"
            return

        self._pipeline = obj.get("pipeline")
        self.model_name = obj.get("model_name")
        self.class_names = list(obj.get("class_names") or [])
        self.feature_columns = list(obj.get("feature_columns") or [])
        if self._pipeline is None or not self.class_names \
                or not self.feature_columns:
            self._pipeline = None
            self.load_error = (
                f"{self.model_path.name} is missing pipeline/class_names/"
                "feature_columns. Re-run train_models.py."
            )

    @property
    def available(self):
        return self._pipeline is not None

    def classify(self, features):
        """Classify one clip's biomechanics feature row.

        ``features`` is the per-clip ``biomechanics_features.csv`` as a
        DataFrame (one row). Returns a result dict; on any failure it returns
        an Unknown result carrying the reason rather than raising.
        """
        if not self.available:
            return _unknown_result("model_unavailable", self.model_name)

        try:
            engineered = feature_engineering.engineer(features)
        except Exception as exc:
            return _unknown_result("feature_error", self.model_name,
                                   n_classes=len(self.class_names))

        # Align to the exact column order the model was trained on. Anything
        # the clip did not produce becomes NaN and is handled by the
        # pipeline's own imputer.
        try:
            X = engineered.reindex(columns=self.feature_columns)
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.fillna(np.nan).astype(float)
        except Exception as exc:
            return _unknown_result("feature_error", self.model_name,
                                   n_classes=len(self.class_names))

        try:
            proba = self._pipeline.predict_proba(X)[0]
        except Exception:
            return _unknown_result("feature_error", self.model_name,
                                   n_classes=len(self.class_names))

        proba = np.asarray(proba, dtype=float).ravel()
        if proba.size != len(self.class_names):
            return _unknown_result("feature_error", self.model_name,
                                   n_classes=len(self.class_names))

        probabilities = {
            name: float(p) for name, p in zip(self.class_names, proba)
        }
        best_idx = int(np.argmax(proba))
        predicted = self.class_names[best_idx]
        confidence = float(proba[best_idx])

        if predicted == UNKNOWN:
            return _unknown_result("model_unknown", self.model_name,
                                   len(self.class_names), confidence,
                                   probabilities, predicted)

        if predicted not in SUPPORTED_SHOT_TYPES:
            return _unknown_result("unsupported_class", self.model_name,
                                   len(self.class_names), confidence,
                                   probabilities, predicted)

        if confidence < CONFIDENCE_THRESHOLD:
            return _unknown_result("low_confidence", self.model_name,
                                   len(self.class_names), confidence,
                                   probabilities, predicted)

        return {
            "shot_type": predicted,
            "predicted": predicted,
            "confidence": confidence,
            "threshold": CONFIDENCE_THRESHOLD,
            "is_unknown": False,
            "reason": "ok",
            "reason_label": REASON_LABELS["ok"],
            "model_name": self.model_name,
            "n_classes": len(self.class_names),
            "probabilities": probabilities,
        }


_CLASSIFIER = None


def get_classifier(model_path=MODEL_PATH):
    """Return a process-wide cached classifier (loads the pickle once)."""
    global _CLASSIFIER
    if _CLASSIFIER is None or _CLASSIFIER.model_path != Path(model_path):
        _CLASSIFIER = ShotClassifier(model_path)
    return _CLASSIFIER


def classify_video(video_name, output_dir=None, save=True):
    """Classify an already-processed clip by its output folder name.

    Returns the result dict. Missing or unreadable features degrade to an
    Unknown result instead of raising, so a bad clip never breaks the
    pipeline.
    """
    video_name = Path(video_name).stem
    folder = Path(output_dir) / video_name if output_dir else \
        OUTPUT_DATA / video_name
    features_file = folder / "biomechanics_features.csv"
    classifier = get_classifier()

    if not classifier.available:
        result = _unknown_result("model_unavailable", classifier.model_name)
    elif not features_file.exists():
        result = _unknown_result("feature_error", classifier.model_name)
    else:
        try:
            df = pd.read_csv(features_file)
        except Exception:
            df = None
        if df is None or df.empty:
            result = _unknown_result("feature_error", classifier.model_name)
        else:
            result = classifier.classify(df)

    result["video_name"] = video_name
    if save:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / "shot_classification.json", "w",
                      encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
        except OSError:
            pass
    return result


def _fmt_conf(value):
    return "n/a" if value is None else f"{value:.3f}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python ml_model.py <video_name> [<video_name> ...]")
        return 1

    classifier = get_classifier()
    if not classifier.available:
        print("[ml_model] WARNING:", classifier.load_error)
    else:
        print(f"[ml_model] model={classifier.model_name} "
              f"classes={classifier.class_names} "
              f"threshold={CONFIDENCE_THRESHOLD}")

    for name in argv:
        result = classify_video(name)
        print()
        print("=" * 60)
        print(f"VIDEO      : {result['video_name']}")
        print(f"SHOT TYPE  : {result['shot_type']}")
        print(f"PREDICTED  : {result['predicted']}")
        print(f"CONFIDENCE : {_fmt_conf(result['confidence'])} "
              f"(threshold {result['threshold']})")
        print(f"REASON     : {result['reason_label']}")
        if result["probabilities"]:
            print("POSTERIOR  :")
            for cls, p in sorted(result["probabilities"].items(),
                                 key=lambda kv: kv[1], reverse=True):
                mark = "*" if cls == result["predicted"] else " "
                print(f"   {mark} {cls:<12} {p:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
