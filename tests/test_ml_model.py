"""Unit tests for ml_model.py."""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import ml_model
from ml_model import ShotClassifier, _unknown_result, classify_video, get_classifier


class TestUnknownResult:
    """Tests for the _unknown_result helper function."""

    def test_basic_unknown(self):
        result = _unknown_result("low_confidence", "TestModel", 6, 0.4, {"cut": 0.4, "drive": 0.6}, "drive")
        assert result["shot_type"] == "unknown"
        assert result["predicted"] == "drive"
        assert result["confidence"] == 0.4
        assert result["threshold"] == ml_model.CONFIDENCE_THRESHOLD
        assert result["is_unknown"] is True
        assert result["reason"] == "low_confidence"
        assert result["model_name"] == "TestModel"
        assert result["n_classes"] == 6

    def test_minimal_unknown(self):
        result = _unknown_result("model_unavailable")
        assert result["shot_type"] == "unknown"
        assert result["predicted"] is None
        assert result["confidence"] is None
        assert result["probabilities"] == {}


class TestShotClassifier:
    """Tests for the ShotClassifier class."""

    def test_load_missing_model(self):
        """Should handle missing model file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = ShotClassifier(model_path=Path(tmpdir) / "missing.pkl")
            assert not classifier.available
            assert "not found" in classifier.load_error

    def test_load_corrupted_model(self):
        """Should handle corrupted pickle gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad.pkl"
            bad_path.write_bytes(b"not a pickle")
            classifier = ShotClassifier(model_path=bad_path)
            assert not classifier.available
            assert "Could not load" in classifier.load_error

    def test_load_incomplete_model(self):
        """Should handle model missing required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "incomplete.pkl"
            with open(bad_path, "wb") as f:
                pickle.dump({"pipeline": None}, f)
            classifier = ShotClassifier(model_path=bad_path)
            assert not classifier.available
            assert "missing pipeline" in classifier.load_error.lower()

    def test_classify_unavailable(self):
        """classify should return unknown when model unavailable."""
        classifier = ShotClassifier(model_path=Path("/nonexistent.pkl"))
        features = pd.DataFrame([{"dummy": 1.0}])
        result = classifier.classify(features)
        assert result["shot_type"] == "unknown"
        assert result["reason"] == "model_unavailable"


class TestClassifyVideo:
    """Tests for the classify_video function."""

    def test_missing_features_file(self, tmp_path):
        """Should return unknown when features file missing."""
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()

        result = classify_video("test_video", output_dir=tmp_path, save=False)
        assert result["shot_type"] == "unknown"
        assert result["reason"] == "feature_error"

    def test_empty_features_file(self, tmp_path):
        """Should return unknown when features file empty."""
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()
        (video_dir / "biomechanics_features.csv").write_text("col1\n")

        with patch("ml_model.get_classifier") as mock_get:
            mock_classifier = MagicMock()
            mock_classifier.available = True
            mock_get.return_value = mock_classifier

            result = classify_video("test_video", output_dir=tmp_path, save=False)
            assert result["shot_type"] == "unknown"
            assert result["reason"] == "feature_error"

    def test_saves_json_when_save_true(self, tmp_path):
        """Should write shot_classification.json when save=True."""
        video_dir = tmp_path / "test_video"
        video_dir.mkdir()

        # Create a valid features file
        features_df = pd.DataFrame({
            "feature1": [1.0],
            "feature2": [2.0],
        })
        features_df.to_csv(video_dir / "biomechanics_features.csv", index=False)

        with patch("ml_model.get_classifier") as mock_get:
            mock_classifier = MagicMock()
            mock_classifier.available = True
            mock_classifier.classify.return_value = {
                "shot_type": "drive",
                "predicted": "drive",
                "confidence": 0.8,
                "threshold": 0.55,
                "is_unknown": False,
                "reason": "ok",
                "reason_label": "Confident match",
                "model_name": "TestModel",
                "n_classes": 6,
                "probabilities": {"drive": 0.8, "cut": 0.2},
            }
            mock_get.return_value = mock_classifier

            result = classify_video("test_video", output_dir=tmp_path, save=True)

            # Check file was written
            json_path = video_dir / "shot_classification.json"
            assert json_path.exists()

            with open(json_path) as f:
                saved = json.load(f)
            assert saved["shot_type"] == "drive"


class TestSupportedShotTypes:
    """Tests for SUPPORTED_SHOT_TYPES constant."""

    def test_contains_expected_shots(self):
        expected = {"cut", "defence", "drive", "flick", "pull shot", "unknown"}
        assert set(ml_model.SUPPORTED_SHOT_TYPES) == expected

    def test_unknown_included(self):
        assert "unknown" in ml_model.SUPPORTED_SHOT_TYPES


class TestReasonLabels:
    """Tests for REASON_LABELS mapping."""

    def test_all_reasons_have_labels(self):
        reasons = [
            "ok", "model_unavailable", "feature_error",
            "model_unknown", "unsupported_class", "low_confidence"
        ]
        for reason in reasons:
            assert reason in ml_model.REASON_LABELS
            assert isinstance(ml_model.REASON_LABELS[reason], str)
            assert len(ml_model.REASON_LABELS[reason]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])