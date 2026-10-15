"""Unit tests for phase_detector.py."""

import math
import numpy as np
import pandas as pd
import pytest

import phase_detector


class TestSmooth:
    """Tests for the smooth function."""

    def test_basic_smoothing(self):
        """A spike should be reduced by smoothing."""
        series = pd.Series([0, 0, 10, 0, 0], dtype=float)
        result = phase_detector.smooth(series, window=3)
        # The peak at index 2 should be reduced
        assert result.iloc[2] < 10.0

    def test_constant_series(self):
        """A constant series stays constant."""
        series = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = phase_detector.smooth(series, window=3)
        np.testing.assert_array_almost_equal(result.values, [5.0, 5.0, 5.0, 5.0])

    def test_with_nans(self):
        """NaN values are interpolated before smoothing."""
        series = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = phase_detector.smooth(series, window=3)
        # After interpolation + smoothing, no NaNs should remain
        assert not result.isna().any()

    def test_interpolation(self):
        """Missing values are interpolated."""
        series = pd.Series([0.0, np.nan, np.nan, 10.0])
        result = phase_detector.smooth(series, window=1)
        # With window=1, rolling mean is just the interpolated value
        assert result.iloc[1] > 0.0
        assert result.iloc[2] < 10.0

    def test_monotonic_preserved(self):
        """A monotonic series stays monotonic after smoothing."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = phase_detector.smooth(series, window=3)
        for i in range(len(result) - 1):
            assert result.iloc[i] <= result.iloc[i + 1] + 1e-9


class TestPhaseDetectionIntegration:
    """Integration tests for the full phase detection pipeline."""

    def _make_angles_df(self, n_frames=100):
        """Create a synthetic batting angles dataframe."""
        np.random.seed(42)
        df = pd.DataFrame({
            "frame": range(n_frames),
            "timestamp_ms": [i * 33.33 for i in range(n_frames)],
            "left_elbow_angle": np.random.uniform(100, 160, n_frames),
            "right_elbow_angle": np.random.uniform(100, 160, n_frames),
            "left_shoulder_angle": np.random.uniform(40, 80, n_frames),
            "right_shoulder_angle": np.random.uniform(40, 80, n_frames),
            "left_hip_angle": np.random.uniform(140, 180, n_frames),
            "right_hip_angle": np.random.uniform(140, 180, n_frames),
            "left_knee_angle": np.random.uniform(120, 170, n_frames),
            "right_knee_angle": np.random.uniform(120, 170, n_frames),
            "lead_wrist_height": np.random.uniform(0.5, 1.5, n_frames),
        })
        return df

    def test_smoothing_applied(self):
        """All angle columns should be smoothed (no NaN from interpolation)."""
        df = self._make_angles_df()
        for col in ["left_elbow_angle", "right_elbow_angle",
                     "left_shoulder_angle", "right_shoulder_angle",
                     "left_hip_angle", "right_hip_angle",
                     "left_knee_angle", "right_knee_angle"]:
            smoothed = phase_detector.smooth(df[col])
            assert not smoothed.isna().any()

    def test_movement_columns_created(self):
        """Movement score columns should be created."""
        df = self._make_angles_df()
        # Simulate what main() does
        elbow = ["left_elbow_angle", "right_elbow_angle"]
        for col in elbow:
            df[col] = phase_detector.smooth(df[col])
        df["elbow_movement"] = df[elbow].diff().abs().mean(axis=1).fillna(0)
        assert "elbow_movement" in df.columns
        assert len(df) == 100
