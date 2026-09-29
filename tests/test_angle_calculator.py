"""Unit tests for angle_calculator.py."""

import math
import numpy as np
import pandas as pd
import pytest

import angle_calculator


class TestCalculateAngle:
    """Tests for the calculate_angle function."""

    def test_right_angle(self):
        """A 90-degree angle at the origin."""
        a = [1, 0, 0]
        b = [0, 0, 0]
        c = [0, 1, 0]
        result = angle_calculator.calculate_angle(a, b, c)
        assert abs(result - 90.0) < 1e-6

    def test_straight_angle(self):
        """A 180-degree angle (straight line)."""
        a = [1, 0, 0]
        b = [0, 0, 0]
        c = [-1, 0, 0]
        result = angle_calculator.calculate_angle(a, b, c)
        assert abs(result - 180.0) < 1e-6

    def test_zero_angle(self):
        """A 0-degree angle (same direction)."""
        a = [1, 0, 0]
        b = [0, 0, 0]
        c = [2, 0, 0]
        result = angle_calculator.calculate_angle(a, b, c)
        assert abs(result - 0.0) < 1e-6

    def test_45_degree_angle(self):
        """A 45-degree angle."""
        a = [1, 0, 0]
        b = [0, 0, 0]
        c = [1, 1, 0]
        result = angle_calculator.calculate_angle(a, b, c)
        assert abs(result - 45.0) < 1e-6

    def test_3d_angle(self):
        """Angle calculation in 3D space."""
        a = [1, 0, 0]
        b = [0, 0, 0]
        c = [0, 0, 1]
        result = angle_calculator.calculate_angle(a, b, c)
        assert abs(result - 90.0) < 1e-6

    def test_empty_arrays(self):
        """Empty arrays return NaN."""
        result = angle_calculator.calculate_angle([], [], [])
        assert math.isnan(result)

    def test_zero_norm(self):
        """Zero-length vectors return NaN."""
        a = [0, 0, 0]
        b = [0, 0, 0]
        c = [1, 0, 0]
        result = angle_calculator.calculate_angle(a, b, c)
        assert math.isnan(result)

    def test_nan_in_input(self):
        """NaN in input returns NaN."""
        a = [np.nan, 0, 0]
        b = [0, 0, 0]
        c = [0, 1, 0]
        result = angle_calculator.calculate_angle(a, b, c)
        assert math.isnan(result)

    def test_arccos_clipping(self):
        """Values slightly beyond [-1, 1] due to float precision are clipped."""
        # These would produce cos slightly > 1.0 without clipping
        a = [1, 0, 0]
        b = [0, 0, 0]
        c = [1, 1e-10, 0]
        result = angle_calculator.calculate_angle(a, b, c)
        assert 0 <= result <= 180


class TestGetPoint:
    """Tests for the get_point function."""

    def test_valid_point(self):
        row = {"nose_x": 0.5, "nose_y": 0.3, "nose_z": 0.1}
        result = angle_calculator.get_point(row, "nose")
        assert result == [0.5, 0.3, 0.1]

    def test_missing_column(self):
        """Missing column raises KeyError (not NaN)."""
        row = {"other_x": 0.5, "other_y": 0.3, "other_z": 0.1}
        with pytest.raises(KeyError):
            angle_calculator.get_point(row, "nose")

    def test_non_numeric(self):
        row = {"nose_x": "abc", "nose_y": "def", "nose_z": "ghi"}
        result = angle_calculator.get_point(row, "nose")
        assert all(math.isnan(v) for v in result)

    def test_none_values(self):
        row = {"nose_x": None, "nose_y": None, "nose_z": None}
        result = angle_calculator.get_point(row, "nose")
        assert all(math.isnan(v) for v in result)


class TestPointArray:
    """Tests for the _point_array function."""

    def test_basic_extraction(self):
        df = pd.DataFrame({
            "left_shoulder_x": [0.1, 0.2],
            "left_shoulder_y": [0.3, 0.4],
            "left_shoulder_z": [0.5, 0.6],
        })
        result = angle_calculator._point_array(df, "left_shoulder")
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.3, 0.5])
        np.testing.assert_array_almost_equal(result[1], [0.2, 0.4, 0.6])

    def test_missing_columns(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = angle_calculator._point_array(df, "left_shoulder")
        assert result.shape == (2, 3)
        assert np.all(np.isnan(result))

    def test_non_numeric_values(self):
        df = pd.DataFrame({
            "left_shoulder_x": ["a", "b"],
            "left_shoulder_y": [0.3, 0.4],
            "left_shoulder_z": [0.5, 0.6],
        })
        result = angle_calculator._point_array(df, "left_shoulder")
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        assert result[0, 1] == 0.3


class TestVisibilityMask:
    """Tests for the visibility_mask function."""

    def test_no_visibility_column(self):
        """Returns None when no visibility column exists."""
        df = pd.DataFrame({"left_shoulder_x": [0.1]})
        result = angle_calculator.visibility_mask(df, "left_shoulder", 0.3)
        assert result is None

    def test_all_above_threshold(self):
        df = pd.DataFrame({"left_shoulder_visibility": [0.8, 0.9, 0.7]})
        result = angle_calculator.visibility_mask(df, "left_shoulder", 0.3)
        np.testing.assert_array_equal(result, [True, True, True])

    def test_mixed_visibility(self):
        df = pd.DataFrame({"left_shoulder_visibility": [0.8, 0.2, 0.5]})
        result = angle_calculator.visibility_mask(df, "left_shoulder", 0.3)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_nan_visibility(self):
        df = pd.DataFrame({"left_shoulder_visibility": [0.8, np.nan, 0.5]})
        result = angle_calculator.visibility_mask(df, "left_shoulder", 0.3)
        np.testing.assert_array_equal(result, [True, False, True])


class TestVectorizedAngle:
    """Tests for the _vectorized_angle function."""

    def test_basic_angles(self):
        a = np.array([[1, 0, 0], [1, 0, 0]], dtype=float)
        b = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
        c = np.array([[0, 1, 0], [-1, 0, 0]], dtype=float)
        result = angle_calculator._vectorized_angle(a, b, c)
        assert abs(result[0] - 90.0) < 1e-6
        assert abs(result[1] - 180.0) < 1e-6

    def test_with_nan(self):
        a = np.array([[1, 0, 0], [np.nan, 0, 0]], dtype=float)
        b = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
        c = np.array([[0, 1, 0], [0, 1, 0]], dtype=float)
        result = angle_calculator._vectorized_angle(a, b, c)
        assert abs(result[0] - 90.0) < 1e-6
        assert math.isnan(result[1])

    def test_zero_norm(self):
        a = np.array([[0, 0, 0]], dtype=float)
        b = np.array([[0, 0, 0]], dtype=float)
        c = np.array([[1, 0, 0]], dtype=float)
        result = angle_calculator._vectorized_angle(a, b, c)
        assert math.isnan(result[0])
