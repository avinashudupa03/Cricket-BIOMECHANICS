"""Unit tests for pose_extractor.py core utilities."""

import math
import pytest

import pose_extractor
from pose_extractor import (
    get_center, get_bbox, bbox_size, get_shape, shape_distance,
    central_prior, _shoulder_width_ratio, _leg_spread_ratio, LM
)


class TestGetCenter:
    """Tests for get_center function."""

    def test_valid_pose(self):
        pose = [LM() for _ in range(25)]
        pose[11] = LM(x=0.4, y=0.4)
        pose[12] = LM(x=0.6, y=0.4)
        pose[23] = LM(x=0.45, y=0.6)
        pose[24] = LM(x=0.55, y=0.6)
        center = pose_extractor.get_center(pose)
        assert center is not None
        assert abs(center[0] - 0.5) < 0.01
        assert abs(center[1] - 0.5) < 0.01

    def test_empty_pose(self):
        pose = []
        assert pose_extractor.get_center(pose) is None

    def test_missing_joints(self):
        pose = [LM() for _ in range(11)]  # Only up to index 10
        assert pose_extractor.get_center(pose) is None


class TestGetBBox:
    """Tests for get_bbox function."""

    def test_valid_pose(self):
        pose = [
            LM(x=0.1, y=0.2),
            LM(x=0.9, y=0.8),
            LM(x=0.5, y=0.5),
        ]
        bbox = pose_extractor.get_bbox(pose)
        assert bbox == (0.1, 0.2, 0.9, 0.8)

    def test_empty_pose(self):
        assert pose_extractor.get_bbox([]) is None


class TestBBoxSize:
    """Tests for bbox_size function."""

    def test_basic(self):
        bbox = (0.1, 0.2, 0.9, 0.8)
        w, h = pose_extractor.bbox_size(bbox)
        assert abs(w - 0.8) < 1e-10
        assert abs(h - 0.6) < 1e-10

    def test_zero_size_guard(self):
        bbox = (0.5, 0.5, 0.5, 0.5)
        w, h = pose_extractor.bbox_size(bbox)
        assert abs(w - 1e-6) < 1e-10
        assert abs(h - 1e-6) < 1e-10


class TestGetShape:
    """Tests for get_shape function."""

    def test_valid(self):
        pose = [LM() for _ in range(29)]
        # Set up a simple pose
        for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            pose[i] = LM(x=0.5, y=0.5)
        center = (0.5, 0.5)
        bbox = (0.4, 0.4, 0.6, 0.6)
        shape = pose_extractor.get_shape(pose, center, bbox)
        assert shape is not None
        assert len(shape) == 12

    def test_none_center(self):
        pose = [LM() for _ in range(29)]
        assert pose_extractor.get_shape(pose, None, (0, 0, 1, 1)) is None

    def test_none_bbox(self):
        pose = [LM() for _ in range(29)]
        assert pose_extractor.get_shape(pose, (0.5, 0.5), None) is None


class TestShapeDistance:
    """Tests for shape_distance function."""

    def test_identical_shapes(self):
        shape1 = [(0.1, 0.2)] * 12
        shape2 = [(0.1, 0.2)] * 12
        dist = pose_extractor.shape_distance(shape1, shape2)
        assert dist == 0.0

    def test_different_shapes(self):
        shape1 = [(0.0, 0.0)] * 12
        shape2 = [(1.0, 1.0)] * 12
        dist = pose_extractor.shape_distance(shape1, shape2)
        assert abs(dist - math.sqrt(2.0)) < 1e-10

    def test_none_shape(self):
        assert pose_extractor.shape_distance(None, [(0, 0)] * 12) == 999.0
        assert pose_extractor.shape_distance([(0, 0)] * 12, None) == 999.0

    def test_mismatched_length(self):
        shape1 = [(0, 0)] * 12
        shape2 = [(0, 0)] * 10
        assert pose_extractor.shape_distance(shape1, shape2) == 999.0


class TestCentralPrior:
    """Tests for central_prior function."""

    def test_center_position(self):
        """Center of expected batting zone should score high."""
        score = pose_extractor.central_prior((0.50, 0.35))
        assert score > 0.9

    def test_far_position(self):
        """Far from batting zone should score low."""
        score = pose_extractor.central_prior((0.0, 0.0))
        assert score < 0.01

    def test_none_center(self):
        assert pose_extractor.central_prior(None) == 0.0


class TestShoulderWidthRatio:
    """Tests for _shoulder_width_ratio function."""

    def test_valid(self):
        pose = [LM() for _ in range(25)]
        pose[11] = LM(x=0.4, y=0.4)
        pose[12] = LM(x=0.6, y=0.4)
        pose[23] = LM(x=0.45, y=0.6)
        pose[24] = LM(x=0.55, y=0.6)
        ratio = pose_extractor._shoulder_width_ratio(pose)
        assert ratio > 0

    def test_short_pose(self):
        pose = [LM() for _ in range(20)]
        assert pose_extractor._shoulder_width_ratio(pose) == 0.0

    def test_nan_values(self):
        pose = [LM() for _ in range(25)]
        pose[11] = LM(x=float('nan'), y=0.4)
        assert pose_extractor._shoulder_width_ratio(pose) == 0.0


class TestLegSpreadRatio:
    """Tests for _leg_spread_ratio function."""

    def test_valid(self):
        pose = [LM() for _ in range(29)]
        pose[23] = LM(x=0.45, y=0.6)
        pose[24] = LM(x=0.55, y=0.6)
        pose[27] = LM(x=0.4, y=0.8)
        pose[28] = LM(x=0.6, y=0.8)
        ratio = pose_extractor._leg_spread_ratio(pose)
        assert ratio > 0

    def test_short_pose(self):
        pose = [LM() for _ in range(25)]
        assert pose_extractor._leg_spread_ratio(pose) == 0.0


class TestIsBatsmanStance:
    """Tests for _is_batsman_stance function."""

    def test_valid_stance(self):
        pose = [LM() for _ in range(29)]
        # Set up a batting-like stance
        pose[11] = LM(x=0.4, y=0.4, visibility=0.9)
        pose[12] = LM(x=0.6, y=0.4, visibility=0.9)
        pose[23] = LM(x=0.45, y=0.6, visibility=0.9)
        pose[24] = LM(x=0.55, y=0.6, visibility=0.9)
        pose[27] = LM(x=0.35, y=0.8, visibility=0.9)
        pose[28] = LM(x=0.65, y=0.8, visibility=0.9)
        pose[13] = LM(x=0.35, y=0.3, visibility=0.9)
        pose[14] = LM(x=0.3, y=0.2, visibility=0.9)
        pose[15] = LM(x=0.25, y=0.15, visibility=0.9)
        pose[16] = LM(x=0.65, y=0.3, visibility=0.9)
        pose[17] = LM(x=0.7, y=0.2, visibility=0.9)
        pose[18] = LM(x=0.75, y=0.15, visibility=0.9)

        score = pose_extractor._is_batsman_stance(pose)
        assert 0 <= score <= 1

    def test_short_pose(self):
        pose = [LM() for _ in range(20)]
        assert pose_extractor._is_batsman_stance(pose) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])