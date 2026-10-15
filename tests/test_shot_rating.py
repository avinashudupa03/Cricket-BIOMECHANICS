"""Unit tests for shot_rating.py."""

import pytest

import shot_rating
from shot_rating import (
    ShotRating,
    FactorScore,
    rate_shot,
    rating_to_dict,
    build_extra,
    _clamp,
    _norm_linear,
    _norm_inverse,
    _bell,
    _mean,
    _effective_weights,
    _classify,
    _control_side,
    _symmetry,
    _stats_std,
    FACTOR_CODES,
    FACTOR_WEIGHTS,
    CLASSIFICATION,
)


class TestClamp:
    def test_within_range(self):
        assert _clamp(5.0) == 5.0

    def test_below_range(self):
        assert _clamp(-1.0) == 0.0

    def test_above_range(self):
        assert _clamp(11.0) == 10.0

    def test_custom_range(self):
        assert _clamp(15.0, low=10.0, high=20.0) == 15.0
        assert _clamp(5.0, low=10.0, high=20.0) == 10.0


class TestNormLinear:
    def test_at_good(self):
        assert _norm_linear(10.0, good=10.0, poor=0.0) == 10.0

    def test_at_poor(self):
        assert _norm_linear(0.0, good=10.0, poor=0.0) == 0.0

    def test_midpoint(self):
        assert _norm_linear(5.0, good=10.0, poor=0.0) == 5.0

    def test_beyond_good(self):
        assert _norm_linear(15.0, good=10.0, poor=0.0) == 10.0

    def test_none_input(self):
        assert _norm_linear(None, good=10.0, poor=0.0) is None

    def test_equal_good_poor(self):
        assert _norm_linear(5.0, good=5.0, poor=5.0) == 10.0
        assert _norm_linear(3.0, good=5.0, poor=5.0) == 0.0


class TestNormInverse:
    """_norm_inverse: small values score higher.

    With good=10.0 (large, scores 0) and poor=0.0 (small, scores 10):
    """

    def test_at_good(self):
        """value=good (large) scores 0."""
        assert _norm_inverse(10.0, good=10.0, poor=0.0) == 0.0

    def test_at_poor(self):
        """value=poor (small) scores 10."""
        assert _norm_inverse(0.0, good=10.0, poor=0.0) == 10.0

    def test_midpoint(self):
        assert _norm_inverse(5.0, good=10.0, poor=0.0) == 5.0

    def test_none_input(self):
        assert _norm_inverse(None, good=10.0, poor=0.0) is None


class TestBell:
    def test_at_centre(self):
        assert _bell(5.0, centre=5.0, half_span=2.0) == 10.0

    def test_at_edge(self):
        assert _bell(7.0, centre=5.0, half_span=2.0) == 0.0
        assert _bell(3.0, centre=5.0, half_span=2.0) == 0.0

    def test_beyond_edge(self):
        assert _bell(10.0, centre=5.0, half_span=2.0) == 0.0

    def test_halfway(self):
        assert _bell(6.0, centre=5.0, half_span=2.0) == 5.0

    def test_none_input(self):
        assert _bell(None, centre=5.0, half_span=2.0) is None


class TestMean:
    def test_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_with_nones(self):
        assert _mean([1.0, None, 3.0]) == 2.0

    def test_all_nones(self):
        assert _mean([None, None]) is None

    def test_empty(self):
        assert _mean([]) is None


class TestEffectiveWeights:
    def test_all_available(self):
        weights = _effective_weights(FACTOR_CODES)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_some_missing(self):
        available = ["swing", "timing", "contact"]
        weights = _effective_weights(available)
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert len(weights) == 3

    def test_none_available(self):
        weights = _effective_weights([])
        assert weights == {}

    def test_weight_redistribution(self):
        """Missing factor weight is redistributed proportionally."""
        available = ["swing", "timing"]
        weights = _effective_weights(available)
        # Original weights: swing=0.09, timing=0.15, total=0.24
        # Redistributed: swing = 0.09/0.24 = 0.375, timing = 0.15/0.24 = 0.625
        assert abs(weights["swing"] - 0.375) < 1e-6
        assert abs(weights["timing"] - 0.625) < 1e-6


class TestClassify:
    def test_excellent(self):
        assert _classify(9.5) == "Excellent"

    def test_very_good(self):
        assert _classify(8.5) == "Very Good"

    def test_good(self):
        assert _classify(7.5) == "Good"

    def test_average(self):
        assert _classify(6.5) == "Average"

    def test_below_average(self):
        assert _classify(5.5) == "Below Average"

    def test_poor(self):
        assert _classify(3.0) == "Poor"

    def test_zero(self):
        assert _classify(0.0) == "Poor"


class TestControlSide:
    def test_right_handed(self):
        """Left arm more extended => right-handed batsman."""
        f = {"left_elbow_angle_at_impact": 160, "right_elbow_angle_at_impact": 120}
        assert _control_side(f) == "left"

    def test_left_handed(self):
        """Right arm more extended => left-handed batsman."""
        f = {"left_elbow_angle_at_impact": 120, "right_elbow_angle_at_impact": 160}
        assert _control_side(f) == "right"

    def test_missing_data(self):
        assert _control_side({}) == "right"

    def test_none_values(self):
        f = {"left_elbow_angle_at_impact": None, "right_elbow_angle_at_impact": None}
        assert _control_side(f) == "right"


class TestSymmetry:
    def test_perfect_symmetry(self):
        assert _symmetry(10.0, 10.0, good=20.0) == 10.0

    def test_max_asymmetry(self):
        assert _symmetry(0.0, 20.0, good=20.0) == 0.0

    def test_partial(self):
        result = _symmetry(10.0, 15.0, good=20.0)
        assert result == 7.5

    def test_none_input(self):
        assert _symmetry(None, 10.0) is None
        assert _symmetry(10.0, None) is None


class TestStatsStd:
    def test_basic(self):
        result = _stats_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 2.0) < 1e-6

    def test_empty(self):
        assert _stats_std([]) == 0.0

    def test_single_value(self):
        assert _stats_std([5.0]) == 0.0

    def test_identical_values(self):
        assert _stats_std([3.0, 3.0, 3.0]) == 0.0


class TestRateShot:
    """Integration tests for the full rate_shot function."""

    def test_empty_features(self):
        """Rating with no data should return 0 with 0 confidence."""
        result = rate_shot({})
        assert result.rating == 0.0
        assert result.confidence == 0.0
        assert result.classification == "Poor"

    def test_none_features(self):
        """None features should be treated as empty."""
        result = rate_shot(None)
        assert result.rating == 0.0

    def test_full_features(self):
        """Rating with comprehensive features should produce a valid score."""
        features = {
            "downswing_duration_ms": 200,
            "maximum_movement_score": 10.0,
            "average_movement_score": 5.0,
            "impact_frames": 2,
            "followthrough_duration_ms": 800,
            "left_elbow_angle_min": 30, "left_elbow_angle_max": 170,
            "right_elbow_angle_min": 30, "right_elbow_angle_max": 170,
            "left_knee_angle_min": 80, "left_knee_angle_max": 170,
            "right_knee_angle_min": 80, "right_knee_angle_max": 170,
            "left_hip_angle_min": 100, "left_hip_angle_max": 180,
            "right_hip_angle_min": 100, "right_hip_angle_max": 180,
            "left_shoulder_angle_min": 20, "left_shoulder_angle_max": 100,
            "right_shoulder_angle_min": 20, "right_shoulder_angle_max": 100,
            "left_elbow_angle_at_impact": 160,
            "right_elbow_angle_at_impact": 140,
            "left_shoulder_angle_at_impact": 65,
            "right_shoulder_angle_at_impact": 60,
            "left_knee_angle_at_impact": 160,
            "right_knee_angle_at_impact": 155,
            "left_hip_angle_at_impact": 170,
            "right_hip_angle_at_impact": 165,
            "left_elbow_angle_std": 15,
            "right_elbow_angle_std": 18,
            "right_knee_angle_std": 20,
            "bat_speed_torso_per_s": 2.5,
            "torso_rotation_range_swing": 40,
            "lead_wrist_height_range_swing": 1.2,
            "ankle_separation_at_impact": 0.5,
            "stride_length_torso": 1.0,
        }
        result = rate_shot(features, shot_type="drive")
        assert 0 <= result.rating <= 10
        assert 0 <= result.confidence <= 1
        assert result.shot_type == "drive"
        assert len(result.factors) == 12

    def test_rating_deterministic(self):
        """Same input should always produce the same rating."""
        features = {
            "downswing_duration_ms": 250,
            "maximum_movement_score": 8.0,
            "average_movement_score": 4.0,
            "impact_frames": 3,
            "followthrough_duration_ms": 600,
            "left_elbow_angle_min": 40, "left_elbow_angle_max": 160,
            "right_elbow_angle_min": 40, "right_elbow_angle_max": 160,
            "left_knee_angle_min": 90, "left_knee_angle_max": 170,
            "right_knee_angle_min": 90, "right_knee_angle_max": 170,
            "left_hip_angle_min": 110, "left_hip_angle_max": 180,
            "right_hip_angle_min": 110, "right_hip_angle_max": 180,
            "left_shoulder_angle_min": 30, "left_shoulder_angle_max": 90,
            "right_shoulder_angle_min": 30, "right_shoulder_angle_max": 90,
            "left_elbow_angle_at_impact": 155,
            "right_elbow_angle_at_impact": 145,
            "left_shoulder_angle_at_impact": 60,
            "right_shoulder_angle_at_impact": 55,
            "left_knee_angle_at_impact": 160,
            "right_knee_angle_at_impact": 158,
            "left_hip_angle_at_impact": 170,
            "right_hip_angle_at_impact": 168,
            "left_elbow_angle_std": 12,
            "right_elbow_angle_std": 14,
            "right_knee_angle_std": 18,
        }
        result1 = rate_shot(features, shot_type="cut")
        result2 = rate_shot(features, shot_type="cut")
        assert result1.rating == result2.rating
        assert result1.confidence == result2.confidence

    def test_with_extra_data(self):
        """Extra per-frame data should activate more factors."""
        features = {
            "downswing_duration_ms": 200,
            "maximum_movement_score": 9.0,
            "average_movement_score": 4.5,
            "impact_frames": 2,
            "followthrough_duration_ms": 700,
            "left_elbow_angle_min": 30, "left_elbow_angle_max": 170,
            "right_elbow_angle_min": 30, "right_elbow_angle_max": 170,
            "left_knee_angle_min": 80, "left_knee_angle_max": 170,
            "right_knee_angle_min": 80, "right_knee_angle_max": 170,
            "left_hip_angle_min": 100, "left_hip_angle_max": 180,
            "right_hip_angle_min": 100, "right_hip_angle_max": 180,
            "left_shoulder_angle_min": 20, "left_shoulder_angle_max": 100,
            "right_shoulder_angle_min": 20, "right_shoulder_angle_max": 100,
            "left_elbow_angle_at_impact": 160,
            "right_elbow_angle_at_impact": 140,
            "left_shoulder_angle_at_impact": 65,
            "right_shoulder_angle_at_impact": 60,
            "left_knee_angle_at_impact": 160,
            "right_knee_angle_at_impact": 155,
            "left_hip_angle_at_impact": 170,
            "right_hip_angle_at_impact": 165,
            "left_elbow_angle_std": 15,
            "right_elbow_angle_std": 18,
            "right_knee_angle_std": 20,
        }
        extra = {
            "head_tracks": [{"x": 0.5, "y": 0.3} for _ in range(10)],
            "backlift": {"ratios": [0.8, 0.9, 1.0, 0.85, 0.95]},
            "stride": {"front_ratio": 1.2, "back_ratio": 0.1},
        }
        result = rate_shot(features, extra=extra, shot_type="drive")
        # With extra data, more factors should be available
        available_count = sum(1 for f in result.factors if f.available)
        assert available_count > 5

    def test_missing_factors_redistributed(self):
        """When factors are missing, weights are redistributed."""
        features = {
            "downswing_duration_ms": 200,
            "maximum_movement_score": 8.0,
            "average_movement_score": 4.0,
            "impact_frames": 2,
            "followthrough_duration_ms": 600,
            "left_elbow_angle_min": 40, "left_elbow_angle_max": 160,
            "right_elbow_angle_min": 40, "right_elbow_angle_max": 160,
            "left_knee_angle_min": 90, "left_knee_angle_max": 170,
            "right_knee_angle_min": 90, "right_knee_angle_max": 170,
            "left_hip_angle_min": 110, "left_hip_angle_max": 180,
            "right_hip_angle_min": 110, "right_hip_angle_max": 180,
            "left_shoulder_angle_min": 30, "left_shoulder_angle_max": 90,
            "right_shoulder_angle_min": 30, "right_shoulder_angle_max": 90,
            "left_elbow_angle_at_impact": 155,
            "right_elbow_angle_at_impact": 145,
            "left_shoulder_angle_at_impact": 60,
            "right_shoulder_angle_at_impact": 55,
            "left_knee_angle_at_impact": 160,
            "right_knee_angle_at_impact": 158,
            "left_hip_angle_at_impact": 170,
            "right_hip_angle_at_impact": 168,
            "left_elbow_angle_std": 12,
            "right_elbow_angle_std": 14,
            "right_knee_angle_std": 18,
        }
        result = rate_shot(features, shot_type="drive")
        # covered_weight should be less than 1.0 since some factors are missing
        assert result.covered_weight < 1.0
        assert result.covered_weight > 0.0


class TestRatingToDict:
    def test_serialization(self):
        features = {
            "downswing_duration_ms": 200,
            "maximum_movement_score": 8.0,
            "average_movement_score": 4.0,
            "impact_frames": 2,
            "followthrough_duration_ms": 600,
            "left_elbow_angle_min": 40, "left_elbow_angle_max": 160,
            "right_elbow_angle_min": 40, "right_elbow_angle_max": 160,
            "left_knee_angle_min": 90, "left_knee_angle_max": 170,
            "right_knee_angle_min": 90, "right_knee_angle_max": 170,
            "left_hip_angle_min": 110, "left_hip_angle_max": 180,
            "right_hip_angle_min": 110, "right_hip_angle_max": 180,
            "left_shoulder_angle_min": 30, "left_shoulder_angle_max": 90,
            "right_shoulder_angle_min": 30, "right_shoulder_angle_max": 90,
            "left_elbow_angle_at_impact": 155,
            "right_elbow_angle_at_impact": 145,
            "left_shoulder_angle_at_impact": 60,
            "right_shoulder_angle_at_impact": 55,
            "left_knee_angle_at_impact": 160,
            "right_knee_angle_at_impact": 158,
            "left_hip_angle_at_impact": 170,
            "right_hip_angle_at_impact": 168,
            "left_elbow_angle_std": 12,
            "right_elbow_angle_std": 14,
            "right_knee_angle_std": 18,
        }
        result = rate_shot(features, shot_type="drive")
        d = rating_to_dict(result)

        assert d["shot_type"] == "drive"
        assert "rating" in d
        assert "classification" in d
        assert "confidence" in d
        assert "factors" in d
        assert len(d["factors"]) == 12
        assert "strengths" in d
        assert "weaknesses" in d
        assert "explanation" in d
        assert "unavailable_metrics" in d


class TestFactorWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(FACTOR_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_factors_have_weights(self):
        for code in FACTOR_CODES:
            assert code in FACTOR_WEIGHTS
            assert FACTOR_WEIGHTS[code] > 0

    def test_classification_bands_cover_full_range(self):
        """Classification bands should cover 0-10."""
        # Bands are ordered highest to lowest: (9-10.01), (8-8.99), ..., (0-4.99)
        lows = [b[0] for b in CLASSIFICATION]
        highs = [b[1] for b in CLASSIFICATION]
        assert lows[-1] == 0.0  # lowest band starts at 0
        assert highs[0] > 10.0  # highest band ends above 10
        # Bands are ordered from highest to lowest
        for i in range(len(CLASSIFICATION) - 1):
            assert lows[i] > lows[i + 1]
