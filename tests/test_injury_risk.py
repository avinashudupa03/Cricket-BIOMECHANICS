"""Unit tests for injury_risk.py."""

import math
import numpy as np
import pandas as pd
import pytest

import injury_risk
from injury_risk import InjuryAnalyzer, _num, _sustained_mask, _severity_for, SEVERITY_PENALTY


class TestNum:
    """Tests for the _num helper function."""

    def test_valid_float(self):
        assert _num("3.14") == 3.14
        assert _num(5) == 5.0
        assert _num(5.0) == 5.0

    def test_none_input(self):
        assert _num(None) is None
        assert _num(None, default=1.0) == 1.0

    def test_nan_input(self):
        assert _num(float('nan')) is None
        assert _num(float('inf')) is None

    def test_invalid_string(self):
        assert _num("abc") is None


class TestSustainedMask:
    """Tests for the _sustained_mask function."""

    def test_short_run_removed(self):
        """Runs shorter than min_run should be removed."""
        mask = np.array([True, True, False, True, True, True, False])
        result = _sustained_mask(mask, min_run=3)
        assert not result[0]  # 2-frame run removed
        assert not result[1]
        assert result[3] and result[4] and result[5]  # 3-frame run kept

    def test_exact_min_run_kept(self):
        """Run exactly equal to min_run should be kept."""
        mask = np.array([True, True, True, False])
        result = _sustained_mask(mask, min_run=3)
        assert all(result[:3])

    def test_all_false(self):
        mask = np.array([False, False, False])
        result = _sustained_mask(mask, min_run=3)
        assert not any(result)


class TestSeverityFor:
    """Tests for the _severity_for function."""

    def test_elbow_hyperextension_high(self):
        sev, cue = _severity_for("elbow_hyperextension", 195)
        assert sev == "high"
        assert "Stop locking" in cue

    def test_elbow_hyperextension_moderate(self):
        sev, cue = _severity_for("elbow_hyperextension", 187)
        assert sev == "moderate"
        assert "Watch the elbow" in cue

    def test_elbow_hyperextension_mild(self):
        sev, cue = _severity_for("elbow_hyperextension", 180)
        assert sev == "mild"
        assert "Occasional" in cue

    def test_knee_valgus_high(self):
        sev, cue = _severity_for("knee_valgus", 0.40)
        assert sev == "high"

    def test_knee_valgus_moderate(self):
        sev, cue = _severity_for("knee_valgus", 0.30)
        assert sev == "moderate"

    def test_knee_valgus_mild(self):
        sev, cue = _severity_for("knee_valgus", 0.20)
        assert sev == "mild"

    def test_deep_knee_flexion_high(self):
        sev, cue = _severity_for("deep_knee_flexion", 80)
        assert sev == "high"

    def test_deep_knee_flexion_moderate(self):
        sev, cue = _severity_for("deep_knee_flexion", 90)
        assert sev == "moderate"

    def test_deep_knee_flexion_mild(self):
        sev, cue = _severity_for("deep_knee_flexion", 100)
        assert sev == "mild"

    def test_trunk_lean_high(self):
        sev, cue = _severity_for("trunk_lean", 25)
        assert sev == "high"

    def test_trunk_lean_moderate(self):
        sev, cue = _severity_for("trunk_lean", 16)
        assert sev == "moderate"

    def test_trunk_lean_mild(self):
        sev, cue = _severity_for("trunk_lean", 10)
        assert sev == "mild"

    def test_trunk_flexion_high(self):
        sev, cue = _severity_for("trunk_flexion", 80)
        assert sev == "high"

    def test_trunk_flexion_moderate(self):
        sev, cue = _severity_for("trunk_flexion", 90)
        assert sev == "moderate"

    def test_trunk_flexion_mild(self):
        sev, cue = _severity_for("trunk_flexion", 100)
        assert sev == "mild"

    def test_unknown_metric(self):
        sev, cue = _severity_for("unknown_metric", 100)
        assert sev is None
        assert cue is None


class TestInjuryAnalyzer:
    """Integration tests for InjuryAnalyzer."""

    def _make_test_data(self, folder):
        """Create synthetic test data files."""
        import shutil
        from pathlib import Path

        # batting_phases.csv
        n_frames = 100
        phases = []
        for i in range(n_frames):
            if i < 20:
                phase = "Stance"
            elif i < 40:
                phase = "Backlift"
            elif i < 60:
                phase = "Downswing"
            elif i <= 64:
                phase = "Impact"
            else:
                phase = "Follow-through"
            phases.append(phase)

        df_phases = pd.DataFrame({
            "frame": range(n_frames),
            "timestamp_ms": [i * 33.33 for i in range(n_frames)],
            "phase": phases,
            "left_elbow_angle": np.random.uniform(100, 160, n_frames),
            "right_elbow_angle": np.random.uniform(100, 160, n_frames),
            "left_knee_angle": np.random.uniform(120, 170, n_frames),
            "right_knee_angle": np.random.uniform(120, 170, n_frames),
            "left_shoulder_angle": np.random.uniform(40, 80, n_frames),
            "right_shoulder_angle": np.random.uniform(40, 80, n_frames),
            "left_hip_angle": np.random.uniform(140, 180, n_frames),
            "right_hip_angle": np.random.uniform(140, 180, n_frames),
            "left_elbow_angle_at_impact": [160] * n_frames,
            "right_elbow_angle_at_impact": [140] * n_frames,
            "left_knee_angle_at_impact": [160] * n_frames,
            "right_knee_angle_at_impact": [155] * n_frames,
            "left_shoulder_angle_at_impact": [65] * n_frames,
            "right_shoulder_angle_at_impact": [60] * n_frames,
            "left_hip_angle_at_impact": [170] * n_frames,
            "right_hip_angle_at_impact": [165] * n_frames,
            "movement_smooth": np.random.uniform(0, 10, n_frames),
            "lead_wrist_height": np.random.uniform(0.5, 1.5, n_frames),
            "torso_rotation": np.random.uniform(0, 45, n_frames),
            "head_vertical": np.random.uniform(-0.5, 0.5, n_frames),
            "head_lateral": np.random.uniform(-0.3, 0.3, n_frames),
            "lead_wrist_to_hip": np.random.uniform(0.5, 1.5, n_frames),
            "left_ankle_to_hip_lateral": np.random.uniform(-0.5, 0.5, n_frames),
            "right_ankle_to_hip_lateral": np.random.uniform(-0.5, 0.5, n_frames),
        })
        df_phases.to_csv(folder / "batting_phases.csv", index=False)

        # batting_landmarks.csv (minimal)
        df_landmarks = pd.DataFrame({
            "frame": range(n_frames),
            "timestamp_ms": [i * 33.33 for i in range(n_frames)],
            "nose_x": [0.5] * n_frames,
            "nose_y": [0.3] * n_frames,
            "nose_z": [0.1] * n_frames,
            "left_shoulder_x": [0.4] * n_frames,
            "left_shoulder_y": [0.4] * n_frames,
            "left_shoulder_z": [0.1] * n_frames,
            "right_shoulder_x": [0.6] * n_frames,
            "right_shoulder_y": [0.4] * n_frames,
            "right_shoulder_z": [0.1] * n_frames,
            "right_hip_y": [0.6] * n_frames,
            "left_hip_y": [0.6] * n_frames,
            "left_wrist_y": [0.2] * n_frames,
            "right_wrist_y": [0.2] * n_frames,
            "left_ankle_x": [0.3] * n_frames,
            "right_ankle_x": [0.7] * n_frames,
            "left_ankle_z": [0.2] * n_frames,
            "right_ankle_z": [0.2] * n_frames,
        })
        df_landmarks.to_csv(folder / "batting_landmarks.csv", index=False)

    def test_analyzer_loads(self, tmp_path):
        """Test that analyzer can load test data."""
        self._make_test_data(tmp_path)
        analyzer = InjuryAnalyzer("test_video")
        analyzer.folder = tmp_path
        assert analyzer.load()

    def test_analyzer_missing_phases(self, tmp_path):
        """Test graceful handling of missing phases file."""
        analyzer = InjuryAnalyzer("test_video")
        analyzer.folder = tmp_path
        assert not analyzer.load()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])