"""Shared pytest fixtures for the cricket biomechanics test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so tests can import modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_landmarks_df():
    """Create a small synthetic landmarks dataframe for testing."""
    np.random.seed(42)
    n = 50
    data = {"frame": range(n), "timestamp_ms": [i * 33.33 for i in range(n)]}
    for name in ("nose", "left_shoulder", "right_shoulder", "left_elbow",
                 "right_elbow", "left_wrist", "right_wrist", "left_hip",
                 "right_hip", "left_knee", "right_knee", "left_ankle",
                 "right_ankle"):
        data[f"{name}_x"] = np.random.uniform(0.3, 0.7, n)
        data[f"{name}_y"] = np.random.uniform(0.2, 0.8, n)
        data[f"{name}_z"] = np.random.uniform(-0.1, 0.1, n)
        data[f"{name}_visibility"] = np.random.uniform(0.5, 1.0, n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Create a minimal biomechanics features dict for rating tests."""
    return {
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
