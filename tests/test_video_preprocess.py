"""Unit tests for video_preprocess.py."""

import numpy as np
import pytest

import video_preprocess


class TestEnhanceFrame:
    """Tests for the enhance_frame function."""

    def test_none_input(self):
        """None input returns None."""
        result = video_preprocess.enhance_frame(None)
        assert result is None

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = video_preprocess.enhance_frame(frame)
        assert result.shape == frame.shape

    def test_dtype_preserved(self):
        """Output dtype matches input dtype."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = video_preprocess.enhance_frame(frame)
        assert result.dtype == frame.dtype

    def test_contrast_increased(self):
        """L-channel contrast should increase after CLAHE."""
        # Create synthetic image with dark region
        synthetic = np.zeros((240, 320, 3), dtype=np.uint8)
        rng = np.random.default_rng(0)
        synthetic += rng.integers(0, 255, size=synthetic.shape, dtype=np.uint8)
        # Add a strong dark band
        synthetic[60:180, 80:240] //= 4

        out = video_preprocess.enhance_frame(synthetic)

        def _l_std(img):
            import cv2
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            return float(lab[:, :, 0].std())

        assert _l_std(out) > _l_std(synthetic)

    def test_geometry_preserved(self):
        """Spatial layout must not change - only intensities."""
        # Create a frame with a known pattern
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[25:75, 25:75] = [255, 0, 0]  # Red square in center

        out = video_preprocess.enhance_frame(frame)

        # The red square should still be in the same position
        # (just with different intensity values)
        center_pixel_before = frame[50, 50]
        center_pixel_after = out[50, 50]

        # Both should be non-zero (the square is still there)
        assert np.any(center_pixel_before > 0)
        assert np.any(center_pixel_after > 0)


class TestShouldEnhance:
    """Tests for the should_enhance function."""

    def test_returns_bool(self):
        """Should return a boolean."""
        result = video_preprocess.should_enhance()
        assert isinstance(result, bool)


class TestSelftest:
    """Test the built-in selftest."""

    def test_selftest_passes(self):
        """The internal selftest should pass."""
        # This should not raise
        result = video_preprocess.selftest()
        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])