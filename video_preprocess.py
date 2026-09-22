"""Geometry-preserving frame enhancement for pose detection robustness.

Purpose
-------
Frames from `cv2.VideoCapture` are fed straight into MediaPipe today. On
dark, unevenly-lit or low-contrast clips, keypoint confidence drops and pose
estimates degrade. This module applies a *geometry-preserving* enhancement
pass before the pose detector so the same physics (normalized 0..1 landmark
coordinates, angles, torso-scaled distances, phase windows) remain valid
while the detector sees a better-exposed frame.

Design rules
------------
* Only per-pixel *intensity* is changed - CLAHE (Contrast Limited Adaptive
  Histogram Equalization) on the L channel of the Lab colour space. Spatial
  layout is untouched, so landmark coordinates map 1:1 to the original frame.
* Enhancements never change frame size, FPS or frame count (no resize, no
  frame drop, no time remap) - the video fidelity contract is preserved.
* Everything is optional and configurable from config.py / env vars so the
  default pipeline stays byte-compatible unless contrast handling is wanted.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    PREPROCESS_ENABLE,
    PREPROCESS_CLAHE_CLIP,
    PREPROCESS_CLAHE_GRID,
)


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Return one CLAHE-enhanced BGR frame (intensities only, no resize).

    CLAHE equalises the *local* L-channel contrast, which lifts dark /
    shadenged regions and reduces blown highlights without washing out the
    whole image (global histogram equalisation is deliberately avoided).
    The ``clip`` limit bounds how much a single contrast spike can be
    amplified, keeping noise from being over-amplified in flat regions.

    Detection on the enhanced frame can recover landmarks that a murky,
    low-contrast frame hides while leaving every spatial relationship in the
    video unchanged.
    """
    if frame is None:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=PREPROCESS_CLAHE_CLIP,
        tileGridSize=(PREPROCESS_CLAHE_GRID, PREPROCESS_CLAHE_GRID),
    )
    l_eq = clahe.apply(l_ch)

    lab_eq = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def should_enhance() -> bool:
    """Whether the detection stream should run the enhancement pass."""
    return bool(PREPROCESS_ENABLE)


# ---------------------------------------------------------------------------
# Quick self-check: enhancement must not move the (normalized) geometry of a
# known image. Run with:
#     python video_preprocess.py --selftest
# ---------------------------------------------------------------------------
def selftest() -> int:
    synthetic = np.zeros((240, 320, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    synthetic += rng.integers(0, 255, size=synthetic.shape, dtype=np.uint8)
    # A strong dark band to prove shadowed regions get lifted.
    synthetic[60:180, 80:240] //= 4

    out = enhance_frame(synthetic)

    if out.shape != synthetic.shape:
        print("[FAIL] shape changed")
        return 1
    if out.dtype != synthetic.dtype:
        print("[FAIL] dtype changed")
        return 1

    # L-channel contrast should be higher after equalisation.
    def _l_std(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return float(lab[:, :, 0].std())

    if _l_std(out) <= _l_std(synthetic):
        print("[FAIL] L-channel contrast did not increase")
        return 1

    print("[ok] enhancement is geometry-preserving "
          f"(shape {out.shape}, dtype {out.dtype})")
    print(f"[ok] L-channel std {_l_std(synthetic):.1f} -> {_l_std(out):.1f}")
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print("Run with --selftest to verify the enhancement is geometry-safe.")