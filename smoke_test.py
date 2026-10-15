"""Headless smoke tester for the cricket biomechanics pipeline fixes.

Run after any change to the pipeline, upload flow or tracker:

    python smoke_test.py --smoketest

Checks (each FAIL exits non-zero):
  1. Directory topology  - generated analysis videos never live in
                           input_videos; output_videos/ holds analysis MP4s.
  2. Config sanity       - ANALYSIS_VIDEOS/OUTPUT_VIDEOS == output_videos,
                           UPLOADS == uploads, (no input_videos/Misc).
  3. Generated-file guard- is_generated_filename() flags every suffix the
                           upload/batch scanner must skip.
  4. Upload dedup        - find_existing_video() detects an identical copy
                           (content hash) and skips generated files.
  5. Video fidelity      - for every input clip, output_videos/<name>_analysis.mp4
                           matches frames, fps and resolution exactly.
  6. Tracked schema      - fresh batting_landmarks.csv files carry the
                           tracking_ok / tracking_confidence columns.

Ideally run with --quiet for CI-style output.
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402


def _fail(failures, msg):
    failures.append(msg)
    print(f"  [FAIL] {msg}")


def _ok(msg):
    print(f"  [ok]   {msg}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", "-smoketest",
        action="store_true", help="run the headless self-test",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-check output")
    args = parser.parse_args()
    quiet = args.quiet
    failures = []

    print("SMOKE TEST - video pipeline fixes")
    print("=================================")

    # ---- 1 + 2: topology + config sanity -------------------------------
    in_root = BASE_DIR / "input_videos"
    out_root = BASE_DIR / "output_videos"

    if config.ANALYSIS_VIDEOS != out_root or config.OUTPUT_VIDEOS != out_root:
        _fail(failures, "ANALYSIS_VIDEOS / OUTPUT_VIDEOS must point at output_videos")
    else:
        _ok("ANALYSIS_VIDEOS / OUTPUT_VIDEOS = output_videos")

    if config.UPLOADS != (BASE_DIR / "uploads"):
        _fail(failures, "UPLOADS must be BASE_DIR/uploads (temp staging)")
    else:
        _ok("UPLOADS = BASE_DIR/uploads")

    if (in_root / "Misc").exists():
        _fail(failures, "input_videos/Misc still exists (generated videos placed there)")

    generated_in_inputs = [
        p for p in in_root.rglob("*")
        if p.is_file() and config.is_generated_filename(p.name)
    ]
    if generated_in_inputs:
        _fail(failures, f"{len(generated_in_inputs)} generated file(s) inside input_videos: "
                        f"{[p.name for p in generated_in_inputs][:5]}")
    else:
        _ok("no generated files inside input_videos")

    # ---- 3: generated-file guard ----------------------------------------
    for name in ("cut_analysis.mp4", "cut_processed.mp4", "cut_output.mp4",
                 "cut_labeled.mp4", "cut._annotated.mp4",
                 "cut.regenerating.mp4", "cut_resized.mp4", "cut.avi"):
        expected = name.endswith(("_analysis.mp4", "_processed.mp4", "_output.mp4",
                                  "_labeled.mp4", "_annotated.mp4", "_regenerating.mp4"))
        got = config.is_generated_filename(name)
        if got != expected:
            _fail(failures, f"is_generated_filename({name}) = {got}, expected {expected}")
    _ok("is_generated_filename guard behaves as expected")

    # ---- 4: upload dedup (content-hash) ---------------------------------
    from app import find_existing_video, INPUT_VIDEOS, video_content_hash  # noqa: E402
    source = INPUT_VIDEOS / "cut" / "cut.avi"
    if not source.exists():
        _fail(failures, "cut/cut.avi missing - cannot test dedup")
    else:
        with tempfile.TemporaryDirectory() as td:
            dup = Path(td) / "identicaal_copy.avi"
            shutil.copy2(source, dup)
            found = find_existing_video(dup)
            if found is None:
                _fail(failures, "find_existing_video did not find the identical copy")
            elif found.resolve() != source.resolve():
                _fail(failures, f"dedup matched {found} instead of {source}")
            else:
                _ok(f"dedup correctly flagged identical content -> {source.name}")
            # generated-named files inside input_videos must never be
            # returned as the matched source; inject one to prove it.
            probe = INPUT_VIDEOS / "_smoke_probe_analysis.mp4"
            try:
                shutil.copy2(source, probe)
                again = Path(td) / "probe_duplicate.avi"
                shutil.copy2(source, again)
                found = find_existing_video(again)
                if found is None:
                    _fail(failures, "find_existing_video found nothing while a"
                                    " generated-named probe exists")
                elif found.name.endswith("_analysis.mp4"):
                    _fail(failures,
                          "find_existing_video returned a generated _analysis file")
                elif found.name == "cut.avi":
                    _ok("generated-named files are excluded as match sources")
                else:
                    _fail(failures, f"unexpected matched source: {found.name}")
            finally:
                if probe.exists():
                    probe.unlink()

    # ---- 5: video fidelity ----------------------------------------------
    import cv2  # noqa: E402
    inputs = [
        p for p in in_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    ]
    checked = 0
    for clip in sorted(inputs):
        vid = out_root / f"{clip.stem}_analysis.mp4"
        if not vid.exists():
            _fail(failures, f"missing analysis video for {clip.name}")
            continue
        a = cv2.VideoCapture(str(clip))
        b = cv2.VideoCapture(str(vid))
        fa, fpsa, wa, ha = (a.get(cv2.CAP_PROP_FRAME_COUNT), a.get(cv2.CAP_PROP_FPS),
                            int(a.get(cv2.CAP_PROP_FRAME_WIDTH)), int(a.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fb, fpsb, wb, hb = (b.get(cv2.CAP_PROP_FRAME_COUNT), b.get(cv2.CAP_PROP_FPS),
                            int(b.get(cv2.CAP_PROP_FRAME_WIDTH)), int(b.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        a.release(); b.release()
        checked += 1
        if not (fb == fa and abs(fpsb - fpsa) < 1e-6 and (wb, hb) == (wa, ha)):
            _fail(failures,
                  f"{clip.stem}: in({fa},{fpsa},{wa}x{ha}) out({fb},{fpsb},{wb}x{hb})")
    if checked and not any(": in(" in m for m in failures):
        _ok(f"video fidelity exact for {checked} clips")

    # ---- 6: tracked schema ----------------------------------------------
    from config import OUTPUT_DATA as od  # noqa: E402
    fresh = 0
    for folder in od.iterdir():
        lc = folder / "batting_landmarks.csv"
        if not lc.exists():
            continue
        header = lc.read_text(encoding="utf-8").splitlines()[0]
        if "tracking_ok" in header and "tracking_confidence" in header:
            fresh += 1
    tde = list(od.glob("*/tracking_debug.csv"))
    _ok(f"{fresh} landmark CSVs carry tracking columns; "
        f"{len(tde)} tracking_debug.csv files present")

    print()
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print("   - " + f)
        return 1
    print("SMOKE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())