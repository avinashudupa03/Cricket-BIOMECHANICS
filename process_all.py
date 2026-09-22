"""Batch-process every video in input_videos/ using multiprocessing.

Multiple videos are analysed in parallel via a process pool. Each worker
runs the full per-video pipeline (process_video.py) in its own Python
process so MediaPipe, OpenCV and pandas never share memory.

Usage:
    python process_all.py                 # auto-detect worker count
    python process_all.py --workers 4     # force 4 parallel workers
    python process_all.py --workers 1     # sequential (original behaviour)
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import MAX_WORKERS, NO_CONSOLE_FLAGS, is_generated_filename

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def _process_one(video_path):
    """Run the pipeline for a single video. Designed to be called in a worker
    process via ProcessPoolExecutor, so it must be a top-level function.

    Returns (video_path, success: bool, elapsed: float, error: str | None).
    """
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "process_video.py", str(video_path)],
            creationflags=NO_CONSOLE_FLAGS,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            err = result.stderr.strip()[-500:] if result.stderr else "Unknown error"
            return (str(video_path), False, elapsed, err)
        return (str(video_path), True, elapsed, None)
    except Exception as exc:
        return (str(video_path), False, time.time() - start, str(exc))


def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process cricket batting videos in parallel."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS}, auto-detected).",
    )
    args = parser.parse_args()
    workers = max(1, args.workers)

    input_folder = Path(__file__).resolve().parent / "input_videos"

    videos = [
        p for p in input_folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not is_generated_filename(p.name)
    ]

    if not videos:
        print("No videos found in input_videos.")
        return

    print()
    print("===================================")
    print("BATCH CRICKET BIOMECHANICS ANALYSIS")
    print("===================================")
    print(f"Videos found:  {len(videos)}")
    print(f"Workers:       {workers}")
    print()

    completed = []
    failed = []
    batch_start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_video = {
            pool.submit(_process_one, v): v for v in videos
        }

        for future in as_completed(future_to_video):
            video_path, ok, elapsed, error = future.result()
            name = Path(video_path).stem

            if ok:
                completed.append((name, elapsed))
                status = f"  [OK]   {name}  ({_fmt_time(elapsed)})"
            else:
                failed.append((name, error, elapsed))
                status = f"  [FAIL] {name}  ({_fmt_time(elapsed)})"
                if error:
                    # Show a short error summary (last line of traceback)
                    short = str(error).splitlines()[-1][:120]
                    print(f"         {short}")

            done = len(completed) + len(failed)
            print(f"  ({done}/{len(videos)}) {status}")

    total_elapsed = time.time() - batch_start

    print()
    print("===================================")
    print("BATCH PROCESSING COMPLETE")
    print("===================================")
    print(f"  Total time:  {_fmt_time(total_elapsed)}")
    print(f"  Completed:   {len(completed)}")
    print(f"  Failed:      {len(failed)}")

    if completed:
        avg = sum(t for _, t in completed) / len(completed)
        print(f"  Avg / video: {_fmt_time(avg)}")

    if failed:
        print()
        print("Failed videos:")
        for name, error, _ in failed:
            short = str(error).splitlines()[-1][:100]
            print(f"  - {name}: {short}")

    print()


if __name__ == "__main__":
    main()
