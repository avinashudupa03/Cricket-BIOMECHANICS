#!/usr/bin/env python
"""Test the integrated YOLO + MediaPipe pipeline on defence.avi"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pose_extractor_yolo import YOLOGuidedPoseExtractor
from landmark_extractor import LandmarkExtractor

from config import (
    OUTPUT_DATA, OUTPUT_VIDEOS, NO_CONSOLE_FLAGS, FAST_MODE
)

import subprocess
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(command):
    print()
    print("===================================")
    print("RUNNING:", " ".join(command))
    print("===================================")

    result = subprocess.run(
        command,
        creationflags=NO_CONSOLE_FLAGS,
        cwd=str(SCRIPT_DIR),
    )

    if result.returncode != 0:
        print()
        print("ERROR: Step failed.")
        sys.exit(result.returncode)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python test_integrated_pipeline.py <video_path> [shot_type]")
        print()
        print("Example:")
        print("python test_integrated_pipeline.py input_videos\\defence\\fast_left_0000036.avi defence")
        return

    video_path = Path(sys.argv[1]).resolve()

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    video_name = video_path.stem
    shot_type = sys.argv[2] if len(sys.argv) > 2 else Path(video_path).parent.name

    print()
    print("===================================")
    print("INTEGRATED YOLO + MEDIAPIPE PIPELINE")
    print("===================================")
    print(f"Video: {video_path}")
    print(f"Shot type: {shot_type}")
    print()

    # -----------------------------------------
    # Step 1: YOLO-guided pose extraction
    # -----------------------------------------
    model_path = SCRIPT_DIR / "models" / "pose_landmarker_heavy.task"
    
    if not model_path.exists():
        print("ERROR: Pose model not found.")
        sys.exit(1)

    pose_extractor = YOLOGuidedPoseExtractor(model_path)

    print()
    print("===================================")
    print("RUNNING: YOLO-guided pose extraction")
    print("===================================")
    print(f"Video: {video_path}")
    print(f"FPS: (auto-detected)")
    print()

    frames_data, _, batsman_confidence = pose_extractor.analyze(
        str(video_path),
        fps=0  # auto-detect
    )

    total_frames = len(frames_data)

    if batsman_confidence == 0.0:
        print("[WARN] BATSMAN NOT DETECTED - confidence is 0.0")

    pose_extractor.close()

    # -----------------------------------------
    # Step 2: Build landmark rows + tracking debug CSV
    # -----------------------------------------
    landmark_extractor = LandmarkExtractor()

    output_folder = SCRIPT_DIR / "output_data" / video_name
    output_folder.mkdir(parents=True, exist_ok=True)

    rows = []
    debug_rows = []

    for frame_number in range(total_frames):
        item = frames_data[frame_number]
        timestamp_ms = int((frame_number / 25.0) * 1000)  # default 25 fps

        row = landmark_extractor.extract(
            frame_number,
            timestamp_ms,
            item["pose"],
            tracking_ok=item["tracking_ok"],
            tracking_confidence=item["confidence"],
            interpolated=item.get("interpolated", False)
        )
        rows.append(row)

        center = item["batsman_center"]
        debug_rows.append([
            frame_number,
            timestamp_ms,
            item["n_detected"],
            item["selected_person_index"] if item["selected_person_index"] is not None else "",
            round(item["confidence"], 4),
            center[0] if center is not None else "",
            center[1] if center is not None else "",
            int(item["identity_switch_detected"]),
            int(not item["tracking_ok"]),
            "",  # phase: filled in by finalize_analysis.py
        ])

    # Save landmarks
    landmark_file = output_folder / "batting_landmarks.csv"
    landmark_extractor.output_file = landmark_file
    landmark_extractor.save(rows)

    debug_file = output_folder / "tracking_debug.csv"
    with open(debug_file, "w", newline="", encoding="utf-8") as fh:
        import csv
        writer = csv.writer(fh)
        writer.writerow([
            "frame_number", "timestamp_ms", "num_people_detected",
            "selected_person_index", "tracking_confidence",
            "batsman_center_x", "batsman_center_y",
            "identity_switch", "tracking_uncertain", "phase",
        ])
        writer.writerows(debug_rows)

    print(f"Tracking debug saved to: {debug_file}")

    # -----------------------------------------
    # Step 3-5: Run finalize_analysis (angles, phases, features, rating)
    # -----------------------------------------
    run_step([
        sys.executable,
        str(SCRIPT_DIR / "finalize_analysis.py"),
        video_name,
        shot_type
    ])

    # -----------------------------------------
    # Step 6: Generate annotated output video
    # -----------------------------------------
    run_step([
        sys.executable,
        str(SCRIPT_DIR / "output_video.py"),
        str(video_path),
        shot_type,
        str(OUTPUT_VIDEOS / f"{video_name}_analysis.mp4"),
    ])

    # -----------------------------------------
    # Complete
    # -----------------------------------------
    print()
    print("===================================")
    print("PIPELINE COMPLETE")
    print("===================================")
    print()
    print(f"Video: {video_name}")
    print(f"Output: {output_folder}")
    print()
    print("Generated files:")

    for file in [
        "batting_landmarks.csv",
        "batting_angles.csv",
        "batting_phases.csv",
        "biomechanics_features.csv",
        "shot_rating.csv",
        "injury_risk.csv",
        "injury_risk_summary.csv"
    ]:
        path = output_folder / file
        if path.exists():
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file}")

    analysis = OUTPUT_VIDEOS / f"{video_name}_analysis.mp4"
    if analysis.exists():
        print(f"  [OK] {analysis}")
    else:
        print(f"  [MISSING] {analysis}")


if __name__ == "__main__":
    main()