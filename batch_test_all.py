#!/usr/bin/env python
"""Batch test the integrated YOLO + MediaPipe pipeline on all videos."""

import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pose_extractor_yolo import YOLOGuidedPoseExtractor
from landmark_extractor import LandmarkExtractor

from config import OUTPUT_DATA, OUTPUT_VIDEOS, NO_CONSOLE_FLAGS

import subprocess
import cv2
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(command):
    print(f"  Running: {' '.join(command)}")
    result = subprocess.run(
        command,
        creationflags=NO_CONSOLE_FLAGS,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout, result.stderr


def test_single_video(video_path, shot_type, output_csv_writer):
    """Test a single video and record results."""
    video_name = Path(video_path).stem
    video_path = Path(video_path)
    
    result = {
        "video_name": video_name,
        "shot_type": shot_type,
        "num_frames": 0,
        "yolo_detection_success": False,
        "selected_batsman_id": None,
        "num_id_switches": 0,
        "frames_batsman_lost": 0,
        "mediapipe_pose_success_rate": 0.0,
        "output_csv_ok": False,
        "output_video_ok": False,
        "errors_warnings": [],
    }
    
    print(f"\n{'='*60}")
    print(f"Testing: {video_name} ({shot_type})")
    print(f"{'='*60}")
    
    model_path = SCRIPT_DIR / "models" / "pose_landmarker_heavy.task"
    
    if not model_path.exists():
        result["errors_warnings"].append("Pose model not found")
        return result
    
    try:
        from pose_extractor_yolo import YOLOGuidedPoseExtractor
        
        pose_extractor = YOLOGuidedPoseExtractor(str(model_path))
        
        frames_data, _, batsman_confidence = pose_extractor.analyze(
            str(video_path),
            fps=0  # auto-detect
        )
        
        total_frames = len(frames_data)
        result["num_frames"] = total_frames
        result["yolo_detection_success"] = True
        
        # Collect tracking stats
        batsman_ids_per_frame = []
        frames_with_batsman = 0
        frames_roi_mp = 0
        frames_mp_failed = 0
        
        for i, frame_data in enumerate(frames_data):
            if frame_data["batsman_detected"]:
                frames_with_batsman += 1
            if frame_data["tracking_ok"]:
                frames_roi_mp += 1
            else:
                frames_mp_failed += 1
            
            # Use actual track_id from selection, not array index
            sel_id = frame_data.get("selected_batsman_track_id")
            if sel_id is not None:
                batsman_ids_per_frame.append(sel_id)
            else:
                # Fallback: if we have the selection object, use its track_id
                # This is a fallback - ideally frames_data should store track_id
                bid = frame_data.get("selected_person_index")
                if bid is not None:
                    # This is array index, not track_id - mark as such
                    batsman_ids_per_frame.append(f"idx_{bid}")
                else:
                    batsman_ids_per_frame.append(None)
        
        # Count ID switches
        prev_id = None
        switches = 0
        for bid in batsman_ids_per_frame:
            if bid is not None and bid != prev_id:
                if prev_id is not None:
                    switches += 1
                prev_id = bid
        
        result["selected_batsman_id"] = prev_id
        result["num_id_switches"] = switches
        result["frames_batsman_lost"] = total_frames - frames_with_batsman
        result["mediapipe_pose_success_rate"] = frames_roi_mp / max(total_frames, 1) if total_frames > 0 else 0.0
        
        # Save landmarks and debug CSV
        output_folder = OUTPUT_DATA / video_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        landmark_extractor = LandmarkExtractor()
        
        rows = []
        debug_rows = []
        
        for frame_number in range(total_frames):
            item = frames_data[frame_number]
            timestamp_ms = int((frame_number / 25.0) * 1000)
            
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
                "",
            ])
        
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
        
        # Run finalize_analysis
        finalize_cmd = [
            "python", str(SCRIPT_DIR / "finalize_analysis.py"),
            video_name, shot_type
        ]
        success, stdout, stderr = run_step(finalize_cmd)
        if not success:
            result["errors_warnings"].append(f"finalize_analysis failed: {stderr[:500]}")
        
        # Check output CSVs
        required_csvs = [
            "batting_landmarks.csv",
            "batting_angles.csv",
            "batting_phases.csv",
            "biomechanics_features.csv",
            "shot_rating.csv",
            "injury_risk.csv",
            "injury_risk_summary.csv"
        ]
        
        output_folder = OUTPUT_DATA / video_name
        csv_ok = all((output_folder / f).exists() for f in required_csvs)
        result["output_csv_ok"] = csv_ok
        if not csv_ok:
            missing = [f for f in required_csvs if not (output_folder / f).exists()]
            result["errors_warnings"].append(f"Missing CSVs: {missing}")
        
        # Generate output video
        video_cmd = [
            "python", str(SCRIPT_DIR / "output_video.py"),
            str(video_path), shot_type,
            str(OUTPUT_VIDEOS / f"{video_name}_analysis.mp4"),
        ]
        success, stdout, stderr = run_step(video_cmd)
        if not success:
            result["errors_warnings"].append(f"output_video failed: {stderr[:500]}")
        
        video_file = OUTPUT_VIDEOS / f"{video_name}_analysis.mp4"
        result["output_video_ok"] = video_file.exists()
        
        # Verify output video
        if video_file.exists():
            cap = cv2.VideoCapture(str(video_file))
            if cap.isOpened():
                out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if out_frames != len(frames_data):
                    result["errors_warnings"].append(f"Frame count mismatch: expected {len(frames_data)}, got {out_frames}")
            else:
                result["errors_warnings"].append("Output video could not be opened")
                result["output_video_ok"] = False
        
        pose_extractor.close()
        
    except Exception as e:
        result["errors_warnings"].append(f"Exception: {type(e).__name__}: {str(e)[:500]}")
        import traceback
        result["errors_warnings"].append(traceback.format_exc()[:1000])
    
    # Print summary
    print(f"  Frames: {result['num_frames']}")
    print(f"  Selected ID: {result['selected_batsman_id']}")
    print(f"  ID Switches: {result['num_id_switches']}")
    print(f"  Frames Lost: {result['frames_batsman_lost']}")
    print(f"  MP Success Rate: {result['mediapipe_pose_success_rate']:.1%}")
    print(f"  CSV OK: {result['output_csv_ok']}")
    print(f"  Video OK: {result['output_video_ok']}")
    if result["errors_warnings"]:
        print(f"  Errors: {result['errors_warnings']}")
    
    # Write to CSV
    output_csv_writer.writerow([
        result["video_name"],
        result["shot_type"],
        result["num_frames"],
        result["yolo_detection_success"],
        result["selected_batsman_id"],
        result["num_id_switches"],
        result["frames_batsman_lost"],
        f"{result['mediapipe_pose_success_rate']:.3f}",
        result["output_csv_ok"],
        result["output_video_ok"],
        "; ".join(result["errors_warnings"]),
    ])
    
    return result


def main():
    # Discover all videos
    input_root = SCRIPT_DIR / "input_videos"
    videos = []
    
    for shot_folder in sorted(input_root.iterdir()):
        if not shot_folder.is_dir():
            continue
        shot_type = shot_folder.name
        for video_file in sorted(shot_folder.glob("*.avi")):
            videos.append((video_file, shot_type))
    
    print(f"Found {len(videos)} videos to test")
    
    # Output CSV
    report_path = OUTPUT_DATA / "batch_test_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_name", "shot_type", "num_frames", "yolo_detection_success",
            "selected_batsman_id", "num_id_switches", "frames_batsman_lost",
            "mediapipe_pose_success_rate", "output_csv_ok", "output_video_ok",
            "errors_warnings"
        ])
        
        all_results = []
        for video_path, shot_type in videos:
            result = test_single_video(video_path, shot_type, writer)
            all_results.append(result)
        
        # Print summary table
        print("\n" + "="*100)
        print("BATCH TEST SUMMARY")
        print("="*100)
        print(f"{'Video':<25} {'Type':<12} {'Frames':>6} {'MP Rate':>8} {'Switches':>8} {'Lost':>6} {'CSV':>4} {'Video':>5} {'Errors'}")
        print("-"*100)
        
        for r in all_results:
            mp_rate = f"{r['mediapipe_pose_success_rate']:.1%}"
            print(f"{r['video_name']:<25} {r['shot_type']:<12} {r['num_frames']:>6} {mp_rate:>8} "
                  f"{r['num_id_switches']:>8} {r['frames_batsman_lost']:>6} "
                  f"{'OK' if r['output_csv_ok'] else 'FAIL':>4} {'OK' if r['output_video_ok'] else 'FAIL':>5} "
                  f"{len(r['errors_warnings'])}")
        
        print(f"\nReport saved to: {report_path}")
        
        # Also save as JSON for detailed analysis
        json_path = OUTPUT_DATA / "batch_test_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Detailed JSON saved to: {json_path}")


if __name__ == "__main__":
    main()