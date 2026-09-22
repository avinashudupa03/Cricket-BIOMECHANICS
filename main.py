import cv2
import math
import sys
from pathlib import Path

from pose_extractor import PoseExtractor
from landmark_extractor import LandmarkExtractor

from config import POSE_STRIDE

BASE_DIR = Path(__file__).resolve().parent

# The batsman is locked over the whole clip, so POSE_STRIDE must be 1 for the
# two-pass tracker to build honest per-frame trajectories. A higher stride
# (env override) degrades identity robustness; kept supported but warned.
TRACKING_SMOOTHING_ALPHA = 0.3


def smooth_poses(frames_data, alpha=TRACKING_SMOOTHING_ALPHA):
    """Very light EMA temporal smoothing of the locked batsman landmarks.

    Applied per-coordinate only for consecutive tracked frames (reset at
    any gap), with a small alpha so the fast downswing/impact motion is not
    reduced. This only stabilises landmark jitter; it never touches the
    tracking_ok / confidence flags, and poses on TRACKING UNCERTAIN frames
    are left empty.
    """
    prev = None
    for item in frames_data:
        pose = item["pose"]
        if pose is None or not item["tracking_ok"]:
            prev = None
            continue
        if prev is None:
            prev = [LM_COPY(lm) for lm in pose]
            continue
        for cur, old in zip(pose, prev):
            if math.isfinite(cur.x) and math.isfinite(old.x):
                cur.x = old.x + alpha * (cur.x - old.x)
                cur.y = old.y + alpha * (cur.y - old.y)
                cur.z = old.z + alpha * (cur.z - old.z)
        prev = [LM_COPY(lm) for lm in pose]
    return frames_data


def LM_COPY(lm):
    clone = type(lm)()
    for attr in ("x", "y", "z", "visibility", "presence"):
        setattr(clone, attr, getattr(lm, attr))
    return clone


def main():

    # -----------------------------------------
    # Command-line options
    # -----------------------------------------

    no_display = "--no-display" in sys.argv

    # -----------------------------------------
    # Get video path from command line
    # -----------------------------------------

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if len(args) < 1:

        print("Usage:")
        print("python main.py <video_path> [--no-display]")
        print()
        print("Example:")
        print("python main.py input_videos\\drive\\drive1.mp4")
        print("(add --no-display to skip the live OpenCV preview window)")
        return

    input_video = Path(args[0]).resolve()

    if not input_video.exists():

        print(f"ERROR: Video not found: {input_video}")
        sys.exit(1)

    # -----------------------------------------
    # MediaPipe model
    # -----------------------------------------

    model_path = BASE_DIR / "models" / "pose_landmarker_full.task"

    if not model_path.exists():

        print("ERROR: Pose model not found.")
        sys.exit(1)

    # -----------------------------------------
    # Output folder based on video name
    # -----------------------------------------

    video_name = input_video.stem

    output_folder = (
        BASE_DIR /
        "output_data" /
        video_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------
    # Open video
    # -----------------------------------------

    cap = cv2.VideoCapture(
        str(input_video)
    )

    if not cap.isOpened():

        print("ERROR: Could not open video.")
        sys.exit(1)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if POSE_STRIDE != 1:
        print(
            f"[WARN] POSE_STRIDE={POSE_STRIDE}: the tracker detects every "
            f"frame regardless; stride is ignored for correctness."
        )

    # -----------------------------------------
    # Two-pass temporal tracking
    # -----------------------------------------

    pose_extractor = PoseExtractor(
        model_path
    )

    landmark_extractor = LandmarkExtractor()

    print()
    print("===================================")
    print("CRICKET BIOMECHANICS ANALYSIS")
    print("===================================")
    print(f"RUNNING: main.py")
    print(f"Video: {input_video}")
    print(f"FPS: {fps}")
    print(f"Total frames: {input_frames}")
    print(f"Pose stride: {POSE_STRIDE} (all frames are detected)")
    print()
    print("Pass 1: detecting player poses on every frame (full frame + "
          "central band)...")
    print()

    frames_data, best_traj = pose_extractor.analyze(
        str(input_video),
        fps=fps
    )

    total_frames = len(frames_data)

    if best_traj is None:
        print("[WARN] No batsman trajectory could be locked - all frames "
              "will be TRACKING UNCERTAIN.")

    # -----------------------------------------
    # Light temporal smoothing of the locked poses
    # -----------------------------------------

    smooth_poses(frames_data)

    # -----------------------------------------
    # Build landmark rows + tracking debug CSV
    # -----------------------------------------

    rows = []
    debug_rows = []

    for frame_number in range(total_frames):

        item = frames_data[frame_number]

        timestamp_ms = int(
            (frame_number / fps) * 1000
        )

        row = landmark_extractor.extract(
            frame_number,
            timestamp_ms,
            item["pose"],
            tracking_ok=item["tracking_ok"],
            tracking_confidence=item["confidence"]
        )

        rows.append(row)

        center = item["batsman_center"]

        debug_rows.append([
            frame_number,
            item["n_detected"],
            item["selected_person_index"] if item["selected_person_index"] is not None else "",
            int(item["tracking_ok"]),
            round(item["confidence"], 4),
            int(item["identity_switch_detected"]),
            center[0] if center is not None else "",
            center[1] if center is not None else "",
        ])

    # -----------------------------------------
    # Cleanup
    # -----------------------------------------

    pose_extractor.close()

    # -----------------------------------------
    # Save landmarks
    # -----------------------------------------

    landmark_file = (
        output_folder /
        "batting_landmarks.csv"
    )

    landmark_extractor.output_file = (
        landmark_file
    )

    landmark_extractor.save(rows)

    debug_file = (
        output_folder /
        "tracking_debug.csv"
    )

    with open(debug_file, "w", newline="", encoding="utf-8") as fh:
        import csv
        writer = csv.writer(fh)
        writer.writerow([
            "frame_number",
            "number_of_detected_people",
            "selected_person_index",
            "tracking_ok",
            "tracking_confidence",
            "identity_switch_detected",
            "batsman_center_x",
            "batsman_center_y",
        ])
        writer.writerows(debug_rows)

    print(f"Tracking debug saved to: {debug_file}")

    # -----------------------------------------
    # Tracking summary
    # -----------------------------------------

    tracked = sum(1 for it in frames_data if it["tracking_ok"])
    uncertain = total_frames - tracked
    switches = sum(1 for it in frames_data if it["identity_switch_detected"])

    mean_conf = (
        sum(it["confidence"] for it in frames_data) / total_frames
        if total_frames else 0.0
    )

    print()
    print("===================================")
    print("PROCESSING COMPLETE")
    print("===================================")
    print(f"Frames processed: {total_frames}")
    print(f"Output folder: {output_folder}")
    print(f"Landmarks: {landmark_file}")
    print()
    print("TRACKING SUMMARY")
    print("----------------")
    print(f"Batsman tracked (tracking_ok): {tracked} ({tracked / max(total_frames, 1) * 100:.1f}%)")
    print(f"TRACKING UNCERTAIN frames: {uncertain} ({uncertain / max(total_frames, 1) * 100:.1f}%)")
    print(f"Identity-switch/defended-lock frames: {switches}")
    print(f"Mean tracking confidence: {mean_conf:.3f}")
    print()

    # Optional live preview (kept for parity with the old --no-display flag).
    if not no_display and frames_data:
        cap = cv2.VideoCapture(str(input_video))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            item = frames_data[idx]
            if item["tracking_ok"] and item["pose"] is not None:
                pose_extractor.draw_pose(frame, item["pose"])
                cv2.putText(frame, "BATSMAN TRACKING", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "TRACKING UNCERTAIN", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("Cricket Batting - Pose Detection", frame)
            idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()