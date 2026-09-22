"""Generate a professional annotated analysis video.

This step runs AFTER main.py / angle_calculator.py / phase_detector.py /
biomechanics_analyzer.py have produced their CSVs. It REUSES:

  * batting_landmarks.csv  - normalized batsman-only poses per frame (tracked)
                              produced by the two-pass PoseExtractor logic
  * batting_phases.csv     - per-frame phase, movement score and joint angles

It does NOT re-run MediaPipe and does NOT duplicate any pose-detection logic.

Fidelity contract (root-cause fix)
----------------------------------
The annotated video is a FAITHFUL, frame-accurate copy of the source clip:
exactly ONE output frame per input frame, at the source FPS and resolution,
covering the FULL source duration. No slow-motion frame repetition and no
minimum-duration padding is ever applied. The only transformation is the
overlay + a lossy H.264 encode for browser playback.

Output: output_videos/<video_name>_analysis.mp4  (browser-playable MP4).
Analysis videos are NEVER written into input_videos, so the upload / batch
scanner can never pick up a generated file and re-upload it.

Overlay honesty
---------------
The skeleton is drawn ONLY on frames where the identity lock is held
(tracking_ok). On frames flagged TRACKING UNCERTAIN the overlay shows that
state instead of re-attaching to the keeper, bowler or non-striker.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

from config import OUTPUT_VIDEOS, NO_CONSOLE_FLAGS

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Landmark / connection definitions (must match pose_extractor.py)
# ---------------------------------------------------------------------------
LANDMARK_COLS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]

# Landmark indices used as the localised "center" for the target box.
BODY_JOINTS = [11, 12, 23, 24]

PHASE_COLORS = {
    "Stance": (96, 165, 250),        # soft blue
    "Backlift": (247, 202, 24),      # amber
    "Downswing": (66, 212, 122),     # green
    "Impact": (255, 84, 84),         # red
    "Follow-through": (168, 85, 247) # purple
}


def pick_writer(path, fps, size):
    """Return an open VideoWriter for the annotated output.

    Prefer mp4v first: on this machine the OpenH264 (H.264) encoder is not
    available (the OpenH264 DLL is missing), so an avc1 writer "opens" but
    logs a real encoder failure and can yield an unreliable file. The mp4v
    intermediate is cheap to produce and is then re-encoded to browser-safe
    H.264 by :func:`transcode_to_h264`.
    """
    order = ["mp4v", "avc1"]
    for codec in order:
        try:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (size[0], size[1]),
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
        except Exception:
            continue
    raise RuntimeError("No usable MP4 video encoder found (tried mp4v, avc1).")


def transcode_to_h264(path):
    """Re-encode a written video to H.264 (yuv420p, faststart) in place.

    Browsers cannot decode the mp4v/avc1 containers OpenCV tends to produce, so
    the generated file is transcoded with the ffmpeg binary bundled with
    imageio-ffmpeg for guaranteed HTML5 playback. Frame count, fps and
    duration are preserved by the transcode. Returns True on success.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    except Exception:
        return False

    tmp = path.with_suffix(".h264_tmp.mp4")
    if tmp.exists():
        tmp.unlink()

    command = [
        ffmpeg, "-y", "-i", str(path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(tmp),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            creationflags=NO_CONSOLE_FLAGS,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            path.unlink(missing_ok=True)
            tmp.rename(path)
            return True
    except Exception:
        pass

    if tmp.exists():
        tmp.unlink()
    return False


def get_center(landmarks):
    pts = [landmarks[i] for i in BODY_JOINTS if landmarks.get(i) is not None]
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def draw_overlay(frame, info):
    """Overlay the batsman-only skeleton + professional HUD on a frame.

    The skeleton is drawn ONLY when tracking_ok. Otherwise the frame shows
    TRACKING UNCERTAIN - the skeleton is never moved onto another player.
    """
    pose = info.get("pose")
    track_ok = bool(info.get("track_ok", False))
    track_conf = info.get("track_conf", 0.0)
    phase = info.get("phase", "")
    h, w = frame.shape[:2]
    phase_col = PHASE_COLORS.get(phase, (200, 200, 200))

    # ---- Skeleton (batsman only, only when identity lock is held) --------
    if pose and track_ok:
        pts = {}
        for idx, (nx, ny) in pose.items():
            x = int(nx * w)
            y = int(ny * h)
            pts[idx] = (x, y)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 5, (102, 217, 255), -1)
                cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
        for a, b in CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], (102, 217, 255), 2)

    # ---- Phase flag (right) -----------------------------------------------
    phase_label = "IMPACT" if phase == "Impact" else phase
    (tw, th), _ = cv2.getTextSize(
        phase_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    )
    tx = w - tw - 24
    ty = 34
    if phase == "Impact":
        cv2.rectangle(frame, (tx - 14, 12), (w - 10, ty + th + 8),
                      phase_col, -1)
        cv2.rectangle(frame, (tx - 14, 12), (w - 10, ty + th + 8),
                      (255, 255, 255), 1)
        cv2.putText(frame, phase_label, (tx, ty + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    else:
        cv2.putText(frame, phase_label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, phase_col, 2)
        cv2.putText(frame, "PHASE", (tx, ty - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ---- Tracking state (left, under the title) ---------------------------
    track_label = f"TRACK CONF {track_conf:.2f}" if track_ok else "TRACKING UNCERTAIN"
    track_col = (102, 217, 255) if track_ok else (255, 84, 84)
    cv2.putText(frame, track_label, (16, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, track_col, 1)

    # ---- Title (left) ------------------------------------------------------
    cv2.putText(frame, info["video_name"], (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Shot: {info['shot_type']}", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PHASE_COLORS["Stance"], 1)

    # ---- Bottom panel ------------------------------------------------------
    pan_h = 74
    cv2.rectangle(frame, (0, h - pan_h), (w, h), (10, 12, 22), -1)
    cv2.line(frame, (0, h - pan_h), (w, h - pan_h), (40, 44, 60), 1)

    left = 16
    top = h - pan_h + 14
    cv2.putText(frame, f"FRAME {info['frame']:04d}", (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"MOVEMENT {info.get('movement', 0.0):.2f}", (left, top + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (102, 217, 255), 1)
    cv2.putText(frame, f"IMPACT FRAME {info.get('impact_frame', '--')}",
                (left, top + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_col, 1)

    # key angles at right side of panel
    ang = info.get("angles", {})
    labels = [
        ("ELBOW", "elbow"),
        ("KNEE", "knee"),
        ("SHOULDER", "shoulder"),
        ("HIP", "hip"),
    ]
    col_x = w - 18
    for name, key in labels:
        val = ang.get(key)
        text = f"{name} {int(round(val)) if val is not None else '--'}"
        (tww, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = col_x - tww
        cv2.putText(frame, text, (x, top), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1)
        top += 24

    return frame


def build_frame_landmarks(df_land):
    """Return ({frame: {landmark_index: (nx, ny)}}, {frame: (ok, conf)}).

    Backwards compatible: legacy CSVs (pre-tracking-columns) are treated as
    fully tracked (ok=True when a pose exists) so old analysis still renders;
    fresh CSVs with the tracking_ok / tracking_confidence columns drive the
    honest TRACKING UNCERTAIN rendering.
    """
    has_tracking = "tracking_ok" in df_land.columns \
        and "tracking_confidence" in df_land.columns
    out = {}
    tracking = {}
    for _, row in df_land.iterrows():
        fr = int(row["frame"])
        pose = {}
        for idx, name in enumerate(LANDMARK_COLS):
            x = row.get(f"{name}_x")
            y = row.get(f"{name}_y")
            if pd.notna(x) and pd.notna(y):
                pose[idx] = (float(x), float(y))
        out[fr] = pose if pose else None
        if has_tracking:
            ok = bool(row.get("tracking_ok", 0))
            try:
                conf = float(row.get("tracking_confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
        else:
            ok = bool(pose)
            conf = 1.0
        tracking[fr] = (ok, conf)
    return out, tracking


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python output_video.py <video_path> [shot_type] [output_path]")
        return

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return

    shot_type = sys.argv[2] if len(sys.argv) > 2 else ""
    if not shot_type:
        shot_type = video_path.parent.name  # input_videos/<shot_type>/...

    video_name = video_path.stem
    out_folder = BASE_DIR / "output_data" / video_name
    phases_file = out_folder / "batting_phases.csv"
    land_file = out_folder / "batting_landmarks.csv"

    if not phases_file.exists() or not land_file.exists():
        print("ERROR: batting_phases.csv / batting_landmarks.csv not found.")
        print("Run the pipeline first (process_video.py).")
        return

    df_phases = pd.read_csv(phases_file)
    df_land = pd.read_csv(land_file)
    frame_poses, frame_tracking = build_frame_landmarks(df_land)

    # phase start/end + impact frame for the timeline
    impact_frames = df_phases[df_phases["phase"] == "Impact"]["frame"]
    impact_frame = int(impact_frames.iloc[len(impact_frames) // 2]) \
        if len(impact_frames) else None

    # per-frame lookup
    angles_by_frame = {}
    phase_by_frame = {}
    movement_by_frame = {}
    for _, row in df_phases.iterrows():
        fr = int(row["frame"])
        phase_by_frame[fr] = row["phase"]
        movement_by_frame[fr] = float(
            row["movement_smooth"]) if pd.notna(row["movement_smooth"]) else 0.0
        angles_by_frame[fr] = {
            "elbow": (row.get("left_elbow_angle") if pd.notna(row.get("left_elbow_angle")) else row.get("right_elbow_angle")),
            "knee": (row.get("left_knee_angle") if pd.notna(row.get("left_knee_angle")) else row.get("right_knee_angle")),
            "shoulder": (row.get("left_shoulder_angle") if pd.notna(row.get("left_shoulder_angle")) else row.get("right_shoulder_angle")),
            "hip": (row.get("left_hip_angle") if pd.notna(row.get("left_hip_angle")) else row.get("right_hip_angle")),
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: Could not open video for annotated output.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not fps:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480
    input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_duration = input_frames / fps if fps else 0.0

    out_videos_dir = OUTPUT_VIDEOS
    out_videos_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else out_videos_dir / f"{video_name}_analysis.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer, codec = pick_writer(out_path, fps, (width, height))
    print(f"[output_video] encoder: {codec} -> {out_path}")

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ok_flag, conf = frame_tracking.get(frame_no, (False, 0.0))

        info = {
            "frame": frame_no,
            "video_name": video_name,
            "shot_type": shot_type or "Unknown",
            "phase": phase_by_frame.get(frame_no, ""),
            "impact_frame": impact_frame,
            "movement": movement_by_frame.get(frame_no, 0.0),
            "angles": angles_by_frame.get(frame_no, {}),
            "pose": frame_poses.get(frame_no),
            "track_ok": ok_flag,
            "track_conf": conf,
        }
        draw_overlay(frame, info)
        # Fidelity: exactly ONE output frame per input frame.
        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()

    if transcode_to_h264(out_path):
        print("[output_video] transcoded to H.264 for browser playback")
    else:
        print("[output_video] NOTE: kept original encoding "
              "(may not play in all browsers)")

    # ------------------------------------------------------------------
    # VIDEO VALIDATION - verify the annotated output is faithful to input.
    # ------------------------------------------------------------------
    verify = cv2.VideoCapture(str(out_path))
    out_frames = int(verify.get(cv2.CAP_PROP_FRAME_COUNT)) \
        if verify.isOpened() else -1
    out_fps = verify.get(cv2.CAP_PROP_FPS) if verify.isOpened() else 0.0
    out_w = int(verify.get(cv2.CAP_PROP_FRAME_WIDTH)) if verify.isOpened() else -1
    out_h = int(verify.get(cv2.CAP_PROP_FRAME_HEIGHT)) if verify.isOpened() else -1
    out_duration = (out_frames / out_fps) if (verify.isOpened() and out_fps) else 0.0
    if verify.isOpened():
        verify.release()

    print()
    print("VIDEO VALIDATION")
    print("----------------")
    print(f"  Input : frames={input_frames} fps={fps:.2f} "
          f"duration={input_duration:.3f}s resolution={width}x{height}")
    print(f"  Output: frames={out_frames} fps={out_fps:.2f} "
          f"duration={out_duration:.3f}s resolution={out_w}x{out_h}")
    print(f"  Frame-accurate (input==output frames): "
          f"{'PASS' if out_frames == input_frames else 'FAIL'}")
    print(f"  Resolution preserved: "
          f"{'PASS' if (out_w, out_h) == (width, height) else 'FAIL'}")

    print()
    print(f"[output_video] COMPLETE: {out_path} "
          f"({frame_no} frames, {frame_no / fps:.2f}s)")


if __name__ == "__main__":
    main()