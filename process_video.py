
import subprocess
import sys
import time
from pathlib import Path

from config import ANALYSIS_VIDEOS, NO_CONSOLE_FLAGS
from database import update_pipeline_job

SCRIPT_DIR = Path(__file__).resolve().parent

STAGE_POSE = "Detecting player pose and extracting landmarks"
STAGE_ANALYSIS = "Calculating joint angles, batting phases, features and ratings"
STAGE_VIDEO = "Rendering annotated video"

STAGE_UPLOAD = "Uploaded video"


def report_stage(job_id, current, completed_steps):
    """Update progress without interrupting analysis if status reporting fails."""
    if not job_id:
        return

    try:
        update_pipeline_job(
            job_id,
            status="running",
            current=current,
            steps=completed_steps,
        )
    except Exception as exc:
        print(f"WARNING: Could not update progress: {exc}", flush=True)


def run_step(command):
    print()
    print("===================================")
    print("RUNNING:", " ".join(map(str, command)))
    print("===================================", flush=True)

    result = subprocess.run(
        command,
        creationflags=NO_CONSOLE_FLAGS,
        cwd=str(SCRIPT_DIR),
    )

    if result.returncode != 0:
        print()
        print("ERROR: Step failed.", flush=True)
        sys.exit(result.returncode)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python process_video.py <video_path> [shot_type] [job_id]")
        print()
        print("Example:")
        print(r"python process_video.py input_videos\defence.avi defence")
        return

    video_path = Path(sys.argv[1]).resolve()

    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", flush=True)
        sys.exit(1)

    video_name = video_path.stem

    shot_type = (
        sys.argv[2]
        if len(sys.argv) > 2
        else video_path.parent.name
    )

    job_id = sys.argv[3] if len(sys.argv) > 3 else None

    print()
    print("===================================")
    print("CRICKET BIOMECHANICS PIPELINE")
    print("===================================")
    print(f"Video: {video_path}")
    print(f"Shot type: {shot_type}")
    print()

    # Step 1: Extract pose landmarks.
    report_stage(
        job_id,
        STAGE_POSE,
        [STAGE_UPLOAD],
    )

    run_step([
        sys.executable,
        str(SCRIPT_DIR / "main.py"),
        str(video_path),
        "--no-display",
    ])

    # Step 2: Calculate angles, phases, features, ratings and injury risk.
    report_stage(
        job_id,
        STAGE_ANALYSIS,
        [STAGE_UPLOAD, STAGE_POSE],
    )

    run_step([
        sys.executable,
        str(SCRIPT_DIR / "finalize_analysis.py"),
        video_name,
        shot_type,
    ])

    # Step 3: Generate the annotated output video.
    report_stage(
        job_id,
        STAGE_VIDEO,
        [STAGE_UPLOAD, STAGE_POSE, STAGE_ANALYSIS],
    )

    run_step([
        sys.executable,
        str(SCRIPT_DIR / "output_video.py"),
        str(video_path),
        shot_type,
    ])

    # Verify generated outputs.
    output_folder = SCRIPT_DIR / "output_data" / video_name

    print()
    print("===================================")
    print("PIPELINE COMPLETE")
    print("===================================")
    print(f"Video: {video_name}")
    print(f"Output: {output_folder}")
    print()
    print("Generated files:")

    for filename in [
        "batting_landmarks.csv",
        "batting_angles.csv",
        "batting_phases.csv",
        "biomechanics_features.csv",
        "shot_rating.csv",
        "injury_risk.csv",
        "injury_risk_summary.csv",
    ]:
        output_file = output_folder / filename
        status = "[OK]" if output_file.exists() else "[MISSING]"
        print(f"  {status} {filename}")

    analysis_video = ANALYSIS_VIDEOS / f"{video_name}_analysis.mp4"

    print()
    if analysis_video.exists():
        print(f"  [OK] {analysis_video}")
    else:
        print(f"  [MISSING] {analysis_video}")


if __name__ == "__main__":
    main()
