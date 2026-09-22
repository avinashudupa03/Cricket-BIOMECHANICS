import subprocess
import sys
from pathlib import Path

from config import ANALYSIS_VIDEOS, NO_CONSOLE_FLAGS

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
        print("python process_video.py <video_path>")
        print()
        print("Example:")
        print("python process_video.py input_videos\\defence.avi")
        return

    video_path = Path(sys.argv[1]).resolve()

    if not video_path.exists():

        print(
            f"ERROR: Video not found: {video_path}"
        )
        return

    video_name = video_path.stem

    print()
    print("===================================")
    print("CRICKET BIOMECHANICS PIPELINE")
    print("===================================")
    print(f"Video: {video_path}")
    print()

# -----------------------------------------
    # Step 1: Extract landmarks
    # -----------------------------------------

    run_step([
        sys.executable,
        str(SCRIPT_DIR / "main.py"),
        str(video_path),
        "--no-display"
    ])

    # -----------------------------------------
    # Step 2-5: Angles, phases, features and rating run in ONE process to
    # avoid paying for four separate Python/pandas startups.
    # -----------------------------------------

    shot_type = Path(video_path).parent.name

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
        shot_type
    ])

    # -----------------------------------------
    # Complete
# -----------------------------------------

    output_folder = (
        SCRIPT_DIR /
        "output_data" /
        video_name
    )

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

    analysis = (
        ANALYSIS_VIDEOS /
        f"{video_name}_analysis.mp4"
    )

    print()

    if analysis.exists():

        print(f"  [OK] {analysis}")

    else:

        print(f"  [MISSING] {analysis}")


if __name__ == "__main__":
    main()
