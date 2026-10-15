import subprocess, sys, time
from pathlib import Path
from config import ANALYSIS_VIDEOS, NO_CONSOLE_FLAGS, is_generated_filename

BASE = Path(__file__).resolve().parent
INPUT = BASE / "input_videos"
OUT = BASE / "output_data"

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv"}
processed = [p for p in OUT.iterdir()
             if p.is_dir()
             and (p / "batting_phases.csv").exists()
             and (p / "batting_landmarks.csv").exists()]

sources = {}
for folder in INPUT.iterdir():
    if not folder.is_dir():
        continue
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in VIDEO_EXT \
                and not is_generated_filename(f.name):
            sources.setdefault(f.stem, f)

ok, fail = 0, 0
for p in sorted(processed):
    name = p.name
    out = ANALYSIS_VIDEOS / f"{name}_analysis.mp4"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {name} (exists, {out.stat().st_size} bytes)")
        ok += 1
        continue
    src = sources.get(name)
    if src is None:
        print(f"[FAIL] {name}: no source video found")
        fail += 1
        continue
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(BASE / "output_video.py"), str(src),
         src.parent.name],
        capture_output=True, text=True, cwd=str(BASE),
        creationflags=NO_CONSOLE_FLAGS,
    )
    secs = time.time() - t0
    if r.returncode == 0 and out.exists():
        print(f"[OK]   {name} in {secs:.1f}s -> {out.stat().st_size} bytes")
        ok += 1
    else:
        print(f"[FAIL] {name} ({secs:.1f}s) rc={r.returncode}")
        print((r.stderr or r.stdout or "")[-1200:])
        fail += 1

print(f"\nDONE ok={ok} fail={fail}")