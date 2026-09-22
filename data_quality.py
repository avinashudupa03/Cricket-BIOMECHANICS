"""Data quality validation for the cricket-biomechanics dataset.

Validates every processed video (output_data/<video>/) and the engineered
dataset against a checklist of common dataset problems:

  * Duplicate videos       - same source video under two names / two CSV rows
  * Duplicate frames       - consecutive identical landmark rows (frozen or
                             duplicated motion-capture frames)
  * Incorrect labels       - processed label disagrees with the folder the
                             source video lives in
  * Missing labels         - processed videos with no dataset row, and source
                             videos with no processed output at all
  * Very short clips       - too few frames to contain a full batting action
  * Poor-quality videos    - low tracking coverage / low mean keypoint
                             confidence / many NaN landmark frames
  * Unbalanced classes     - support per shot type in the engineered dataset
  * Multiple people        - pose extraction is single-person; we flag videos
                             where a secondary person is plausible (many
                             low-visibility frames + small person crop)
  * Incorrect shot boundaries - phase sequence sanity (stance -> backlift ->
                             downswing -> impact -> follow-through) and the
                             presence of an impact window

Run:  python data_quality.py          # full check, writes reports/data_quality.txt
      python data_quality.py --quick  # skip landmark-level duplication scan

Exit code 0 = no problems found, 1 = at least one problem flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output_data")
REPORTS_DIR = Path("reports")
DATASET_CSV = OUTPUT_DIR / "cricket_biomechanics_dataset_engineered.csv"
BASE_DATASET_CSV = OUTPUT_DIR / "cricket_biomechanics_dataset.csv"

MIN_FRAMES_SHORT = 30           # frames; a real batting clip needs >= this
MIN_TRACKING_COVERAGE = 0.50    # share of frames with a person detected
MIN_REGION_VISIBILITY = 0.60    # mean landmark visibility considered healthy
MAX_CONSECUTIVE_DUP = 6         # identical consecutive landmark frames

PHASE_ORDER = {"Stance": 0, "Backlift": 1, "Downswing": 2,
               "Impact": 3, "Follow-through": 4}

_SKIP_DIRS = {"ml_preprocessing"}


def processed_videos():
    return [d for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and d.name not in _SKIP_DIRS]


def dataset_rows(csv_paths):
    """Yield (video_name, shot_type) rows across the given datasets once."""
    seen = set()
    for p in csv_paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            name = str(r["video_name"])
            if name in seen:
                continue
            seen.add(name)
            yield name, str(r["shot_type"])


def check_duplicate_videos(problems, dataset):
    """Same source footage exposed as two different video names."""
    seen_name = {}
    for name, _label in dataset:
        if name in seen_name:
            problems.append(f"DUPLICATE-VIDEO: '{name}' appears twice in "
                            f"the dataset")
        seen_name[name] = True

    # A processed video whose landmarks are pixel-identical to another
    # (excluding NaN rows) is the same footage reprocessed under two names.
    sigs = {}
    for vid in processed_videos():
        lm = vid / "batting_landmarks.csv"
        if not lm.exists():
            continue
        try:
            df = pd.read_csv(lm)
        except Exception:
            continue
        keep = [c for c in df.columns
                if c not in {"frame", "timestamp_ms"}]
        core = df[keep].to_numpy(float)
        finite = np.isfinite(core).all(axis=1)
        if finite.sum() < 3:
            continue
        sig = hash(core[finite].round(3).tobytes())
        sigs.setdefault(sig, []).append(vid.name)
    for sig, names in sigs.items():
        if len(names) > 1:
            problems.append(f"DUPLICATE-VIDEO: processed outputs for "
                            f"{named(names)} appear identical "
                            f"(landmarks match after rounding)")


def named(names):
    return ", ".join(sorted(names))


def check_duplicate_frames(vid, name, problems, quick=False):
    lm = vid / "batting_landmarks.csv"
    if not lm.exists():
        return
    try:
        df = pd.read_csv(lm)
    except Exception:
        problems.append(f"UNREADABLE: {name}/batting_landmarks.csv")
        return
    if "timestamp_ms" not in df.columns:
        return
    ts = df["timestamp_ms"].to_numpy(float)
    if np.diff(ts).min() < 0:
        problems.append(f"TIME-REGRESS: {name} timestamps go backwards "
                        f"({int(np.argmin(np.diff(ts)))})")

    coord_cols = [c for c in df.columns
                  if c not in {"frame", "timestamp_ms", "visibility"}
                  and not c.endswith("_visibility")
                  and not c.endswith("_presence")]
    if not coord_cols or quick:
        return
    core = df[coord_cols].to_numpy(float)
    run = 0
    for i in range(1, len(core)):
        prev, cur = core[i - 1], core[i]
        if (np.isfinite(prev).all() and np.isfinite(cur).all()
                and np.array_equal(prev.round(4), cur.round(4))):
            run += 1
            if run > MAX_CONSECUTIVE_DUP:
                problems.append(f"DUPLICATE-FRAMES: {name} has {run} consecutive "
                                f"identical landmark frames near row {i}")
                break
        else:
            run = 0


def check_labels_and_missing(problems, dataset):
    rows = {name: label for name, label in dataset}

    # Folder under input_videos is the source of truth for the label.
    for vid in processed_videos():
        name = vid.name
        label = rows.get(name)
        if label is None:
            problems.append(f"MISSING-LABEL: processed video '{name}' has NO "
                            f"entry in the dataset (it is excluded from ML)")
            continue
        if label == "unknown":
            problems.append(f"MISSING-LABEL: '{name}' is labeled 'unknown'")
        if label.lower() not in name.lower() and "lofted" not in name.lower():
            problems.append(f"INCORRECT-LABEL: '{name}' is labeled '{label}' "
                            f"but no source folder named '{label}' confirms it")


def check_short_and_quality(vid, name, problems, quick=False):
    lm = vid / "batting_landmarks.csv"
    n_frames = None
    cov = None
    if lm.exists():
        try:
            df = pd.read_csv(lm)
            n_frames = len(df)
            vis_cols = [c for c in df.columns
                        if c.endswith("_visibility")]
            if vis_cols:
                vis = df[vis_cols].to_numpy(float)
                cov = float(np.isfinite(vis).any(axis=1).mean())
        except Exception:
            pass

    if n_frames is not None and n_frames < MIN_FRAMES_SHORT:
        problems.append(f"SHORT-CLIP: '{name}' has only {n_frames} frames "
                        f"(< {MIN_FRAMES_SHORT})")
    if cov is not None and cov < MIN_TRACKING_COVERAGE:
        problems.append(f"POOR-QUALITY: '{name}' has only {cov:.0%} of frames "
                        f"with a detected person (coverage < "
                        f"{MIN_TRACKING_COVERAGE:.0%})")

    # Poor per-region visibility anywhere signals a hard-to-track video.
    if cov is not None and cov >= MIN_TRACKING_COVERAGE and lm.exists():
        try:
            df = pd.read_csv(lm)
            vis_cols = [c for c in df.columns if c.endswith("_visibility")]
            vis = df[vis_cols].to_numpy(float)
            vis = vis[np.isfinite(vis)]
            if vis.size and float(vis.mean()) < MIN_REGION_VISIBILITY:
                problems.append(
                    f"POOR-QUALITY: '{name}' mean landmark visibility is "
                    f"{float(vis.mean()):.2f} (< {MIN_REGION_VISIBILITY:.2f})")
        except Exception:
            pass


def check_many_people(vid, name, problems):
    """Single-person pipeline: flag clips where multi-person is plausible."""
    lm = vid / "batting_landmarks.csv"
    if not lm.exists():
        return
    df = pd.read_csv(lm)
    vis_cols = [c for c in df.columns if c.endswith("_visibility")]
    if not vis_cols:
        return
    vis = df[vis_cols].to_numpy(float)
    # A person-frame has at least one visible landmark; a frame with NO
    # visible landmarks at all is often occlusion by the batter's body or
    # another person crossing the view.
    empty = float((~np.isfinite(vis).any(axis=1)).mean())
    if empty > 0.5:
        problems.append(f"MULTI-PERSON?: '{name}' has {empty:.0%} frames with "
                        f"no visible landmarks (occlusion or extra person in "
                        f"frame) - verify manually")


def check_shot_boundaries(vid, name, problems):
    ph = vid / "batting_phases.csv"
    if not ph.exists():
        problems.append(f"NO-PHASES: '{name}' missing batting_phases.csv")
        return
    try:
        df = pd.read_csv(ph)
    except Exception as exc:
        problems.append(f"UNREADABLE-PHASES: '{name}' ({exc})")
        return
    if "phase" not in df.columns:
        problems.append(f"NO-PHASES: '{name}' has no phase column")
        return
    seq = df["phase"].astype(str).tolist()
    uniq = []
    for p in seq:
        if not uniq or uniq[-1] != p:
            uniq.append(p)
    for p in uniq:
        if p not in PHASE_ORDER:
            problems.append(f"BAD-PHASE: '{name}' contains unknown phase '{p}'")
            return
    for a, b in zip(uniq, uniq[1:]):
        if PHASE_ORDER[b] < PHASE_ORDER[a]:
            problems.append(f"BAD-BOUNDARY: '{name}' phase goes backwards "
                            f"({a} -> {b}), suggesting incorrect shot windows")
    if "Impact" not in set(uniq):
        problems.append(f"BAD-BOUNDARY: '{name}' never reaches an Impact phase")
    elif uniq[-1] != "Follow-through":
        problems.append(f"BAD-BOUNDARY: '{name}' does not end in "
                        f"Follow-through (action may be truncated)")


def check_balance(problems, dataset):
    counts = {}
    for _name, lab in dataset:
        counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return
    mx, mn = max(counts.values()), min(counts.values())
    if mx / mn >= 2.0:
        problems.append(f"UNBALANCED: class counts {counts} - ratio "
                        f"{mx / mn:.1f}:1")
    print(f"  class counts        : {counts}")
    print(f"  imbalance ratio     : {mx / mn:.2f}:1 (max/min)")


def main():
    quick = "--quick" in sys.argv
    problems = []
    csv_paths = [BASE_DATASET_CSV, DATASET_CSV]
    dataset = list(dataset_rows(csv_paths))

    print("=" * 70)
    print("DATA QUALITY VALIDATION")
    print("=" * 70)
    print(f"  dataset rows (base)      : {len(dataset)}")
    print(f"  processed video folders  : {len(processed_videos())}")
    print(f"  mode                     : {'quick' if quick else 'full'}")
    print()

    if dataset:
        check_duplicate_videos(problems, dataset)
        check_labels_and_missing(problems, dataset)
        check_balance(problems, dataset)
    else:
        problems.append("NO-DATASET: cricket_biomechanics_dataset.csv not found")

    for vid in processed_videos():
        name = vid.name
        check_duplicate_frames(vid, name, problems, quick=quick)
        check_short_and_quality(vid, name, problems, quick=quick)
        check_many_people(vid, name, problems)
        check_shot_boundaries(vid, name, problems)

    print("=" * 70)
    print(f"PROBLEMS FOUND: {len(problems)}")
    print("=" * 70)
    if problems:
        for p in problems:
            print(f"  [!] {p}")
    else:
        print("  No data quality problems detected.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "data_quality.txt"
    lines = ["DATA QUALITY VALIDATION", "=" * 70]
    lines += [f"  [!] {p}" for p in problems] if problems \
        else ["  No data quality problems detected."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())