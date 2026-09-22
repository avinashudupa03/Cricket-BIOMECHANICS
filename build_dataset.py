import hashlib
import pandas as pd
from pathlib import Path

from config import is_generated_filename


VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}


def find_source_video(video_name, input_root):
    """Locate the source video file for a processed video name.

    The video_name is the stem of the original video file (no extension), so we
    search for any video file *inside a subfolder* whose stem equals video_name.
    Returns the resolved file Path (or None). The parent folder of that file is
    the shot type.
    """
    if not input_root.exists():
        return None

    for folder in input_root.iterdir():
        if not folder.is_dir():
            continue
        for video_file in folder.iterdir():
            if not video_file.is_file():
                continue
            if video_file.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if is_generated_filename(video_file.name):
                continue
            if video_file.stem == video_name:
                return video_file.resolve()
    return None


def content_hash(video_file, chunk_size=1 << 20):
    """Return an MD5 hash of a video file's bytes.

    Used to detect identical (duplicated) clips. Identical input videos produce
    identical analysis outputs, so keeping more than one copy in the ML dataset
    would leak identical content across LOOCV train/test folds and inflate the
    reported accuracy.
    """
    digest = hashlib.md5()
    with open(video_file, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main():

    output_root = Path("output_data")
    input_root = Path("input_videos")
    dataset_file = output_root / "cricket_biomechanics_dataset.csv"

    feature_files = sorted(
        output_root.glob("*/biomechanics_features.csv")
    )

    if not feature_files:
        print("No biomechanics feature files found.")
        print("Run process_all.py first, then rebuild the dataset.")
        return

    all_data = []
    video_names = set()
    unknown = []
    duplicates = []
    seen_hashes = {}

    for file in feature_files:

        video_name = file.parent.name

        if video_name in video_names:
            print(f"[WARN] Duplicate video detected, skipped: {video_name}")
            continue

        source = find_source_video(video_name, input_root)

        if source is not None:
            digest = content_hash(source)
            canonical = seen_hashes.get(digest)
            if canonical is not None:
                duplicates.append((video_name, canonical))
                print(f"[SKIP] '{video_name}' is an identical copy of "
                      f"'{canonical}' (same video content) - kept once only.")
                continue
            seen_hashes[digest] = video_name

        shot_type = source.parent.name if source is not None else None

        if shot_type is None:
            shot_type = "unknown"
            unknown.append(video_name)

        df = pd.read_csv(file)

        # Poor-quality guard: clips where the pose tracker never locked on the
        # batsman (0% coverage) or that are far too short to contain a shot
        # produce only NaN features and would poison the dataset with imputed
        # noise. Skip them explicitly as POOR QUALITY instead.
        coverage = None
        n_frames = None
        if "quality_tracking_coverage" in df.columns:
            _c = df["quality_tracking_coverage"].fillna(0)
            coverage = float(_c.iloc[0]) if len(_c) else None
        if "quality_frame_count" in df.columns:
            _f = df["quality_frame_count"].fillna(0)
            n_frames = int(_f.iloc[0]) if len(_f) else None

        if coverage is not None and coverage <= 0:
            print(f"[SKIP] '{video_name}' has 0% pose-tracking coverage "
                  f"(poor quality) - excluded from the dataset.")
            continue
        if n_frames is not None and n_frames < 10:
            print(f"[SKIP] '{video_name}' only has {n_frames} frame(s) - "
                  f"too short to contain a shot - excluded.")
            continue

        df.insert(0, "video_name", video_name)
        df.insert(1, "shot_type", shot_type)

        all_data.append(df)
        video_names.add(video_name)

    if not all_data:
        print("No valid feature files found.")
        return

    dataset = pd.concat(all_data, ignore_index=True)

    # Safety: never treat the aggregate dataset itself as an input feature file.
    # It is already excluded by the glob above, but keep the check anyway.
    if dataset_file in feature_files:
        print("[WARN] Dataset file itself appeared as input; removed.")
        dataset_file.unlink()

    dataset.to_csv(dataset_file, index=False)

    print()
    print("===================================")
    print("RESEARCH DATASET CREATED")
    print("===================================")
    print(f"Videos included: {len(dataset)} (unique clips)")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Output: {dataset_file}")
    print()

    if duplicates:
        print(f"Duplicate clips excluded: {len(duplicates)}")
        for dup, canon in duplicates:
            print(f"    {dup} == {canon}")
        print()

    if unknown:
        print(f"[WARN] Videos with unresolved shot_type (set to 'unknown'): {unknown}")
        print()

    print("Shot distribution:")
    print(dataset["shot_type"].value_counts())
    print()

    print("Missing values per column (if any):")
    missing = dataset.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("    (none)")
    else:
        for col, count in missing.items():
            print(f"    {col}: {count}")
    print()
    print("Videos:")
    print(dataset[["video_name", "shot_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
