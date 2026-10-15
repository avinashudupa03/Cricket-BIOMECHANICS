"""Report generation for cricket biomechanics analysis.

Generates human-readable text reports and CSV summaries from processed
video data. Each report combines:

  * Biomechanics summary (from ``biomechanics_features.csv``)
  * Shot rating breakdown (from ``shot_rating.csv``)
  * Injury risk summary (from ``injury_risk_summary.csv``)
  * ML classification result (from ``shot_classification.json``)
  * Measurement provenance labels (from ``measurement_provenance.txt``)

Reports are written to ``reports/<video_name>/`` as:

  * ``summary.txt``       - human-readable plain-text report
  * ``summary.csv``       - single-row CSV with key metrics
  * ``factors.csv``       - per-factor rating breakdown

Usage:
    python report_generator.py <video_name> [video_name ...]

If no video names are given, a batch summary is generated across all
processed videos and written to ``reports/batch_summary.txt`` and
``reports/batch_summary.csv``.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DATA = BASE_DIR / "output_data"
REPORTS_DIR = BASE_DIR / "reports"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Optional[List[Dict]]:
    """Load a CSV file into a list of dicts, or None if missing/empty."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return None


def _load_json(path: Path) -> Optional[Dict]:
    """Load a JSON file, or None if missing/invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_text(path: Path) -> Optional[str]:
    """Load a text file, or None if missing."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _safe_float(value, default: float = 0.0) -> float:
    """Coerce a value to float, returning default on failure."""
    if value is None or value == "":
        return default
    try:
        f = float(value)
        if f != f:  # NaN check
            return default
        return f
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_video_report(video_name: str) -> Dict[str, str]:
    """Generate all reports for a single processed video.

    Returns a dict with keys ``summary_txt``, ``summary_csv``, ``factors_csv``
    mapping to the file paths that were written.
    """
    folder = OUTPUT_DATA / video_name
    report_dir = REPORTS_DIR / video_name
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all data sources ------------------------------------------
    features_rows = _load_csv(folder / "biomechanics_features.csv")
    rating_rows = _load_csv(folder / "shot_rating.csv")
    injury_rows = _load_csv(folder / "injury_risk.csv")
    injury_summary = _load_csv(folder / "injury_risk_summary.csv")
    classification = _load_json(folder / "shot_classification.json")
    provenance = _load_text(folder / "measurement_provenance.txt")

    features = features_rows[0] if features_rows else {}
    rating = rating_rows[0] if rating_rows else {}
    injury_sum = injury_summary[0] if injury_summary else {}

    # ---- Build summary.txt ----------------------------------------------
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 72)
    lines.append(f"CRICKET BIOMECHANICS ANALYSIS REPORT")
    lines.append(f"Video: {video_name}")
    lines.append(f"Generated: {now}")
    lines.append("=" * 72)
    lines.append("")

    # --- Shot rating section ---
    lines.append("-" * 72)
    lines.append("SHOT RATING")
    lines.append("-" * 72)
    if rating:
        lines.append(f"  Rating        : {_safe_float(rating.get('rating')):.1f} / 10")
        lines.append(f"  Classification: {rating.get('classification', 'N/A')}")
        lines.append(f"  Confidence    : {_safe_float(rating.get('confidence')):.0%}")
        lines.append(f"  Covered weight: {_safe_float(rating.get('covered_weight')):.0%}")

        strengths = rating.get("strengths", "")
        if strengths:
            lines.append(f"  Strengths     : {strengths}")
        weaknesses = rating.get("weaknesses", "")
        if weaknesses:
            lines.append(f"  Weaknesses    : {weaknesses}")

        explanation = rating.get("explanation", "")
        if explanation:
            lines.append("")
            lines.append(f"  Explanation: {explanation}")
    else:
        lines.append("  No rating data available.")
    lines.append("")

    # --- Factor breakdown ---
    lines.append("-" * 72)
    lines.append("FACTOR BREAKDOWN")
    lines.append("-" * 72)
    factor_codes = [
        "swing", "bat_angle", "timing", "backlift", "contact",
        "power", "follow", "technique", "balance", "footwork",
        "stride", "direction",
    ]
    factor_labels = {
        "swing": "Bat Swing & Bat Speed",
        "bat_angle": "Bat Angle at Contact",
        "timing": "Shot Timing",
        "backlift": "Bat Lift Control",
        "contact": "Quality of Ball Contact",
        "power": "Power Generated",
        "follow": "Follow-through & Technique",
        "technique": "Body Position & Stability",
        "balance": "Head Position & Balance",
        "footwork": "Footwork & Weight Transfer",
        "stride": "Stride & Forward Movement",
        "direction": "Shot Direction & Placement",
    }
    lines.append(f"  {'Factor':<30} {'Score':>6} {'Weight':>8} {'Weighted':>10}")
    lines.append(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*10}")
    for code in factor_codes:
        score_key = f"factor_{code}_score"
        weight_key = f"factor_{code}_weighted"
        score_val = rating.get(score_key, "")
        weight_val = rating.get(weight_key, "")
        score_str = f"{_safe_float(score_val):.1f}" if score_val else "N/A"
        weight_str = f"{_safe_float(weight_val):.2f}" if weight_val else "N/A"
        lines.append(f"  {factor_labels[code]:<30} {score_str:>6} {weight_str:>8} {'':>10}")
    lines.append("")

    # --- Biomechanics metrics ---
    lines.append("-" * 72)
    lines.append("KEY BIOMECHANICS METRICS")
    lines.append("-" * 72)
    key_metrics = [
        ("Impact frame", "impact_frame"),
        ("Impact time (ms)", "impact_time_ms"),
        ("Downswing duration (ms)", "downswing_duration_ms"),
        ("Follow-through duration (ms)", "followthrough_duration_ms"),
        ("Max movement score", "maximum_movement_score"),
        ("Avg movement score", "average_movement_score"),
        ("Bat speed (torso/s)", "bat_speed_torso_per_s"),
        ("Bat speed mean (torso/s)", "bat_speed_mean_torso_per_s"),
        ("Torso rotation range", "torso_rotation_range_swing"),
        ("Lead wrist height range", "lead_wrist_height_range_swing"),
        ("Ankle separation at impact", "ankle_separation_at_impact"),
        ("Stride length (torso)", "stride_length_torso"),
    ]
    for label, key in key_metrics:
        val = features.get(key, "")
        if val:
            lines.append(f"  {label:<35}: {_safe_float(val):.3f}")
        else:
            lines.append(f"  {label:<35}: N/A")
    lines.append("")

    # --- Phase distribution ---
    lines.append("-" * 72)
    lines.append("PHASE DISTRIBUTION")
    lines.append("-" * 72)
    phase_keys = [
        ("stance_frames", "Stance"),
        ("backlift_frames", "Backlift"),
        ("downswing_frames", "Downswing"),
        ("impact_frames", "Impact"),
        ("followthrough_frames", "Follow-through"),
    ]
    for key, label in phase_keys:
        val = features.get(key, "")
        if val:
            lines.append(f"  {label:<20}: {int(_safe_float(val))} frames")
    lines.append("")

    # --- Injury risk ---
    lines.append("-" * 72)
    lines.append("INJURY RISK SCREENING")
    lines.append("-" * 72)
    if injury_sum:
        safety = _safe_float(injury_sum.get("safety_score"), -1)
        risk_level = injury_sum.get("risk_level", "N/A")
        lines.append(f"  Safety score : {safety:.1f} / 10" if safety >= 0 else "  Safety score : N/A")
        lines.append(f"  Risk level   : {risk_level}")
    else:
        lines.append("  No injury risk data available.")

    if injury_rows:
        lines.append("")
        lines.append(f"  {'Metric':<30} {'Side':<8} {'Severity':<10} {'Frames':>6} {'Worst Phase':<15}")
        lines.append(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*6} {'-'*15}")
        for row in injury_rows:
            metric = row.get("metric", "N/A")
            side = row.get("side", "N/A")
            severity = row.get("severity", "N/A")
            frames = row.get("frames", "0")
            worst = row.get("worst_phase", "N/A")
            lines.append(f"  {metric:<30} {side:<8} {severity:<10} {frames:>6} {worst:<15}")
    lines.append("")

    # --- ML classification ---
    lines.append("-" * 72)
    lines.append("ML SHOT CLASSIFICATION")
    lines.append("-" * 72)
    if classification:
        lines.append(f"  Model      : {classification.get('model_name', 'N/A')}")
        lines.append(f"  Predicted  : {classification.get('predicted', 'N/A')}")
        conf = classification.get("confidence")
        conf_str = f"{conf:.3f}" if conf is not None else "N/A"
        lines.append(f"  Confidence : {conf_str} (threshold {classification.get('threshold', 'N/A')})")
        lines.append(f"  Final class: {classification.get('shot_type', 'N/A')}")
        lines.append(f"  Reason     : {classification.get('reason_label', 'N/A')}")
    else:
        lines.append("  No classification data available.")
    lines.append("")

    # --- Provenance ---
    lines.append("-" * 72)
    lines.append("MEASUREMENT PROVENANCE")
    lines.append("-" * 72)
    if provenance:
        lines.append(provenance.strip())
    else:
        lines.append("  No provenance data available.")
    lines.append("")

    # --- Disclaimer ---
    lines.append("=" * 72)
    lines.append("DISCLAIMER")
    lines.append("=" * 72)
    lines.append("This report is generated from single-camera 2D pose estimation")
    lines.append("and is NOT a clinical or professional biomechanical assessment.")
    lines.append("ML shot labels are model predictions on a small dataset and")
    lines.append("should not be treated as ground truth.")
    lines.append("")

    summary_txt = report_dir / "summary.txt"
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    # ---- Build summary.csv ----------------------------------------------
    summary_csv = report_dir / "summary.csv"
    summary_row = {
        "video_name": video_name,
        "rating": rating.get("rating", ""),
        "classification": rating.get("classification", ""),
        "confidence": rating.get("confidence", ""),
        "covered_weight": rating.get("covered_weight", ""),
        "impact_frame": features.get("impact_frame", ""),
        "impact_time_ms": features.get("impact_time_ms", ""),
        "downswing_duration_ms": features.get("downswing_duration_ms", ""),
        "followthrough_duration_ms": features.get("followthrough_duration_ms", ""),
        "maximum_movement_score": features.get("maximum_movement_score", ""),
        "bat_speed_torso_per_s": features.get("bat_speed_torso_per_s", ""),
        "safety_score": injury_sum.get("safety_score", ""),
        "risk_level": injury_sum.get("risk_level", ""),
        "ml_predicted": classification.get("predicted", "") if classification else "",
        "ml_confidence": classification.get("confidence", "") if classification else "",
        "ml_shot_type": classification.get("shot_type", "") if classification else "",
    }
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    # ---- Build factors.csv ----------------------------------------------
    factors_csv = report_dir / "factors.csv"
    with open(factors_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["factor_code", "factor_label", "score", "weighted", "comment"])
        for code in factor_codes:
            score_val = rating.get(f"factor_{code}_score", "")
            weight_val = rating.get(f"factor_{code}_weighted", "")
            # Comments aren't in the CSV, so we reconstruct a simple one
            if score_val:
                sv = _safe_float(score_val)
                if sv >= 8:
                    comment = "Strong"
                elif sv >= 6:
                    comment = "Adequate"
                elif sv >= 4:
                    comment = "Weak"
                else:
                    comment = "Poor"
            else:
                comment = "Unavailable"
            writer.writerow([
                code,
                factor_labels[code],
                score_val or "",
                weight_val or "",
                comment,
            ])

    return {
        "summary_txt": str(summary_txt),
        "summary_csv": str(summary_csv),
        "factors_csv": str(factors_csv),
    }


def generate_batch_summary() -> Dict[str, str]:
    """Generate a batch summary across all processed videos.

    Scans ``output_data/`` for subdirectories that contain a
    ``biomechanics_features.csv`` and writes aggregate statistics to
    ``reports/batch_summary.txt`` and ``reports/batch_summary.csv``.
    """
    if not OUTPUT_DATA.exists():
        return {}

    video_names = sorted([
        d.name for d in OUTPUT_DATA.iterdir()
        if d.is_dir() and (d / "biomechanics_features.csv").exists()
    ])

    if not video_names:
        return {}

    report_dir = REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-video data
    rows: List[Dict] = []
    for name in video_names:
        features_rows = _load_csv(OUTPUT_DATA / name / "biomechanics_features.csv")
        rating_rows = _load_csv(OUTPUT_DATA / name / "shot_rating.csv")
        injury_summary = _load_csv(OUTPUT_DATA / name / "injury_risk_summary.csv")
        classification = _load_json(OUTPUT_DATA / name / "shot_classification.json")

        features = features_rows[0] if features_rows else {}
        rating = rating_rows[0] if rating_rows else {}
        injury_sum = injury_summary[0] if injury_summary else {}

        rows.append({
            "video_name": name,
            "rating": _safe_float(rating.get("rating")),
            "classification": rating.get("classification", ""),
            "confidence": _safe_float(rating.get("confidence")),
            "impact_frame": _safe_float(features.get("impact_frame")),
            "downswing_ms": _safe_float(features.get("downswing_duration_ms")),
            "followthrough_ms": _safe_float(features.get("followthrough_duration_ms")),
            "max_movement": _safe_float(features.get("maximum_movement_score")),
            "bat_speed": _safe_float(features.get("bat_speed_torso_per_s")),
            "safety_score": _safe_float(injury_sum.get("safety_score")),
            "risk_level": injury_sum.get("risk_level", ""),
            "ml_predicted": classification.get("predicted", "") if classification else "",
            "ml_confidence": _safe_float(classification.get("confidence")) if classification else 0.0,
            "ml_shot_type": classification.get("shot_type", "") if classification else "",
        })

    # Compute aggregate stats
    ratings = [r["rating"] for r in rows if r["rating"] > 0]
    confidences = [r["confidence"] for r in rows if r["confidence"] > 0]
    bat_speeds = [r["bat_speed"] for r in rows if r["bat_speed"] > 0]
    safety_scores = [r["safety_score"] for r in rows if r["safety_score"] > 0]

    avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    avg_bat_speed = sum(bat_speeds) / len(bat_speeds) if bat_speeds else 0.0
    avg_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0.0

    # Shot type distribution
    shot_type_counts: Dict[str, int] = {}
    for r in rows:
        st = r.get("ml_shot_type", "") or r.get("ml_predicted", "") or "unknown"
        shot_type_counts[st] = shot_type_counts.get(st, 0) + 1

    # ---- Build batch_summary.txt ----------------------------------------
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 72)
    lines.append("BATCH ANALYSIS SUMMARY")
    lines.append(f"Generated: {now}")
    lines.append(f"Total videos: {len(rows)}")
    lines.append("=" * 72)
    lines.append("")

    lines.append("-" * 72)
    lines.append("AGGREGATE STATISTICS")
    lines.append("-" * 72)
    lines.append(f"  Average rating       : {avg_rating:.2f} / 10")
    lines.append(f"  Average confidence   : {avg_confidence:.0%}")
    lines.append(f"  Average bat speed    : {avg_bat_speed:.3f} torso/s")
    lines.append(f"  Average safety score : {avg_safety:.2f} / 10")
    lines.append("")

    lines.append("-" * 72)
    lines.append("SHOT TYPE DISTRIBUTION")
    lines.append("-" * 72)
    for st, count in sorted(shot_type_counts.items()):
        lines.append(f"  {st:<20}: {count}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("PER-VIDEO SUMMARY")
    lines.append("-" * 72)
    lines.append(f"  {'Video':<30} {'Rating':>6} {'Conf':>6} {'Bat Spd':>8} {'Safety':>7} {'Shot Type':<15}")
    lines.append(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*15}")
    for r in rows:
        lines.append(
            f"  {r['video_name']:<30} "
            f"{r['rating']:>6.1f} "
            f"{r['confidence']:>5.0%} "
            f"{r['bat_speed']:>8.3f} "
            f"{r['safety_score']:>7.1f} "
            f"{r['ml_shot_type']:<15}"
        )
    lines.append("")

    lines.append("=" * 72)
    lines.append("DISCLAIMER")
    lines.append("=" * 72)
    lines.append("Generated from single-camera 2D pose estimation. NOT a clinical")
    lines.append("or professional biomechanical assessment. ML labels are predictions")
    lines.append("on a small dataset and should not be treated as ground truth.")
    lines.append("")

    summary_txt = report_dir / "batch_summary.txt"
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    # ---- Build batch_summary.csv ----------------------------------------
    summary_csv = report_dir / "batch_summary.csv"
    if rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return {
        "summary_txt": str(summary_txt),
        "summary_csv": str(summary_csv),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage:")
        print("  python report_generator.py <video_name> [video_name ...]")
        print("  python report_generator.py --batch")
        print()
        print("Examples:")
        print("  python report_generator.py defence")
        print("  python report_generator.py defence drive flick")
        print("  python report_generator.py --batch")
        return 1

    if argv[0] == "--batch":
        result = generate_batch_summary()
        if result:
            print(f"Batch summary written to:")
            print(f"  {result['summary_txt']}")
            print(f"  {result['summary_csv']}")
        else:
            print("No processed videos found in output_data/")
        return 0

    # Generate individual reports
    for video_name in argv:
        print(f"Generating report for: {video_name}")
        try:
            paths = generate_video_report(video_name)
            print(f"  summary.txt  -> {paths['summary_txt']}")
            print(f"  summary.csv  -> {paths['summary_csv']}")
            print(f"  factors.csv  -> {paths['factors_csv']}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
        print()

    # Also regenerate batch summary
    print("Regenerating batch summary...")
    batch = generate_batch_summary()
    if batch:
        print(f"  {batch['summary_txt']}")
        print(f"  {batch['summary_csv']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
