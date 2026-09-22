# Cricket Biomechanics AI

Automated biomechanical analysis of cricket batting swings using human pose estimation and machine learning. The system detects batting phases, tracks joint angles over time, and classifies shot types from raw video.

## Features

- **Pose estimation** from video frames via MediaPipe/Ultralytics-powered landmark extraction.
- **Joint angle tracking** for elbows, knees, shoulders, and hips across the whole swing.
- **Batting phase detection** — Stance, Backlift, Downswing, Impact, and Follow-through.
- **Biomechanics metrics** — impact frame/time, downswing & follow-through durations, and movement scores.
- **Shot rating (0–10)** — a deterministic, individually weighted rating of shot quality with a confidence score; missing metrics are reported, never invented.
- **Shot classification (ML)** using engineered features with Leave-One-Out Cross-Validation.
- **Interactive web dashboard (Flask)** to upload videos, view results, charts, and ML insights.
- **Report generation** — editable & model comparison outputs, confusion matrix, feature importance.

## Supported Shot Types

- Cut
- Defence
- Drive
- Flick

## Project Structure

```
cricket-biomechanics-ai/
├── app.py                  # Flask web app (upload, results, dashboard, history)
├── run_server.py           # Start the web server
├── config.py               # Path config
├── main.py                 # Entry point for the per-video pipeline
├── process_video.py        # Runs the full processing pipeline for one video
├── process_all.py          # Batch-process all input videos
│
├── pose_extractor.py       # MediaPipe landmark extraction
├── landmark_extractor.py   # Landmark handling utilities
├── angle_calculator.py     # Joint angle computation
├── phase_detector.py       # Batting phase segmentation
├── biomechanics_analyzer.py# Metrics + per-frame movement scores
├── finalize_analysis.py    # Runs angles + phases + features + rating in ONE
│                           # process (saves ~4x pandas startup overhead)
├── shot_rating.py          # Reusable 0-10 shot rating engine
├── shot_rater.py           # CLI for shot_rating.py
├── output_video.py         # Renders annotated analysis video
├── plot_angles.py          # Angle time-series plots
│
├── video_preprocess.py      # Geometry-preserving CLAHE frame enhancement
├── provenance.py            # Scientific-validity labeling of every output
├── data_quality.py          # Dataset validation checks
├── robustness_report.py     # Honest capability audit (tested vs untested)
│
├── feature_engineering.py  # Builds ML feature matrix
├── build_dataset.py        # Assembles dataset from videos
├── prepare_ml_data.py      # Cleans + prepares ML data
├── ml_model.py             # Model definitions
├── train_models.py         # Train classifiers (LOOCV)
├── evaluate_models.py      # Model evaluation + reports
├── feature_importance.py   # Feature importance analysis
├── eda.py                  # Exploratory data analysis
├── data_validation.py      # Dataset validation
├── smoke_test.py           # Headless self-test of the pipeline
│
├── report_generator.py     # Text/CSV report generation
├── templates/              # Flask HTML templates
├── static/                 # Static web assets
│
├── input_videos/           # Uploaded/source videos + input_videos/Misc analysis videos
├── output_data/            # Per-video CSVs + charts
├── reports/                # Eda + model reports
├── models/                 # Saved/trained models
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg (for video processing)

### Setup

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate    # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application

```bash
python run_server.py
# or
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Upload a batting clip, choose a shot type, and view the biomechanics analysis, angle charts, and phase timeline.

### Process a single video (CLI)

```bash
python process_video.py input_videos\\defence.avi
```

**Speed:** pose detection runs on every 2nd frame by default (skipped frames
reuse the last detected pose). To trade a little temporal smoothness for more
speed, set the stride higher — `CBAI_POSE_STRIDE=3`, or `=1` for full per-frame
detection.

### Rate a processed shot (CLI)

```bash
python shot_rater.py defence --shot-type defence
python shot_rater.py drive1 --json   # machine-readable output
```

`shot_rating.py` exposes a pure `rate_shot(features, extra=None, shot_type=...)`
function that any caller (web app, batch pipeline, notebook) can reuse.

### Train the ML models

```bash
python build_dataset.py     # build dataset from input videos
python train_models.py      # train classifiers with LOOCV
python evaluate_models.py   # generate evaluation reports
```

### Batch processing

```bash
python process_all.py       # process every video in input_videos/
python process_all.py --force   # re-process videos that already have outputs
python process_all.py -n        # dry-run: list what would be processed
```

`process_all.py` is **incremental by default**: videos whose analysis outputs
already exist are skipped, so adding new clips under
`input_videos/<shot_type>/` and re-running only processes the new/retry ones.
No video names or counts are hardcoded anywhere - everything is discovered by
scanning `input_videos/`, and each video's shot type comes from its containing
folder. Videos are processed one per worker process (parallel, bounded memory),
never all loaded at once, and LOOCV / dataset rows are per-video, so frames
from one video can never span train and test.

### Adding new videos

```bash
# 1. Drop new clips into the right class folder:
#      input_videos/drive/myshot1.avi
#      input_videos/flick/myshot2.avi
# 2. Process (only the new ones run):
python process_all.py
# 3. Rebuild dataset + retrain + evaluate:
python build_dataset.py
python data_validation.py
python feature_engineering.py
python prepare_ml_data.py
python train_models.py
python evaluate_models.py
```

### Validation & quality

```bash
python data_quality.py      # check for duplicates, bad labels, short/poor clips
python robustness_report.py # honest audit: tested vs untested conditions
python provenance.py        # (re)label scientific validity of all outputs
```

### Reproducible full pipeline

```bash
python process_all.py                     # 1. pose + angles + phases + features
python build_dataset.py                   # 2. assemble dataset from processed videos
python data_validation.py                 # 3. clean/impute (logged, no label edits)
python eda.py                             # 4. exploratory analysis + plots
python feature_engineering.py             # 5. engineered ML features
python prepare_ml_data.py                 # 6. prepare feature matrix
python train_models.py                    # 7. LOOCV training (fixed seed 42)
python evaluate_models.py                 # 8. evaluation + reports
python feature_importance.py              # 9. feature importance report
python data_quality.py                    # 10. final data-quality pass
```
All ML stages use a fixed `RANDOM_SEED = 42`; the model comparison artifact is
`saved to reports/model_comparison.csv`; LOOCV predictions to
`output_data/ml_preprocessing/loo_predictions.pkl` for downstream auditing.

## Accuracy-critical design decisions

These choices are deliberate and documented in the code with the reasoning
that justifies them:

- **Single-camera 2D pose estimation** (`pose_extractor.py`) — MediaPipe
  returns frame-normalised (0..1) coordinates, so resolution and FPS do not
  change feature values. Each estimated landmark carries visibility/presence
  scores that downstream math **filters on** rather than trusting blindly
  (`MIN_VISIBILITY = 0.30`).
- **Two-pass identity-locked tracking** (`pose_extractor.py`) — candidates
  are linked into temporal trajectories and the batsman is selected by
  trajectory score, never by fresh detection per frame. This is what prevents
  identity switches to the keeper/non-striker in multi-person footage.
- **Geometry-preserving CLAHE enhancement** (`video_preprocess.py`) — improves
  dark/low-contrast clips *before* detection by changing only pixel
  intensities, never spatial layout, so angles/phases stay valid.
- **Conditional band re-detection** (`pose_extractor.detect_all`) — the second
  MediaPipe pass on the central crop runs only when the full-frame pass failed
  to find a clean full-body batsman, halving pose cost without changing
  tracking output.
- **SelectKBest inside the pipeline** (`train_models.py`) — feature selection
  is fit on the training fold only, so the held-out LOOCV sample never leaks
  into selection.
- **LOOCV, not a train/test split** (`train_models.py`) — with 9 videos a
  stratified split would leave almost no test data; LOOCV is the honest
  evaluation at this size.
- **No data augmentation, no fabricated labels** — augmentation could not be
  validated under LOOCV (augmented copies leak across folds) and would
  falsify the landmarks biomechanics math depends on.
- **Torso-normalised, per-second features** (`biomechanics_analyzer.py`) —
  distances are divided by torso length and velocities use wall-clock seconds
  so different body proportions and frame rates produce comparable features.

## Scientific validity

Single-camera pose estimation is not clinical measurement. Every output folder
contains `measurement_provenance.txt` classifying each value as MEASURED /
ESTIMATED_2D / ESTIMATED_3D / MODEL_PREDICTION / CONFIDENCE, and ML reports
carry a disclaimer that labels are predictions on a tiny dataset, not ground
truth. See `reports/scientific_validity.txt`. Do not use results for clinical
or professional biomechanical decisions.

### Exploratory / analysis scripts

```bash
python eda.py               # EDA + plots
python feature_importance.py# feature importance analysis
```

## Output

For each processed video, `output_data/<video_name>/` contains:

- `batting_landmarks.csv` — per-frame pose landmarks
- `batting_angles.csv` — per-frame joint angles
- `batting_phases.csv` — phase labels per frame
- `biomechanics_features.csv` — derived biomechanics metrics
- `shot_rating.csv` — per-shot 0-10 rating, factor breakdown, confidence, and commentary
- `joint_angles_chart.png`, `phase_chart.png`, `movement_chart.png` — result charts

Annotated analysis videos are written to `output_videos/` (never inside
`input_videos/`, which keeps freshly generated clips from being re-processed),
and ML/reporting artifacts to `reports/`.

## License

This project is for educational/research use.