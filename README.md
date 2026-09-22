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
├── feature_engineering.py  # Builds ML feature matrix
├── build_dataset.py        # Assembles dataset from videos
├── prepare_ml_data.py      # Cleans + prepares ML data
├── ml_model.py             # Model definitions
├── train_models.py         # Train classifiers (LOOCV)
├── evaluate_models.py      # Model evaluation + reports
├── feature_importance.py   # Feature importance analysis
├── eda.py                  # Exploratory data analysis
├── data_validation.py      # Dataset validation
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
```

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

Annotated analysis videos are written to `input_videos/Misc/` and ML/reporting artifacts to `reports/`.

## License

This project is for educational/research use.