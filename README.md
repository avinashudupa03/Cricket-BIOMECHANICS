# Cricket Biomechanics AI

Automated biomechanical analysis of cricket batting swings using human pose estimation and machine learning. The system detects batting phases, tracks joint angles over time, and classifies shot types from raw video.

## Features

- **Pose estimation** from video frames via MediaPipe/Ultralytics-powered landmark extraction.
- **Joint angle tracking** for elbows, knees, shoulders, and hips across the whole swing.
- **Batting phase detection** — Stance, Backlift, Downswing, Impact, and Follow-through.
- **Biomechanics metrics** — impact frame/time, downswing & follow-through durations, and movement scores.
- **Shot rating (0–10)** — a deterministic, individually weighted rating of shot quality with a confidence score; missing metrics are reported, never invented.
- **Shot classification (ML)** using engineered features with Leave-One-Out Cross-Validation. A clip the model is not confident about is reported as **Unknown** rather than forced into a named shot.
- **React web dashboard** (Vite + React + TypeScript + Tailwind, in `web-main/`) to upload videos, view results, charts, shot prediction and ML insights.
- **Report generation** — editable & model comparison outputs, confusion matrix, feature importance.

## Supported Shot Types

| Class | Folder | Notes |
|---|---|---|
| `cut` | `input_videos/cut/` | |
| `defence` | `input_videos/defence/` | |
| `drive` | `input_videos/drive/` | |
| `flick` | `input_videos/flick/` | |
| `pull shot` | `input_videos/pull shot/` | Multi-word class; see note below |
| `unknown` | `input_videos/unknown/` | Catch-all for unclassifiable clips, **and** the classifier's low-confidence fallback |

The class list is defined once per consumer and must stay in sync:
`app.SHOT_TYPES` / `app.SHOT_TYPE_LABELS`, `data_validation.SHOT_TYPES`,
`ml_model.SUPPORTED_SHOT_TYPES`, and `SHOT_TYPES` / `SHOT_TYPE_LABELS` in
`web-main/web-main/src/types.ts`.

`pull shot` is a **multi-word** class, so `app.normalise_shot_type()` collapses
`"pull shot"`, `"pull_shot"`, `"Pull Shot"` and mixed case onto the single slug
that matches the folder name. Uploads can therefore never create a second,
parallel pull-shot folder.

The class vocabulary is **not** hard-coded into the model. `train_models.py`
derives it from the data (`class_names = list(label_encoder.classes_)`) and
`evaluate_models.py` threads `best["class_names"]` through the confusion matrix,
classification report and summaries. Adding labelled videos under a new
`input_videos/<class>/` folder and re-running the pipeline is all that is needed
to teach the classifier a new class.

### Unknown and the confidence gate

`ml_model.py` is the inference half of the classification system. It loads the
persisted winner from `train_models.py`
(`output_data/ml_preprocessing/best_model.pkl`), applies the same feature
engineering used in training, and reports the model's own posterior.

A clip is only named when the model is confident enough:

- `train_models.CONFIDENCE_THRESHOLD = 0.55` is the single source of truth for
  the gate — `ml_model.py` imports it rather than redefining it.
- Below the threshold, or if the top class is not a supported shot, the result
  is `unknown`. The discarded candidate is still reported as `predicted` with
  its confidence, so nothing is hidden.
- `reason` records why: `ok`, `low_confidence`, `model_unknown`,
  `unsupported_class`, `feature_error`, or `model_unavailable`.

An Unknown clip still carries its **full** analysis — pose, joint angles,
phases, movement scores, rating and injury screening are unaffected. Only the
shot *name* is withheld.

## What changed recently

The project has been updated with several reliability and workflow improvements to
make the pipeline safer and easier to use in practice:

- Added the `pull shot` class as a first-class supported shot type, with
  canonical normalization so uploads, model config and folder names all resolve
  to the same class.
- Fixed the input/output directory contract so generated analysis videos stay in
  `output_videos/` instead of being written into `input_videos/`, preventing
  reprocessing loops and upload pollution.
- Added temporary upload staging in `uploads/` with SHA-256 content-deduplication
  so identical clips are rejected before they can create duplicate jobs or dirty
  the dataset.
- Hardened the batch and upload scanners to ignore generated files such as
  `_analysis.mp4`, `_processed.mp4`, `_output.mp4`, `_labeled.mp4`, and similar
  suffixes. This keeps fresh pipeline outputs from being treated as new inputs.
- Improved pose-estimation robustness with geometry-preserving CLAHE preprocessing
  before landmark extraction, which helps on low-contrast or darker clips.
- Added a headless smoke test that validates directory topology, generated-file
  filters, upload deduplication, analysis-video fidelity and tracking metadata.
- Expanded the web workflow and dashboard/history tracking so processed videos are
  easier to review and monitor from the browser.
- Added quality metadata such as `tracking_ok` and `tracking_confidence` to
  generated tracking CSVs so downstream analysis can assess reliability more
  clearly.

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
├── ml_model.py             # Shot-classification INFERENCE (loads best_model.pkl,
│                           # gates on CONFIDENCE_THRESHOLD, emits Unknown)
├── train_models.py         # Train classifiers (LOOCV) + persist the winner
├── evaluate_models.py      # Model evaluation + reports
├── feature_importance.py   # Feature importance analysis
├── eda.py                  # Exploratory data analysis
├── data_validation.py      # Dataset validation
├── smoke_test.py           # Headless self-test of the pipeline
│
├── report_generator.py     # Text/CSV report generation
├── web-main/web-main/      # React + Vite + TS + Tailwind frontend (the UI)
│                           #   src/views/       Dashboard, Analyze, Results,
│                           #                   History, MLInsights, Reports
│                           #   src/components/  Sidebar, TopBar, ui primitives
│                           #   src/utils/api.ts the only place that calls /api
│                           #   dist/            built bundle Flask serves
├── templates/              # Legacy Jinja UI (fallback when dist/ is absent)
├── static/                 # Legacy static assets (same fallback)
│
├── input_videos/           # Source videos, one folder per shot class
├── output_data/            # Per-video CSVs + charts + classification JSON
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

The UI is the React app in `web-main/web-main`. Flask serves its production
bundle from `web-main/web-main/dist` and falls back to the legacy Jinja
templates only when that directory is missing.

**Build the frontend once** (requires Node 18+):

```bash
cd web-main/web-main
npm install
npm run build      # writes dist/ (index.html + hashed assets)
cd ../..
```

Then start the server:

```bash
python run_server.py
# or
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Upload a batting clip, choose a
shot type, and view the biomechanics analysis, joint angles, shot prediction
with confidence, angle charts and the phase timeline.

For frontend development with hot reload, run the Vite dev server alongside
Flask:

```bash
# terminal 1
python run_server.py

# terminal 2
cd web-main/web-main && npm run dev    # http://127.0.0.1:3000
```

> **Caveat** — `vite.config.ts` does not currently define a dev proxy, so
> `npm run dev` serves the UI but its `/api/*` calls are not forwarded to
> Flask on port 5000 and will fail. Until a `server.proxy` entry is added,
> use `npm run build` and drive the app from `http://127.0.0.1:5000`.

Other useful frontend scripts: `npm run lint` (`tsc --noEmit`),
`npm run preview` (serve the built bundle), `npm run clean`.

> **Note** — the UI imports only `react`, `react-dom` and `lucide-react`.
> `@google/genai`, `express`, `dotenv`, `motion`, `autoprefixer` and `tsx` are
> unused leftovers from the original template and pull in a very large
> dependency tree; if `npm install` is slow on a constrained network, install
> only what the build needs (`vite`, `@vitejs/plugin-react`, `@tailwindcss/vite`,
> `tailwindcss`, `typescript`, plus the three runtime deps) and build with that.

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

### Classify an already-processed shot (CLI)

```bash
python ml_model.py fast_left_00000031          # one or more video names
```

Loads the trained model, prints the predicted class, confidence, threshold and
the full posterior, and rewrites `shot_classification.json` for each clip. The
per-video pipeline calls this automatically via `finalize_analysis.py`, so you
only need it to re-classify existing clips or to inspect a decision.

```bash
python ml_model.py fast_left_00000031
# model=LogisticRegression classes=['cut','defence','drive','flick','pull shot','unknown'] threshold=0.55
# SHOT TYPE  : pull shot
# CONFIDENCE : 0.733 (threshold 0.55)
```

```bash
python ml_model.py fast_left_00000029
# SHOT TYPE  : unknown
# PREDICTED  : pull shot
# CONFIDENCE : 0.506 (threshold 0.55)
# REASON     : Confidence below threshold
```

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
# 1. Drop new clips into the right class folder (note the space in "pull shot"):
#      input_videos/drive/myshot1.avi
#      input_videos/pull shot/myshot2.avi
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

Each clip's label is its containing folder, so a new class only needs a new
folder — there is no registry to edit. Two consequences worth knowing:

- A clip whose source folder cannot be found is labelled `unknown` by
  `build_dataset.py`, so the `unknown` class grows on its own as legacy outputs
  age out of `input_videos/`.
- `build_dataset.py` **skips** clips with 0% pose-tracking coverage or fewer
  than 10 tracked frames (they would contribute only imputed noise). Very short
  or heavily occluded clips — common for `pull shot` — can be dropped this way.
  Watch for the `[SKIP]` lines and re-shoot rather than forcing them in.

Retraining is only required to change the model's behaviour. To re-score clips
that were already analysed, run `python ml_model.py <video_name>` instead.

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
- **LOOCV, not a train/test split** (`train_models.py`) — with a small dataset a
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

### Current model status — read this before trusting a prediction

The classifier is real but **weak**, and the numbers must not be read as
real-world accuracy. As of the last full pipeline run:

| Metric | Value |
|---|---|
| Training samples | 22 videos (after dedup + poor-quality filtering) |
| Classes | 6 |
| Best model | `LogisticRegression` (tied with SVM on macro F1, broken on mean confidence) |
| LOOCV accuracy | 40.9% |
| LOOCV macro F1 | 0.270 |
| Top-2 accuracy | 0.545 |
| Mean confidence | 0.665 |

Consequences you should expect to see in the UI:

- **Most clips are reported as Unknown.** With a 0.55 gate, roughly three
  quarters of analysed clips defer rather than naming a shot. That is the
  threshold doing its job on a weak model, not a broken feature — but if you
  want fewer Unknown results, lower `CONFIDENCE_THRESHOLD` in
  `train_models.py` (it is imported by `ml_model.py`, so one edit changes both
  training-time reporting and inference).
- **Some *correct* predictions are withheld.** `defence` currently tops out
  around 0.50, just under the gate, so those clips show Unknown.
- **The only way to genuinely improve this is more labelled videos.** A new
  class needs enough clips to be separable, not just a folder.

Feature importance values at this sample size are unstable and exploratory —
treat them as a rough ranking, not evidence.

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
- `injury_risk.csv` / `injury_risk_summary.csv` — loading flags and safety score
- `shot_classification.json` — the model's shot prediction: final class, the
  discarded candidate, confidence, threshold, the full 6-class posterior and
  the reason it was accepted or deferred to Unknown
- `measurement_provenance.txt` — MEASURED / ESTIMATED / MODEL_PREDICTION labels
- `joint_angles_chart.png`, `phase_chart.png`, `movement_chart.png` — result charts

Annotated analysis videos are written to `output_videos/` (never inside
`input_videos/`, which keeps freshly generated clips from being re-processed),
and ML/reporting artifacts to `reports/`.

## HTTP API

The frontend talks to Flask only through these JSON endpoints
(`web-main/web-main/src/utils/api.ts` is the single client):

| Endpoint | Returns |
|---|---|
| `GET /api/stats` | Dataset totals, class names + distribution, ML summary |
| `GET /api/videos` | Processed clips (newest first) with rating, classification |
| `GET /api/history` | Same list plus `total` / `rated_count` |
| `GET /api/results/<video>` | Full payload: metrics, joint angles, phases, charts, rating, injury, **classification** |
| `GET /api/ml` | LOOCV evaluation summary and model comparison table |
| `GET /api/reports` | EDA images, model artifacts, validation documents |
| `GET /api/progress/<job_id>` | Pipeline state, completed steps, current step |
| `POST /api/upload` | Multipart `video` + `shot_type`; returns `job_id` to poll |
| `GET /api/dataset.csv` | Engineered dataset download |

`/api/results` returns **both** the *filed* `shot_type` (the folder the clip was
uploaded into) and the *model* `classification`. The UI labels them
**Filed as** and **Model prediction** so the two are never confused.

Uploads are deduplicated by MD5 content hash: re-uploading a clip that is
already present in `input_videos/` returns `{"status": "existing"}` instead of
creating a second copy, and invalid shot types are rejected with HTTP 400.

## License

This project is for educational/research use.