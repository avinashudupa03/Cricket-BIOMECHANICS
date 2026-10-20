
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
import sys
import uuid
import threading
import time

from database import (
    init_db,
    create_pipeline_job,
    update_pipeline_job,
    get_pipeline_job,
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_data"
UPLOAD_DIR = BASE_DIR / "input_videos"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
init_db()

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg'}

# Supported shot types
SHOT_TYPES = ['defence', 'drive', 'cut', 'flick', 'pull shot', 'lofted']

PIPELINE_STAGES = [
    "Uploaded video",
    "Detecting player pose and extracting landmarks",
    "Calculating joint angles, batting phases, features and ratings",
    "Rendering annotated video",
]


def run_pipeline_job(job_id, video_path, video_name, shot_type):
    """Execute the existing analysis pipeline in the background."""
    import subprocess

    try:
        update_pipeline_job(
            job_id,
            status="running",
            current=PIPELINE_STAGES[1],
            steps=[PIPELINE_STAGES[0]]
        )

        command = [
            sys.executable,
            str(BASE_DIR / "process_video.py"),
            str(video_path),
            shot_type,
            job_id
        ]

        result = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            errors="replace"
        )

        if result.stdout:
            app.logger.info("Pipeline output:\n%s", result.stdout)

        if result.returncode != 0:
            details = result.stderr or result.stdout
            raise RuntimeError(
                (details or f"Pipeline exited with code {result.returncode}")[-4000:]
            )

        update_pipeline_job(
            job_id,
            status="done",
            current="Analysis complete",
            steps=PIPELINE_STAGES,
            done=True,
            ok=True,
            error=None,
            finished_at=time.time()
        )

    except Exception as exc:
        app.logger.exception("Pipeline job %s failed", job_id)

        update_pipeline_job(
            job_id,
            status="done",
            current="Analysis failed",
            done=True,
            ok=False,
            error=str(exc),
            finished_at=time.time()
        )


@app.route('/progress/<job_id>')
def progress_page(job_id):
    """Display progress for a specific analysis job."""
    job = get_pipeline_job(job_id)

    if job is None:
        return "Analysis job not found.", 404

    return render_template(
        'progress.html',
        job_id=job_id,
        video_name=job['video_name'],
        shot_type_label=job['shot_type'].replace('_', ' ').title(),
        job_status=0 if job['done'] else 1
    )


@app.route('/progress_status/<job_id>')
def progress_status(job_id):
    """Return the latest status of a background analysis."""
    job = get_pipeline_job(job_id)

    if job is None:
        return jsonify({"state": "missing"}), 404

    return jsonify({
        "state": "done" if job["done"] else "running",
        "ok": job["ok"],
        "current": job["current"],
        "steps": job["steps"],
        "error": job["error"],
        "video_name": job["video_name"]
    })

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_dataset_stats():
    """Get statistics about the dataset from output_data."""
    stats = {
        'total_videos': 0,
        'class_names': [],
        'n_classes': 0,
        'n_features': 0,
        'distribution': {},
        'dataset_exists': False
    }
    
    if OUTPUT_DIR.exists():
        video_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
        stats['total_videos'] = len(video_dirs)
        stats['dataset_exists'] = stats['total_videos'] > 0
        
        # Count by shot type (from classification JSON)
        for vdir in video_dirs:
            class_file = vdir / 'shot_classification.json'
            shot_type = 'unknown'
            if class_file.exists():
                try:
                    with open(class_file) as f:
                        data = json.load(f)
                        shot_type = data.get('shot_type', SHOT_TYPES[0])
                except:
                    pass
            stats['distribution'][shot_type] = stats['distribution'].get(shot_type, 0) + 1
        
        stats['class_names'] = list(stats['distribution'].keys())
        stats['n_classes'] = len(stats['class_names'])
    
    # Features count from biomechanics_features.csv
    if stats['total_videos'] > 0:
        first_video = list(OUTPUT_DIR.iterdir())[0]
        feat_file = first_video / 'biomechanics_features.csv'
        if feat_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(feat_file)
                stats['n_features'] = len(df.columns)
            except:
                stats['n_features'] = 0
    
    return stats


def get_ml_insights():
    """Get ML insights from reports."""
    insights = {
        'available': False,
        'dataset_size': None,
        'n_classes': None,
        'best_model': None,
        'evaluation': None,
        'accuracy': None,
        'comparison_table': None
    }
    
    # Check for model comparison report
    comparison_file = REPORTS_DIR / 'model_comparison.csv'
    if comparison_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(comparison_file)
            if not df.empty:
                insights['available'] = True
                insights['dataset_size'] = df.get('n_samples', [None])[0] if 'n_samples' in df.columns else None
                insights['n_classes'] = df.get('n_classes', [None])[0] if 'n_classes' in df.columns else None
                insights['best_model'] = df.iloc[0].get('model', 'Unknown') if len(df) > 0 else None
                insights['evaluation'] = 'LOOCV'
                insights['accuracy'] = df.iloc[0].get('accuracy', None) if len(df) > 0 else None
                insights['comparison_table'] = df.to_dict(orient='records')
        except:
            pass
    
    return insights


def get_recent_videos(limit=10):
    """Get recently processed videos."""
    videos = []
    if OUTPUT_DIR.exists():
        for vdir in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not vdir.is_dir():
                continue
            
            video_info = {
                'name': vdir.name,
                'results_url': url_for('get_results', video_name=vdir.name),
                'analysis_exists': False,
                'rating': None,
                'shot_type': None,
                'frames': None,
                'processed_at': datetime.fromtimestamp(vdir.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            }
            
            # Check for analysis video
            analysis_video = f"{vdir.name}_analysis.mp4"
            video_path = BASE_DIR / "output_videos" / analysis_video
            video_info['analysis_exists'] = video_path.exists()
            
            # Get rating
            rating_file = vdir / 'shot_rating.csv'
            if rating_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(rating_file)
                    if not df.empty and 'rating' in df.columns:
                        video_info['rating'] = float(df['rating'].iloc[0])
                except:
                    pass
            
            # Get shot type
            class_file = vdir / 'shot_classification.json'
            if class_file.exists():
                try:
                    with open(class_file) as f:
                        data = json.load(f)
                        video_info['shot_type'] = data.get('shot_type', 'unknown')
                except:
                    pass
            
            # Get frame count
            landmarks_file = vdir / 'batting_landmarks.csv'
            if landmarks_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(landmarks_file)
                    video_info['frames'] = len(df)
                except:
                    pass
            
            videos.append(video_info)
            
            if len(videos) >= limit:
                break
    
    return videos


def get_model_files():
    """Get model report files."""
    files = {}
    report_files = {
        'confusion_matrix.png': 'confusion_matrix.png',
        'feature_importance.png': 'feature_importance.png',
        'feature_importance.csv': 'feature_importance.csv',
        'classification_report.txt': 'classification_report.txt',
        'model_results_summary.txt': 'model_results_summary.txt'
    }
    
    for key, filename in report_files.items():
        filepath = REPORTS_DIR / filename
        if filepath.exists():
            if filename.endswith('.png'):
                files[key] = True
            else:
                try:
                    with open(filepath) as f:
                        files[key] = f.read()
                except:
                    files[key] = ''
        else:
            files[key] = None
    
    return files


def get_eda_images():
    """Get EDA images."""
    eda_dir = REPORTS_DIR / 'eda'
    images = []
    if eda_dir.exists():
        for img in sorted(eda_dir.glob('*.png')):
            images.append(img.name)
    return images


@app.context_processor
def inject_globals():
    """Inject global template variables."""
    stats = get_dataset_stats()
    return {
        'dataset_count': stats['total_videos'],
        'stats': stats,
        'ml_insights': get_ml_insights(),
        'recent_videos': get_recent_videos(10),
        'model_files': get_model_files(),
        'eda_images': get_eda_images(),
        'shot_types': SHOT_TYPES
    }

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

@app.route('/api/videos')
def list_videos():
    """List available input videos."""
    videos = []
    if UPLOAD_DIR.exists():
        for video_file in UPLOAD_DIR.rglob('*'):
            if video_file.suffix.lower() in ALLOWED_EXTENSIONS:
                rel_path = video_file.relative_to(UPLOAD_DIR)
                videos.append({
                    'name': video_file.name,
                    'path': str(rel_path),
                    'size': video_file.stat().st_size
                })
    return jsonify({'videos': videos})

@app.route('/api/results/<video_name>')
def get_results(video_name):
    """Get analysis results for a video."""
    video_name = Path(video_name).stem
    result_dir = OUTPUT_DIR / video_name
    
    if not result_dir.exists():
        return jsonify({'error': 'Results not found'}), 404
    
    results = {}
    
    # Load all result files
    result_files = {
        'landmarks': 'batting_landmarks.csv',
        'angles': 'batting_angles.csv',
        'phases': 'batting_phases.csv',
        'features': 'biomechanics_features.csv',
        'rating': 'shot_rating.csv',
        'injury_risk': 'injury_risk.csv',
        'injury_summary': 'injury_risk_summary.csv',
        'tracking_debug': 'tracking_debug.csv',
    }
    
    import math

    def clean_nan(obj):
        """Recursively replace NaN floats with None for JSON serialization."""
        if isinstance(obj, float):
            return None if math.isnan(obj) else obj
        elif isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        return obj

    for key, filename in result_files.items():
        filepath = result_dir / filename
        if filepath.exists():
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                results[key] = clean_nan(df.to_dict(orient='records'))
            except Exception as e:
                results[key] = {'error': str(e)}
    
    # Load shot classification
    class_file = result_dir / 'shot_classification.json'
    if class_file.exists():
        with open(class_file) as f:
            results['classification'] = json.load(f)
    
    # Check for analysis video
    analysis_video = f"{video_name}_analysis.mp4"
    video_path = BASE_DIR / "output_videos" / analysis_video
    if video_path.exists():
        results['analysis_video'] = f"/output_videos/{analysis_video}"
    
    return jsonify(results)

@app.route('/output_videos/<path:filename>')
def serve_video(filename):
    """Serve analysis videos."""
    return send_from_directory(BASE_DIR / "output_videos", filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Upload and analyze a new video."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    shot_type = request.form.get('shot_type', SHOT_TYPES[0])
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    video_path = UPLOAD_DIR / filename
    file.save(str(video_path))
    
    # Run analysis pipeline
    try:
        from main import process_video
        result = process_video(str(video_path), shot_type)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/local', methods=['POST'])
def analyze_local():
    """Analyze a video already in the input_videos folder."""
    data = request.get_json()
    if not data or 'video_path' not in data:
        return jsonify({'error': 'video_path required'}), 400
    
    video_path = Path(data['video_path'])
    if not video_path.is_absolute():
        video_path = UPLOAD_DIR / video_path
    
    if not video_path.exists():
        return jsonify({'error': 'Video not found'}), 404
    
    shot_type = data.get('shot_type', SHOT_TYPES[0])
    
    try:
        from main import process_video
        result = process_video(str(video_path), shot_type)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============= WEB UI ROUTES =============

@app.route('/dashboard')
def dashboard():
    """Dashboard page with analytics overview."""
    stats = get_dataset_stats()
    ml_insights = get_ml_insights()
    recent_videos = get_recent_videos(10)
    model_files = get_model_files()
    eda_images = get_eda_images()
    return render_template('dashboard.html', 
                           stats=stats, 
                           ml_insights=ml_insights,
                           recent_videos=recent_videos,
                           model_files=model_files,
                           eda_images=eda_images)


@app.route('/upload')
def upload_form():
    """Upload video form page."""
    return render_template('upload.html', shot_types=SHOT_TYPES)



@app.route('/upload', methods=['POST'])
def upload_submit():
    """Save uploaded video and start background analysis."""

    if 'video' not in request.files:
        return render_template(
            'upload.html',
            shot_types=SHOT_TYPES,
            error='No video file provided'
        )

    file = request.files['video']

    if not file.filename:
        return render_template(
            'upload.html',
            shot_types=SHOT_TYPES,
            error='No file selected'
        )

    if not allowed_file(file.filename):
        return render_template(
            'upload.html',
            shot_types=SHOT_TYPES,
            error='Invalid file type. Use MP4, AVI, MOV, MPG, or MKV'
        )

    shot_type = request.form.get('shot_type', SHOT_TYPES[0])

    if shot_type not in SHOT_TYPES:
        shot_type = SHOT_TYPES[0]

    filename = secure_filename(file.filename)

    if not filename:
        return render_template(
            'upload.html',
            shot_types=SHOT_TYPES,
            error='Invalid filename'
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    video_path = UPLOAD_DIR / filename
    video_name = Path(filename).stem
    job_id = str(uuid.uuid4())

    try:
        # Save the uploaded video.
        file.save(str(video_path))

        # Register the analysis job in the database.
        create_pipeline_job(
            job_id=job_id,
            video_path=str(video_path),
            video_name=video_name,
            shot_type=shot_type
        )

        # Run analysis in the background.
        worker = threading.Thread(
            target=run_pipeline_job,
            args=(job_id, str(video_path), video_name, shot_type),
            daemon=True
        )
        worker.start()

        # Open the progress page.
        return redirect(url_for('progress_page', job_id=job_id))

    except Exception as exc:
        app.logger.exception('Failed to start analysis')

        return render_template(
            'upload.html',
            shot_types=SHOT_TYPES,
            error=f'Could not start analysis: {exc}'
        ), 500


@app.route('/history')

def history():
    """Analysis history page."""
    stats = get_dataset_stats()
    recent_videos = get_recent_videos(100)  # All videos
    total = len(recent_videos)
    rated_count = sum(1 for v in recent_videos if v['rating'] is not None)
    return render_template('history.html', 
                           total=total, 
                           rated_count=rated_count, 
                           videos=recent_videos)


@app.route('/batch_process', methods=['POST'])
def batch_process():
    """Trigger batch processing of all videos."""
    try:
        from process_all import main as process_all_main
        import subprocess
        import sys
        # Run in background
        subprocess.Popen([sys.executable, 'process_all.py', '--force'], cwd=BASE_DIR)
        return redirect(url_for('dashboard'))
    except Exception as e:
        return redirect(url_for('dashboard'))


@app.route('/report_file/<path:filename>')
def report_file(filename):
    """Serve report files."""
    return send_from_directory(REPORTS_DIR, filename)


@app.route('/results/<video_name>')
def results_page(video_name):
    """Full results page for a video."""
    video_name = Path(video_name).stem
    result_dir = OUTPUT_DIR / video_name
    
    if not result_dir.exists():
        return render_template('results.html', error='Results not found', video_name=video_name)
    
    # Load all result files
    results = {}
    result_files = {
        'landmarks': 'batting_landmarks.csv',
        'angles': 'batting_angles.csv',
        'phases': 'batting_phases.csv',
        'features': 'biomechanics_features.csv',
        'rating': 'shot_rating.csv',
        'injury_risk': 'injury_risk.csv',
        'injury_summary': 'injury_risk_summary.csv',
        'tracking_debug': 'tracking_debug.csv',
    }
    
    import math

    def clean_nan(obj):
        """Recursively replace NaN floats with None for JSON serialization."""
        if isinstance(obj, float):
            return None if math.isnan(obj) else obj
        elif isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        return obj
    
    for key, filename in result_files.items():
        filepath = result_dir / filename
        if filepath.exists():
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                results[key] = clean_nan(df.to_dict(orient='records'))
            except Exception as e:
                results[key] = {'error': str(e)}
    
    # Load shot classification
    classification = {}
    class_file = result_dir / 'shot_classification.json'
    if class_file.exists():
        with open(class_file) as f:
            classification = json.load(f)
    
    shot_type = classification.get('shot_type', 'unknown')
    
    # Check for analysis video
    analysis_video = f"{video_name}_analysis.mp4"
    video_path = BASE_DIR / "output_videos" / analysis_video
    has_analysis = video_path.exists()
    
    # Load charts
    charts = {}
    for chart_name in ['joint_angles_chart.png', 'phase_chart.png', 'movement_chart.png']:
        chart_path = result_dir / chart_name
        if chart_path.exists():
            charts[chart_name.replace('_chart.png', '').replace('joint_angles_', '')] = chart_name
    
    # Prepare data structure for template
    # Rating data
    rating = None
    if results.get('rating'):
        r = results['rating'][0] if isinstance(results['rating'], list) and results['rating'] else {}
        if r and 'rating' in r:
            # Split pipe-separated strengths/weaknesses from CSV
            strengths = r.get('strengths', [])
            if isinstance(strengths, str):
                strengths = [s.strip() for s in strengths.split('|') if s.strip()]
            weaknesses = r.get('weaknesses', [])
            if isinstance(weaknesses, str):
                weaknesses = [w.strip() for w in weaknesses.split('|') if w.strip()]
            
            rating = {
                'rating': r.get('rating', 0),
                'classification': r.get('classification', 'Unknown'),
                'confidence': r.get('confidence', 0),
                'covered_weight': r.get('covered_weight', 0),
                'unavailable_metrics': r.get('unavailable_metrics', []),
                'explanation': r.get('explanation', ''),
                'strengths': strengths,
                'weaknesses': weaknesses,
                'factors': r.get('factors', [])
            }
    
    # Metrics data
    metrics = {}
    if results.get('features'):
        m = results['features'][0] if isinstance(results['features'], list) and results['features'] else {}
        if m:
            metrics = {
                'impact_frame': m.get('impact_frame'),
                'impact_time_ms': m.get('impact_time_ms'),
                'downswing_duration_ms': m.get('downswing_duration_ms'),
                'followthrough_duration_ms': m.get('followthrough_duration_ms'),
                'maximum_movement_score': m.get('maximum_movement_score'),
                'average_movement_score': m.get('average_movement_score'),
                'total_frames': m.get('total_frames'),
                'fps': m.get('fps')
            }
    
    # Phase timeline
    phase_timeline = None
    if results.get('phases'):
        phases_df = results['phases']
        if isinstance(phases_df, list) and phases_df:
            # Find phase segments
            segments = []
            current_phase = None
            start_frame = 0
            total_frames = len(phases_df)
            
            for i, row in enumerate(phases_df):
                phase = row.get('phase', 'Unknown')
                if current_phase is None:
                    current_phase = phase
                    start_frame = i
                elif phase != current_phase:
                    segments.append({
                        'phase': current_phase,
                        'start': start_frame,
                        'end': i - 1
                    })
                    current_phase = phase
                    start_frame = i
            
            # Add last segment
            if current_phase is not None:
                segments.append({
                    'phase': current_phase,
                    'start': start_frame,
                    'end': total_frames - 1
                })
            
            # Find impact frame
            impact_frame = None
            for i, row in enumerate(phases_df):
                if row.get('phase') == 'Impact':
                    impact_frame = i
                    break
            
            phase_timeline = {
                'total_frames': total_frames,
                'segments': segments,
                'impact_frame': impact_frame
            }
    
    # Impact angles
    import math
    impact_angles = {}
    impact_angle_columns = []
    impact_angle_labels = {}
    if results.get('angles'):
        angles_df = results['angles']
        if isinstance(angles_df, list) and angles_df:
            # Find impact frame row - prefer first impact frame with valid data
            # Use phase_timeline's impact_frame if available, else metrics
            impact_idx = phase_timeline.get('impact_frame') if phase_timeline else None
            if impact_idx is None:
                impact_idx = metrics.get('impact_frame', 0) if metrics else 0
            
            # If the impact frame has all NaN angles, search nearby frames for valid data
            def has_valid_angles(row):
                for col, val in row.items():
                    if col not in ['frame', 'timestamp'] and val is not None:
                        try:
                            if isinstance(val, float) and math.isnan(val):
                                continue
                        except (TypeError, ValueError):
                            pass
                        return True
                return False
            
            # Check impact frame and nearby frames for valid angles
            if impact_idx is not None and impact_idx < len(angles_df):
                if not has_valid_angles(angles_df[impact_idx]):
                    # Search forward and backward for a frame with valid angles
                    for offset in range(1, 10):
                        # Check backward first (earlier frames)
                        check_idx = impact_idx - offset
                        if check_idx >= 0 and has_valid_angles(angles_df[check_idx]):
                            impact_idx = check_idx
                            break
                        # Check forward
                        check_idx = impact_idx + offset
                        if check_idx < len(angles_df) and has_valid_angles(angles_df[check_idx]):
                            impact_idx = check_idx
                            break
            
            if impact_idx is not None and impact_idx < len(angles_df):
                impact_row = angles_df[impact_idx]
                for col, val in impact_row.items():
                    if col not in ['frame', 'timestamp'] and val is not None:
                        # Skip NaN values
                        try:
                            if isinstance(val, float) and math.isnan(val):
                                continue
                        except (TypeError, ValueError):
                            pass
                        impact_angles[col] = val
                        impact_angle_columns.append(col)
                        # Create label
                        label = col.replace('_', ' ').title()
                        if 'left' in col.lower():
                            label = 'Left ' + label.replace('Left ', '')
                        elif 'right' in col.lower():
                            label = 'Right ' + label.replace('Right ', '')
                        impact_angle_labels[col] = label
    
    # Injury risk
    injury = {'available': False}
    if results.get('injury_summary'):
        injury_summary = results['injury_summary'][0] if isinstance(results['injury_summary'], list) and results['injury_summary'] else {}
        if injury_summary:
            injury = {
                'available': True,
                'summary': {
                    'safety_score': injury_summary.get('safety_score', 0),
                    'risk_level': injury_summary.get('risk_level', 'Low'),
                    'coverage': injury_summary.get('pose_coverage', 0),
                    'summary': injury_summary.get('summary', ''),
                    'top_concern': injury_summary.get('top_concern', '')
                },
                'flags': results.get('injury_risk', []) if isinstance(results.get('injury_risk'), list) else []
            }
    
    # Features dict
    features = {}
    if results.get('features'):
        f = results['features'][0] if isinstance(results['features'], list) and results['features'] else {}
        features = f
    
    # Plot name for joint angles chart
    plot_name = charts.get('joint_angles', None)
    
    # Recent videos for sidebar
    recent_videos = get_recent_videos(5)
    
    # ML insights
    ml_insights = get_ml_insights()
    
    return render_template('results.html', 
                           video_name=video_name,
                           data={
                               'video_name': video_name,
                               'frames': metrics.get('total_frames'),
                               'rating': rating,
                               'metrics': metrics,
                               'impact_angles': impact_angles,
                               'features': features
                           },
                           shot_type=shot_type,
                           has_analysis=has_analysis,
                           charts=charts,
                           phase_timeline=phase_timeline,
                           impact_angle_columns=impact_angle_columns,
                           impact_angle_labels=impact_angle_labels,
                           injury=injury,
                           ml_insights=ml_insights,
                           recent_videos=recent_videos,
                           plot_name=plot_name)


@app.route('/output_data/<video_name>/<path:filename>')
def serve_output_file(video_name, filename):
    """Serve output data files (charts, etc)."""
    return send_from_directory(OUTPUT_DIR / video_name, filename)


@app.route('/output_video/<video_name>', endpoint='output_video_file')
def serve_output_video(video_name):
    """Serve the annotated analysis video."""
    video_name = Path(video_name).stem
    analysis_video = f"{video_name}_analysis.mp4"
    video_path = BASE_DIR / "output_videos" / analysis_video

    if not video_path.is_file():
        return jsonify({
            "ok": False,
            "error": "Annotated video not found",
            "expected_path": str(video_path)
        }), 404

    return send_from_directory(
        BASE_DIR / "output_videos",
        analysis_video,
        mimetype="video/mp4"
    )


@app.route('/generate_analysis/<video_name>', methods=['POST'])
def generate_analysis(video_name):
    """Regenerate analysis video."""
    video_name = Path(video_name).stem
    try:
        from output_video import create_analysis_video
        video_path = BASE_DIR / "output_videos" / f"{video_name}_analysis.mp4"
        create_analysis_video(str(OUTPUT_DIR / video_name), str(video_path))
        return redirect(url_for('results_page', video_name=video_name))
    except Exception as e:
        return redirect(url_for('results_page', video_name=video_name))


@app.route('/result_image/<video_name>/<path:filename>')
def result_image(video_name, filename):
    """Serve result images (charts)."""
    video_name = Path(video_name).stem
    return send_from_directory(OUTPUT_DIR / video_name, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
