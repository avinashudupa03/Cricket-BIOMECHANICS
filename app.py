
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_data"
UPLOAD_DIR = BASE_DIR / "input_videos"
MODELS_DIR = BASE_DIR / "models"

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg'}

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

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
    
    for key, filename in result_files.items():
        filepath = result_dir / filename
        if filepath.exists():
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                results[key] = df.to_dict(orient='records')
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
    
    shot_type = request.form.get('shot_type', 'unknown')
    
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
    
    shot_type = data.get('shot_type', 'unknown')
    
    try:
        from main import process_video
        result = process_video(str(video_path), shot_type)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
