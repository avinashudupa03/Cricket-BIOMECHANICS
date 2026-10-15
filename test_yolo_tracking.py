#!/usr/bin/env python
"""Test YOLO person tracking with ByteTrack on cricket video."""

import cv2
import csv
import sys
from pathlib import Path
from ultralytics import YOLO

def test_yolo_tracking(video_path, output_path=None, csv_path=None, conf_threshold=0.3, show=False):
    """Run YOLO person tracking with ByteTrack on a video."""
    
    print(f"Loading YOLO model...")
    model = YOLO('yolo11n.pt')
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup output video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output video: {output_path}")
    
    # Setup CSV logging
    csv_writer = None
    csv_file = None
    if csv_path:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'center_x', 'center_y'])
        print(f"Tracking log: {csv_path}")
    
    # COCO class 0 = person
    person_class_id = 0
    
    frame_count = 0
    track_stats = {}  # track_id -> {frames: count, first_frame, last_frame, centers: []}
    
    print("\nRunning tracking with ByteTrack...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO tracking with ByteTrack (custom config)
        # persist=True keeps track IDs across frames
        results = model.track(
            frame, 
            conf=conf_threshold, 
            classes=[person_class_id], 
            tracker="bytetrack_custom.yaml",
            persist=True,
            verbose=False
        )
        
        # Extract tracking results
        detections = results[0].boxes
        tracked_count = 0
        
        if detections is not None and detections.id is not None:
            track_ids = detections.id.cpu().numpy().astype(int)
            boxes = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                tracked_count += 1
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = float(confs[i])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw bounding box with track ID
                color = get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID label
                label = f"ID:{track_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 4, color, -1)
                
                # Log to CSV
                if csv_writer:
                    csv_writer.writerow([frame_count, track_id, x1, y1, x2, y2, conf, center_x, center_y])
                
                # Update track stats
                if track_id not in track_stats:
                    track_stats[track_id] = {
                        'frames': 0,
                        'first_frame': frame_count,
                        'last_frame': frame_count,
                        'centers': []
                    }
                track_stats[track_id]['frames'] += 1
                track_stats[track_id]['last_frame'] = frame_count
                track_stats[track_id]['centers'].append((center_x, center_y))
        
        # Add frame info overlay
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracked persons: {tracked_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Active IDs: {len(track_stats)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if writer:
            writer.write(frame)
        
        if show:
            cv2.imshow('YOLO ByteTrack Person Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Progress
        if frame_count % 30 == 0:
            active_ids = sorted(track_stats.keys())
            print(f"  Frame {frame_count}/{total_frames} - Tracked: {tracked_count} - Active IDs: {active_ids}")
    
    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("TRACKING SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique track IDs: {len(track_stats)}")
    print()
    print("Track ID stability:")
    for track_id in sorted(track_stats.keys()):
        stats = track_stats[track_id]
        duration = stats['last_frame'] - stats['first_frame'] + 1
        coverage = stats['frames'] / duration * 100 if duration > 0 else 0
        print(f"  ID {track_id}: frames={stats['frames']}, "
              f"span={stats['first_frame']}-{stats['last_frame']} "
              f"({duration} frames), coverage={coverage:.1f}%")
    
    # Check for ID switches (large jumps in center position)
    print("\nChecking for potential ID switches (center jumps > 100px):")
    for track_id in sorted(track_stats.keys()):
        centers = track_stats[track_id]['centers']
        if len(centers) < 2:
            continue
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            dist = (dx**2 + dy**2)**0.5
            if dist > 100:
                print(f"  WARNING: ID {track_id} frame {i} -> {i+1}: "
                      f"jump={dist:.1f}px (from {centers[i-1]} to {centers[i]})")
    
    if output_path:
        print(f"\nOutput video saved to: {output_path}")
    if csv_path:
        print(f"Tracking CSV saved to: {csv_path}")

def get_color(track_id):
    """Generate consistent color for a track ID."""
    import hashlib
    hash_val = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    r = (hash_val & 0xFF)
    g = ((hash_val >> 8) & 0xFF)
    b = ((hash_val >> 16) & 0xFF)
    # Ensure colors are visible (not too dark)
    return (max(r, 100), max(g, 100), max(b, 100))

if __name__ == "__main__":
    video_path = Path("input_videos/defence/fast_left_0000036.avi")
    output_path = Path("output_videos/defence_bytetrack_test.mp4")
    csv_path = Path("output_videos/defence_bytetrack_log.csv")
    
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_yolo_tracking(video_path, output_path, csv_path, conf_threshold=0.3, show=False)