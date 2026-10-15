#!/usr/bin/env python
"""Test YOLO tracking + Batsman Selection on cricket video."""

import cv2
import csv
import sys
from pathlib import Path
from ultralytics import YOLO

from batsman_selector import (
    BatsmanSelector, PersonTrack, create_person_track_from_yolo,
    BATSMAN_LOCK_FRAMES, BATSMAN_SWITCH_THRESHOLD, BATSMAN_LOST_TOLERANCE,
    POSITION_WEIGHT, SIZE_WEIGHT, POSE_WEIGHT, TEMPORAL_WEIGHT, MOVEMENT_WEIGHT
)

# Color generation for track IDs
import hashlib
def get_color(track_id):
    hash_val = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    r = max((hash_val & 0xFF), 100)
    g = max(((hash_val >> 8) & 0xFF), 100)
    b = max(((hash_val >> 16) & 0xFF), 100)
    return (b, g, r)  # BGR for OpenCV

def test_batsman_selection(video_path, output_path=None, csv_path=None, conf_threshold=0.3, show=False):
    """Run YOLO tracking + batsman selection on a video."""
    
    print(f"Loading YOLO model...")
    model = YOLO('yolo11n.pt')
    
    print(f"Initializing Batsman Selector...")
    selector = BatsmanSelector()
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cv2.CAP_PROP_FRAME_HEIGHT)
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
        csv_writer.writerow([
            'frame', 'selected_batsman_id', 'selection_confidence', 'selection_reason',
            'track_id', 'track_score', 'pos_score', 'size_score', 'pose_score', 
            'temporal_score', 'movement_score',
            'x1', 'y1', 'x2', 'y2', 'conf', 'center_x', 'center_y'
        ])
        print(f"Selection log: {csv_path}")
    
    # COCO class 0 = person
    person_class_id = 0
    
    frame_count = 0
    track_stats = {}  # track_id -> {frames: count, first_frame, last_frame}
    selection_log = []
    
    print("\nRunning YOLO tracking + Batsman Selection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO tracking with ByteTrack
        results = model.track(
            frame, 
            conf=conf_threshold, 
            classes=[person_class_id], 
            tracker="bytetrack_custom.yaml",
            persist=True,
            verbose=False
        )
        
        # Build tracks dict for selector
        tracks = {}
        detections = results[0].boxes
        
        if detections is not None and detections.id is not None:
            track_ids = detections.id.cpu().numpy().astype(int)
            boxes = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = float(confs[i])
                
                track = create_person_track_from_yolo(track_id, (x1, y1, x2, y2), conf, frame.shape)
                tracks[track_id] = track
                
                # Update track stats
                if track_id not in track_stats:
                    track_stats[track_id] = {
                        'frames': 0,
                        'first_frame': frame_count,
                        'last_frame': frame_count,
                    }
                track_stats[track_id]['frames'] += 1
                track_stats[track_id]['last_frame'] = frame_count
        
        # Run batsman selection
        selection = selector.update(tracks, frame_count)
        
        # Draw all tracks
        if detections is not None and detections.id is not None:
            track_ids = detections.id.cpu().numpy().astype(int)
            boxes = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            
            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = float(confs[i])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                color = get_color(track_id)
                is_batsman = (track_id == selection.selected_track_id)
                
                # Draw bounding box
                thickness = 4 if is_batsman else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Get score
                track_score = selection.all_scores.get(track_id, 0.0)
                
                # Draw track ID and score
                label = f"ID:{track_id} S:{track_score:.2f}"
                if is_batsman:
                    label = "★ " + label + " ★"
                
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 4, color, -1)
                
                # Log to CSV
                if csv_writer:
                    # Find track in selector history for detailed scores
                    track_obj = selector.track_history.get(track_id)
                    if track_obj:
                        csv_writer.writerow([
                            frame_count, selection.selected_track_id, selection.confidence, selection.reason,
                            track_id, track_score,
                            track_obj.position_score, track_obj.size_score, track_obj.pose_score,
                            track_obj.temporal_score, track_obj.movement_score,
                            x1, y1, x2, y2, conf, center_x, center_y
                        ])
                    else:
                        csv_writer.writerow([
                            frame_count, selection.selected_track_id, selection.confidence, selection.reason,
                            track_id, track_score, 0, 0, 0, 0, 0,
                            x1, y1, x2, y2, conf, center_x, center_y
                        ])
        
        # Draw selection info panel
        panel_h = 120
        cv2.rectangle(frame, (0, 0), (width, panel_h), (10, 10, 20), -1)
        cv2.line(frame, (0, panel_h), (width, panel_h), (40, 44, 60), 1)
        
        # Title
        cv2.putText(frame, f"Batsman Selection | Frame: {frame_count}/{total_frames}", (16, 26),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Selected batsman info
        if selection.selected_track_id is not None:
            sel_color = get_color(selection.selected_track_id)
            cv2.putText(frame, f"SELECTED BATSMAN: ID {selection.selected_track_id}", (16, 54),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, sel_color, 2)
            cv2.putText(frame, f"Confidence: {selection.confidence:.3f} | Reason: {selection.reason}", (16, 82),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Lock frames: {selector.lock_frames}/{BATSMAN_LOCK_FRAMES} | Switch thresh: {BATSMAN_SWITCH_THRESHOLD}", (16, 104),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            cv2.putText(frame, "NO BATSMAN SELECTED (below threshold)", (16, 54),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            cv2.putText(frame, f"Best score: {max(selection.all_scores.values()) if selection.all_scores else 0:.3f} (threshold: {0.5})", (16, 82),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Score bar chart (bottom of panel)
        if selection.all_scores:
            bar_y = panel_h + 30
            bar_h = 12
            bar_w_max = 300
            x_start = 16
            y_start = bar_y
            
            # Sort for display
            sorted_scores = sorted(selection.all_scores.items(), key=lambda x: x[1], reverse=True)[:8]
            for i, (tid, score) in enumerate(sorted_scores):
                y = y_start + i * (bar_h + 4)
                color = get_color(tid)
                is_sel = (tid == selection.selected_track_id)
                
                # Background bar
                cv2.rectangle(frame, (x_start, y), (x_start + bar_w_max, y + bar_h), (40, 40, 50), -1)
                # Score bar
                fill_w = int(bar_w_max * score)
                cv2.rectangle(frame, (x_start, y), (x_start + fill_w, y + bar_h), color, -1)
                # Label
                label = f"ID{tid}: {score:.3f}" + (" ★" if is_sel else "")
                cv2.putText(frame, label, (x_start + 4, y + bar_h - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255) if is_sel else (200, 200, 200), 1)
        
        if writer:
            writer.write(frame)
        
        if show:
            cv2.imshow('YOLO + Batsman Selection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Progress
        if frame_count % 30 == 0:
            sel_str = f"ID {selection.selected_track_id}" if selection.selected_track_id else "None"
            active_ids = sorted([t for t in tracks.keys()])
            print(f"  Frame {frame_count}/{total_frames} - Batsman: {sel_str} (conf={selection.confidence:.3f}) - Active: {active_ids}")
        
        selection_log.append(selection)
    
    cap.release()
    if writer:
        writer.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("BATSMAN SELECTION SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique track IDs: {len(track_stats)}")
    
    # Selection stability
    print("\nSelection per frame:")
    prev_id = None
    switch_count = 0
    for i, sel in enumerate(selection_log):
        curr_id = sel.selected_track_id
        if curr_id != prev_id:
            if prev_id is not None:
                switch_count += 1
            print(f"  Frame {i+1:3d}: ID {curr_id} (conf={sel.confidence:.3f}) - {sel.reason}")
            prev_id = curr_id
    print(f"\nTotal batsman ID switches: {switch_count}")
    
    # Track stability
    print("\nTrack ID stability:")
    for track_id in sorted(track_stats.keys()):
        stats = track_stats[track_id]
        duration = stats['last_frame'] - stats['first_frame'] + 1
        coverage = stats['frames'] / duration * 100 if duration > 0 else 0
        was_batsman = any(s.selected_track_id == track_id for s in selection_log)
        batsman_frames = sum(1 for s in selection_log if s.selected_track_id == track_id)
        marker = " ★BATSMAN★" if was_batsman else ""
        print(f"  ID {track_id}: frames={stats['frames']}, "
              f"span={stats['first_frame']}-{stats['last_frame']} "
              f"({duration} frames), coverage={coverage:.1f}%{marker}"
              f" - Selected as batsman for {batsman_frames} frames")
    
    # Score analysis
    if selection_log:
        print("\nFinal score breakdown (last frame):")
        last_sel = selection_log[-1]
        for tid, score in sorted(last_sel.all_scores.items(), key=lambda x: x[1], reverse=True):
            track_obj = selector.track_history.get(tid)
            if track_obj:
                print(f"  ID {tid}: total={score:.3f} "
                      f"(pos={track_obj.position_score:.2f}, "
                      f"size={track_obj.size_score:.2f}, "
                      f"pose={track_obj.pose_score:.2f}, "
                      f"temp={track_obj.temporal_score:.2f}, "
                      f"move={track_obj.movement_score:.2f})")
            else:
                print(f"  ID {tid}: total={score:.3f}")
    
    if output_path:
        print(f"\nOutput video saved to: {output_path}")
    if csv_path:
        print(f"Selection CSV saved to: {csv_path}")

if __name__ == "__main__":
    video_path = Path("input_videos/defence/fast_left_0000036.avi")
    output_path = Path("output_videos/defence_batsman_selection_test.mp4")
    csv_path = Path("output_videos/defence_batsman_selection_log.csv")
    
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_batsman_selection(video_path, output_path, csv_path, conf_threshold=0.3, show=False)