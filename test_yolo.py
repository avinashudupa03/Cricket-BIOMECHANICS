#!/usr/bin/env python
"""Test YOLO person detection on cricket video."""

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

def test_yolo_on_video(video_path, output_path=None, conf_threshold=0.3, show=False):
    """Run YOLO person detection on a video."""
    
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
    
    # Setup output video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Output video: {output_path}")
    
    # COCO class 0 = person
    person_class_id = 0
    
    frame_count = 0
    person_counts = []
    
    print("\nRunning detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, conf=conf_threshold, classes=[person_class_id], verbose=False)
        
        # Extract detections
        detections = results[0].boxes
        person_count = 0
        
        if detections is not None:
            for box in detections:
                cls_id = int(box.cls[0])
                if cls_id == person_class_id:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        person_counts.append(person_count)
        
        # Add frame info overlay
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Persons detected: {person_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if writer:
            writer.write(frame)
        
        if show:
            cv2.imshow('YOLO Person Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Progress
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames} - Persons: {person_count}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    print(f"Total frames processed: {frame_count}")
    print(f"Person count stats:")
    if person_counts:
        print(f"  Min: {min(person_counts)}")
        print(f"  Max: {max(person_counts)}")
        print(f"  Avg: {sum(person_counts)/len(person_counts):.2f}")
        print(f"  Frames with 0 persons: {person_counts.count(0)}")
        print(f"  Frames with 1 person:  {person_counts.count(1)}")
        print(f"  Frames with 2 persons: {person_counts.count(2)}")
        print(f"  Frames with 3+ persons: {sum(1 for c in person_counts if c >= 3)}")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")

if __name__ == "__main__":
    video_path = Path("input_videos/defence/fast_left_0000036.avi")
    output_path = Path("output_videos/defence_yolo_test.mp4")
    
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_yolo_on_video(video_path, output_path, conf_threshold=0.3, show=False)