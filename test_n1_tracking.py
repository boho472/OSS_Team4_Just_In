"""
Test script for running YOLO+SAM2 tracker on n1 video
"""

import sys
import os
from yolo_sam2_tracking import YOLOSAMTracker

def main():
    # Configuration for n1 video
    video_dir = "n1_frames"
    sam_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam_config = "sam21pp_hiera_l.yaml"
    output_dir = "output_n1_tracking"
    
    print("=" * 80)
    print("YOLOv11 + SAM2 Tracking on N1 Video")
    print("=" * 80)
    print()
    
    # Initialize tracker
    tracker = YOLOSAMTracker(
        video_dir=video_dir,
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        output_dir=output_dir,
        yolo_model="yolo11n.pt"  # Using nano model for speed
    )
    
    # Run tracking with automatic object detection
    # YOLO will run every 5 frames to detect and add new objects
    tracker.track_video(
        start_frame=0, 
        end_frame=135,  # Total 136 frames (0-135)
        yolo_interval=5  # Run YOLO every 5 frames
    )
    
    # Create output video
    tracker.create_video(fps=5)  # Match original video FPS
    
    print("=" * 80)
    print("✓ All tasks completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
