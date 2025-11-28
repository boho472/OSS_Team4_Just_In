#!/usr/bin/env python3
"""
YOLO + D4SM Tracking-by-Detection (TBD) Pipeline
-----------------------------------------------
Description:
    Simple TBD approach where YOLOv11 detects objects in each frame,
    and D4SM tracks them. Uses IoU-based matching to associate detections
    with existing tracks.

Pipeline:
    1. YOLOv11 detects objects in current frame
    2. Match detections to existing tracks using IoU
    3. Update existing tracks, add new detections as new tracks
    4. D4SM performs tracking

Usage:
    python yolo_d4sm_tbd.py --images_dir tests/data/n1 --output_dir output_tbd --yolo_model yolov11n.pt
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import glob
from PIL import Image
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tracking_wrapper_mot import DAM4SAMMOT

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics package not found.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x, y, w, h] format.
    
    Args:
        box1: [x, y, w, h]
        box2: [x, y, w, h]
    
    Returns:
        iou: float
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    """
    Match detections to existing tracks using IoU.
    
    Args:
        detections: list of detection dicts with 'bbox' key
        tracks: dict mapping track_id -> bbox [x, y, w, h]
        iou_threshold: minimum IoU for matching
    
    Returns:
        matches: list of (detection_idx, track_id) tuples
        unmatched_detections: list of detection indices
    """
    if len(tracks) == 0:
        return [], list(range(len(detections)))
    
    matches = []
    unmatched_detections = []
    matched_detections = set()
    
    # For each detection, find best matching track
    for det_idx, det in enumerate(detections):
        det_bbox = det['bbox']
        best_iou = 0
        best_track_id = None
        
        for track_id, track_bbox in tracks.items():
            iou = calculate_iou(det_bbox, track_bbox)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        
        if best_iou >= iou_threshold:
            matches.append((det_idx, best_track_id))
            matched_detections.add(det_idx)
        else:
            unmatched_detections.append(det_idx)
    
    return matches, unmatched_detections


def get_distinct_colors(n=50):
    """Generate distinct colors for visualization"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        colors.append(tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])))
    return colors


def draw_segmentation_only(img_cv, tracks_masks, track_ids, colors):
    """
    Draw only tracking segmentation masks (no detection boxes).
    
    Args:
        img_cv: OpenCV image
        tracks_masks: list of masks from D4SM tracker
        track_ids: list of track IDs
        colors: color palette
    
    Returns:
        vis: visualization image with segmentation only
    """
    vis = img_cv.copy()
    
    # Draw tracking masks
    if tracks_masks:
        for i, mask in enumerate(tracks_masks):
            if mask is not None and np.sum(mask) > 0:
                track_id = track_ids[i]
                color = colors[track_id % len(colors)]
                
                # Draw mask contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, color, 3)
                
                # Draw mask overlay (semi-transparent)
                mask_overlay = vis.copy()
                mask_overlay[mask > 0] = color
                cv2.addWeighted(mask_overlay, 0.4, vis, 0.6, 0, vis)
                
                # Draw track ID
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    text = f"ID:{track_id}"
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(vis, (cx - 5, cy - h - 10), (cx + w + 5, cy + 5), color, -1)
                    cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return vis


def draw_results_with_detections(img_cv, detections, tracks_masks, track_ids, colors):
    """
    Draw detection boxes AND tracking masks on image.
    
    Args:
        img_cv: OpenCV image
        detections: list of detection dicts with 'bbox' and 'class_name'
        tracks_masks: list of masks from D4SM tracker
        track_ids: list of track IDs
        colors: color palette
    
    Returns:
        vis: visualization image with both detections and segmentation
    """
    # Start with segmentation
    vis = draw_segmentation_only(img_cv, tracks_masks, track_ids, colors)
    
    # Add detection boxes
    for det in detections:
        x, y, w, h = det['bbox']
        class_name = det.get('class_name', 'obj')
        conf = det.get('confidence', 0)
        
        # Draw bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, y - label_h - 5), (x + label_w, y), (0, 255, 0), -1)
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return vis


def create_video(images_dir, output_path, fps=10):
    """Create video from images"""
    print(f"Generating video: {output_path}")
    images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not images:
        print("No images found for video generation.")
        return
    
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img_path in images:
        video.write(cv2.imread(img_path))
    
    video.release()
    print("Video generation complete.")


def main():
    parser = argparse.ArgumentParser(description="YOLO + D4SM TBD Pipeline")
    
    # Paths
    parser.add_argument('--images_dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output_dir', type=str, default='output_tbd', help='Output directory')
    
    # Model configs
    parser.add_argument('--yolo_model', type=str, default='yolo11n.pt', 
                       help='YOLO model path or name (e.g., yolo11n.pt, yolo11s.pt)')
    parser.add_argument('--d4sm_model_size', type=str, default='tiny', 
                       choices=['tiny', 'small', 'base', 'large'],
                       help='D4SM model size')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='D4SM checkpoint directory')
    
    # Detection and tracking params
    parser.add_argument('--det_conf', type=float, default=0.5, 
                       help='YOLO detection confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                       help='IoU threshold for matching detections to tracks')
    parser.add_argument('--classes', type=int, nargs='+', default=[0],
                       help='Filter by class IDs (default: [0] for person only, use --classes to override)')
    
    # Output options
    parser.add_argument('--save_video', action='store_true', help='Generate output video')
    parser.add_argument('--fps', type=int, default=10, help='Output video FPS')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    frames_out_dir = os.path.join(args.output_dir, "frames")
    frames_det_dir = os.path.join(args.output_dir, "frames_with_detections")
    os.makedirs(frames_out_dir, exist_ok=True)
    os.makedirs(frames_det_dir, exist_ok=True)
    
    # Load images
    image_files = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")) + 
                        glob.glob(os.path.join(args.images_dir, "*.png")))
    if not image_files:
        print(f"ERROR: No images found in {args.images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Initialize YOLO
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    # Initialize D4SM Tracker
    print(f"Initializing D4SM tracker (model_size={args.d4sm_model_size})")
    d4sm_tracker = DAM4SAMMOT(
        model_size=args.d4sm_model_size,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Tracking state
    tracks = {}  # track_id -> current bbox [x, y, w, h]
    next_track_id = 0
    colors = get_distinct_colors()
    tracking_log = []
    
    print("\n" + "="*60)
    print(" YOLO + D4SM TRACKING-BY-DETECTION")
    print("="*60)
    print(f"Detection confidence: {args.det_conf}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Classes filter: {args.classes if args.classes else 'All'}")
    print("="*60 + "\n")
    
    # Process each frame
    for frame_idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"Processing [{frame_idx+1}/{len(image_files)}] {filename}", end="")
        
        # Load image
        img_pil = Image.open(img_path).convert('RGB')
        img_cv = cv2.imread(img_path)
        
        # ========== YOLO Detection ==========
        results = yolo_model(img_path, conf=args.det_conf, classes=args.classes, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # YOLO format: xyxy
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # Convert to xywh
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': yolo_model.names[cls]
                })
        
        print(f" | Detections: {len(detections)}", end="")
        
        # ========== Match detections to tracks ==========
        matches, unmatched_detections = match_detections_to_tracks(
            detections, tracks, args.iou_threshold
        )
        
        # ========== Add new tracks for unmatched detections ==========
        new_tracks = []
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            
            if frame_idx == 0:
                # First frame: use initialize
                if next_track_id == 0:
                    d4sm_tracker.initialize(img_pil, [{'bbox': det['bbox']}])
                else:
                    track_id, mask = d4sm_tracker.add_object(img_pil, {'bbox': det['bbox']})
                    new_tracks.append(track_id)
            else:
                # Subsequent frames: use add_object
                track_id, mask = d4sm_tracker.add_object(img_pil, {'bbox': det['bbox']})
                new_tracks.append(track_id)
            
            tracks[next_track_id] = det['bbox']
            next_track_id += 1
        
        print(f" | New tracks: {len(unmatched_detections)}", end="")
        
        # ========== D4SM Tracking ==========
        if next_track_id > 0:
            if frame_idx > 0 or len(unmatched_detections) > 1:
                # Track on frames after first, or if multiple objects in first frame
                outputs = d4sm_tracker.track(img_pil)
                masks = outputs['masks']
            else:
                # First frame with single object
                masks = []
        else:
            masks = []
        
        # Update track bboxes from masks
        for i, mask in enumerate(masks):
            if mask is not None and np.sum(mask) > 0:
                # Get bbox from mask
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) > 0:
                    x, y = x_indices.min(), y_indices.min()
                    w = x_indices.max() - x + 1
                    h = y_indices.max() - y + 1
                    tracks[i] = [int(x), int(y), int(w), int(h)]
        
        print(f" | Active tracks: {len(tracks)}")
        
        # ========== Visualization ==========
        track_ids = list(range(len(masks)))
        
        # Create segmentation-only visualization (main output)
        vis_seg_only = draw_segmentation_only(img_cv, masks, track_ids, colors)
        info_text = f"Frame: {frame_idx+1}/{len(image_files)} | Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(vis_seg_only, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_seg_only, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Create visualization with detection boxes (debug output)
        vis_with_det = draw_results_with_detections(img_cv, detections, masks, track_ids, colors)
        cv2.putText(vis_with_det, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_with_det, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Save both versions
        out_path = os.path.join(frames_out_dir, filename)
        cv2.imwrite(out_path, vis_seg_only)
        
        out_path_det = os.path.join(frames_det_dir, filename)
        cv2.imwrite(out_path_det, vis_with_det)
        
        # Log
        frame_log = {
            'frame_idx': frame_idx,
            'filename': filename,
            'num_detections': len(detections),
            'num_tracks': len(tracks),
            'detections': detections,
            'new_tracks': new_tracks
        }
        tracking_log.append(frame_log)
    
    # ========== Post Processing ==========
    print("\n" + "="*60)
    print(" TRACKING COMPLETE")
    print("="*60)
    
    # Save tracking log
    log_path = os.path.join(args.output_dir, "tracking_log.json")
    with open(log_path, 'w') as f:
        json.dump(tracking_log, f, indent=2)
    print(f"Saved tracking log to {log_path}")
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    config = vars(args)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Generate video
    if args.save_video:
        video_path = os.path.join(args.output_dir, "result.mp4")
        create_video(frames_out_dir, video_path, fps=args.fps)
        print(f"Saved video to {video_path}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
