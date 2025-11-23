"""
Create a clean tracking video by running SAM2 inference again 
and overlaying masks directly on original frames using OpenCV
"""

import sys
import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

def create_clean_tracking_video(
    video_frames_dir,
    sam_checkpoint,
    sam_config,
    output_video_path,
    yolo_model="yolo11n.pt",
    yolo_interval=5,
    fps=5,
    conf_threshold=0.5,
    target_classes=[0]  # person only
):
    """
    Run tracking and create video with masks overlaid on original frames
    
    Args:
        video_frames_dir: Directory with video frames
        sam_checkpoint: SAM2 checkpoint path
        sam_config: SAM2 config file
        output_video_path: Output video path
        yolo_model: YOLO model name
        yolo_interval: Run YOLO every N frames
        fps: Output video FPS
        conf_threshold: YOLO confidence threshold
        target_classes: List of class IDs to detect
    """
    print("="*80)
    print("Creating Clean Tracking Video")
    print("="*80)
    print()
    
    # Initialize YOLO
    print("Loading YOLO model...")
    yolo = YOLO(yolo_model)
    print(f"✓ YOLO loaded: {yolo_model}\n")
    
    # Initialize SAM2
    print("Loading SAM2 model...")
    predictor = build_sam2_video_predictor(sam_config, sam_checkpoint)
    inference_state = predictor.init_state(video_path=video_frames_dir)
    print("✓ SAM2 loaded\n")
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
    num_frames = len(frame_files)
    print(f"Total frames: {num_frames}\n")
    
    # Get video dimensions from first frame
    first_frame = cv2.imread(os.path.join(video_frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Color palette for different objects
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    current_obj_id = 1
    
    # Helper functions
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def mask_to_box(mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None
        return [np.min(x_indices), np.min(y_indices), 
                np.max(x_indices), np.max(y_indices)]
    
    # Process each frame
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{num_frames}")
        
        # Load original frame
        img_path = os.path.join(video_frames_dir, frame_files[frame_idx])
        frame = cv2.imread(img_path)
        
        current_masks = []
        current_obj_ids = []
        
        # Propagate SAM2 tracking
        if inference_state["obj_ids"]:
            try:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=1
                ):
                    if out_frame_idx == frame_idx:
                        current_obj_ids = out_obj_ids
                        for i in range(len(out_obj_ids)):
                            mask = (out_mask_logits[i].cpu().numpy() > 0.0).squeeze()
                            current_masks.append(mask)
            except Exception as e:
                pass  # Skip errors silently
        
        # Run YOLO detection periodically
        if frame_idx % yolo_interval == 0:
            results = yolo(img_path, conf=conf_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    if target_classes is None or class_id in target_classes:
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Find best matching tracked object
                        best_iou = 0
                        matched_obj_id = None
                        
                        for mask_idx, mask in enumerate(current_masks):
                            tracked_box = mask_to_box(mask)
                            if tracked_box is not None:
                                iou = compute_iou(bbox, tracked_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    matched_obj_id = current_obj_ids[mask_idx]
                        
                        # Add or refine object
                        if best_iou > 0.7:
                            # Refine existing object
                            try:
                                box_array = np.array(bbox, dtype=np.float32)
                                predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx,
                                    obj_id=matched_obj_id,
                                    box=box_array,
                                )
                            except:
                                pass
                        elif best_iou <= 0.3:
                            # Add new object
                            try:
                                box_array = np.array(bbox, dtype=np.float32)
                                predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx,
                                    obj_id=current_obj_id,
                                    box=box_array,
                                )
                                current_obj_id += 1
                            except:
                                pass
        
        # Draw masks on frame
        overlay = frame.copy()
        for i, mask in enumerate(current_masks):
            color = colors[i % len(colors)]
            # Create colored mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask] = color
            # Blend with overlay
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
            
            # Draw bounding box
            bbox = mask_to_box(mask)
            if bbox:
                cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), color, 2)
                # Add object ID label
                cv2.putText(overlay, f"ID:{current_obj_ids[i]}", 
                          (int(bbox[0]), int(bbox[1])-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Write frame to video
        out.write(overlay)
    
    out.release()
    
    # Print summary
    file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"✓ Clean tracking video created!")
    print(f"{'='*80}")
    print(f"File: {output_video_path}")
    print(f"Frames: {num_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Total objects tracked: {current_obj_id - 1}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    create_clean_tracking_video(
        video_frames_dir="n1_frames",
        sam_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        sam_config="sam21pp_hiera_l.yaml",
        output_video_path="n1_tracking_clean.mp4",
        yolo_model="yolo11n.pt",
        yolo_interval=5,
        fps=5,
        conf_threshold=0.5,
        target_classes=[0]  # person only
    )
