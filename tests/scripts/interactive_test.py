#!/usr/bin/env python3
"""
Interactive Test Script for DAM4SAM (Refined)
---------------------------------------------
Features:
1. Interactive BBox Input: Add objects mid-sequence using CLI.
2. Enhanced Helper Frame: Saves 'helper_view.jpg' with a COORDINATE GRID.
3. JSON Logging: Saves user inputs to 'interaction_log.json'.
4. Visualization: Draws IDs, BBoxes, and Masks with distinct colors.
5. Video Generation: Converts results to MP4 at the end.

Usage:
    python interactive_test.py --images_dir n1 --output_dir output_test
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

# Add project root to path for imports
# tests/interactive_test.py -> tests/ -> project_root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking_wrapper_mot import DAM4SAMMOT

def get_distinct_colors(n=20):
    """Generate distinct colors for visualization"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        colors.append(tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])))
    return colors

def draw_grid(img, step=100):
    """Draw a coordinate grid on the image"""
    h, w = img.shape[:2]
    vis = img.copy()
    
    # Draw vertical lines
    for x in range(0, w, step):
        cv2.line(vis, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(vis, str(x), (x+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    # Draw horizontal lines
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(vis, str(y), (2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    return vis

def draw_results(img_cv, masks, obj_ids, colors):
    """Draw masks and IDs on the image"""
    vis = img_cv.copy()
    for i, mask in enumerate(masks):
        if mask is not None and np.sum(mask) > 0:
            obj_id = obj_ids[i]
            color = colors[obj_id % len(colors)]
            
            # Draw mask contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
            
            # Draw ID
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Text background
                text = f"ID:{obj_id}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (cx, cy - h - 5), (cx + w, cy + 5), color, -1)
                cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis

def create_video(images_dir, output_path, fps=10):
    """Create video from images using OpenCV"""
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
    parser = argparse.ArgumentParser(description="Interactive Test Script")
    
    # Default paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_images_dir = os.path.join(project_root, 'n1')
    default_checkpoint_dir = os.path.join(project_root, 'checkpoints')
    
    parser.add_argument('--images_dir', type=str, default=default_images_dir, help='Input image directory')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Output directory')
    parser.add_argument('--model_size', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--checkpoint_dir', type=str, default=default_checkpoint_dir)
    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    frames_out_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_out_dir, exist_ok=True)

    # Load images
    image_files = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))
    if not image_files:
        print(f"No images found in {args.images_dir}")
        return

    # Initialize Tracker
    print("Initializing Tracker...")
    tracker = DAM4SAMMOT(model_size=args.model_size, checkpoint_dir=args.checkpoint_dir)
    
    colors = get_distinct_colors()
    interaction_log = []
    obj_counter = 0
    
    print("\n" + "="*50)
    print(" INTERACTIVE TRACKING TEST (REFINED)")
    print("="*50)
    print("Controls:")
    print("  [Enter] : Process next frame")
    print("  'a'     : Add new object (BBox)")
    print("  'q'     : Quit")
    print("="*50 + "\n")

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        img_pil = Image.open(img_path).convert('RGB')
        img_cv = cv2.imread(img_path)
        
        # --- Interaction Loop ---
        while True:
            print(f"\rFrame {i}/{len(image_files)-1} [{filename}] | Tracked: {obj_counter} objects > ", end="")
            user_input = input().strip().lower()
            
            if user_input == 'q':
                print("\nQuitting...")
                return
            
            elif user_input == 'a':
                # Save helper frame with GRID
                helper_img = draw_grid(img_cv, step=50)
                helper_path = "helper_view.jpg"
                cv2.imwrite(helper_path, helper_img)
                print(f"\n[Helper] Saved '{helper_path}' with COORDINATE GRID. Open it to find coordinates.")
                
                try:
                    bbox_input = input("Enter BBox (x y w h) or 'c' to cancel: ")
                    if bbox_input.lower() == 'c':
                        print("Cancelled.")
                        continue

                    bbox = list(map(int, bbox_input.split()))
                    if len(bbox) != 4:
                        print("Error: Invalid format. Expected 4 integers.")
                        continue
                        
                    # Add object
                    obj_id = f"obj_{obj_counter+1}"
                    
                    if obj_counter == 0:
                        tracker.initialize(img_pil, [{'obj_id': obj_id, 'bbox': bbox}])
                    else:
                        tracker.add_object(img_pil, {'bbox': bbox})
                    
                    # Log
                    log_entry = {
                        "frame": i,
                        "filename": filename,
                        "action": "add_object",
                        "obj_id": obj_id,
                        "bbox": bbox
                    }
                    interaction_log.append(log_entry)
                    print(f"Added {obj_id} at {bbox}")
                    obj_counter += 1
                    
                    # Update img_cv with new bbox for immediate feedback
                    x, y, w, h = bbox
                    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                except ValueError:
                    print("Error: Please enter integers.")
                except Exception as e:
                    print(f"Error adding object: {e}")
            
            else:
                # Proceed to tracking
                break
        
        # --- Tracking ---
        if obj_counter > 0:
            outputs = tracker.track(img_pil)
            masks = outputs['masks']
            # Map tracker outputs to IDs
            current_ids = list(range(1, len(masks) + 1)) 
            vis_img = draw_results(img_cv, masks, current_ids, colors)
        else:
            vis_img = img_cv

        # Save frame
        out_path = os.path.join(frames_out_dir, filename)
        cv2.imwrite(out_path, vis_img)

    # --- Post Processing ---
    print("\n" + "="*50)
    print(" TEST COMPLETE")
    print("="*50)
    
    # Save Log
    log_path = os.path.join(args.output_dir, "interaction_log.json")
    with open(log_path, 'w') as f:
        json.dump(interaction_log, f, indent=4)
    print(f"Saved interaction log to {log_path}")
    
    # Generate Video
    video_path = os.path.join(args.output_dir, "result.mp4")
    create_video(frames_out_dir, video_path)
    print(f"Saved video to {video_path}")

if __name__ == "__main__":
    main()
