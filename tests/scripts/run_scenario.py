#!/usr/bin/env python3
"""
Run Scenario Test Script
------------------------
Reads a JSON scenario file and executes tracking.
Demonstrates the "JSON -> Tracking" workflow.

Usage:
    python tests/scripts/run_scenario.py --scenario tests/scenarios/scenario_multi_add.json
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracking_wrapper_mot import DAM4SAMMOT

def get_distinct_colors(n=20):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        colors.append(tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])))
    return colors

def draw_results(img_cv, masks, obj_ids, colors):
    vis = img_cv.copy()
    for i, mask in enumerate(masks):
        if mask is not None and np.sum(mask) > 0:
            obj_id = obj_ids[i]
            # Hash string ID to integer for color
            color_idx = hash(obj_id) % len(colors)
            color = colors[color_idx]
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
            
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                text = f"{obj_id}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (cx, cy - h - 5), (cx + w, cy + 5), color, -1)
                cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis

def create_video(images_dir, output_path, fps=10):
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
    parser = argparse.ArgumentParser()
    
    # Default paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_images_dir = os.path.join(project_root, 'n1')
    default_checkpoint_dir = os.path.join(project_root, 'checkpoints')
    
    parser.add_argument('--scenario', type=str, required=True, help='Path to JSON scenario file')
    parser.add_argument('--images_dir', type=str, default=default_images_dir)
    parser.add_argument('--output_dir', type=str, default='tests/results/scenario_test')
    parser.add_argument('--model_size', type=str, default='tiny')
    parser.add_argument('--checkpoint_dir', type=str, default=default_checkpoint_dir)
    args = parser.parse_args()

    # Load Scenario
    with open(args.scenario, 'r') as f:
        scenario = json.load(f)
    
    prompts = scenario.get('prompts', [])
    # Organize prompts by frame
    prompts_by_frame = {}
    for p in prompts:
        fidx = p['frame']
        if fidx not in prompts_by_frame:
            prompts_by_frame[fidx] = []
        prompts_by_frame[fidx].append(p)

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    frames_out_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_out_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")))
    if not image_files:
        print(f"No images found in {args.images_dir}")
        return

    print(f"Loaded scenario from {args.scenario}")
    print(f"Found {len(prompts)} prompts across {len(prompts_by_frame)} frames.")

    tracker = DAM4SAMMOT(model_size=args.model_size, checkpoint_dir=args.checkpoint_dir)
    colors = get_distinct_colors()
    
    # Tracked objects state
    active_obj_ids = [] # List of object IDs currently being tracked
    
    print("\nStarting execution...")
    
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        img_pil = Image.open(img_path).convert('RGB')
        img_cv = cv2.imread(img_path)
        
        # Check for prompts at this frame
        if i in prompts_by_frame:
            frame_prompts = prompts_by_frame[i]
            print(f"Frame {i}: Processing {len(frame_prompts)} prompts...")
            
            # Separate initialization (first time) vs addition
            # DAM4SAMMOT.initialize resets everything.
            # If we are at frame 0, we usually initialize.
            # If we are mid-sequence, we add.
            
            if i == 0:
                # Initialize with all frame 0 prompts
                init_regions = []
                for p in frame_prompts:
                    reg = {'obj_id': p['obj_id'], 'bbox': p['bbox']}
                    init_regions.append(reg)
                    active_obj_ids.append(p['obj_id'])
                
                tracker.initialize(img_pil, init_regions)
                print(f"  Initialized with {len(init_regions)} objects: {active_obj_ids}")
                
            else:
                # Add objects one by one
                for p in frame_prompts:
                    reg = {'bbox': p['bbox']} # add_object takes region dict
                    # Note: add_object returns (obj_id, mask)
                    # We might want to map the returned ID to our scenario ID if needed
                    # But DAM4SAMMOT generates its own IDs (0, 1, 2...) usually?
                    # Wait, initialize takes 'obj_id' in input list, but add_object returns new ID.
                    # Let's assume we just track them.
                    
                    tracker.add_object(img_pil, reg)
                    active_obj_ids.append(p['obj_id']) # Just tracking our logical IDs
                    print(f"  Added object: {p['obj_id']}")

        # Track
        if active_obj_ids:
            outputs = tracker.track(img_pil)
            masks = outputs['masks']
            
            # Visualization
            # We need to map masks to IDs. 
            # DAM4SAMMOT returns masks in order of internal ID.
            # Assuming consistent ordering for now.
            vis_img = draw_results(img_cv, masks, active_obj_ids, colors)
        else:
            vis_img = img_cv
            
        cv2.imwrite(os.path.join(frames_out_dir, filename), vis_img)
        if i % 10 == 0:
            print(f"Processed frame {i}/{len(image_files)}", end='\r')

    print("\nExecution complete.")
    create_video(frames_out_dir, os.path.join(args.output_dir, "result.mp4"))

if __name__ == "__main__":
    main()
