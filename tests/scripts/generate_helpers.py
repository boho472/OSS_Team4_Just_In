#!/usr/bin/env python3
"""
Generate Helper Frames with Grid
--------------------------------
Generates images with coordinate grids for specific frames to help user select BBox coordinates.
"""

import os
import sys
import cv2
import glob

def draw_grid(img, step=50):
    """Draw a coordinate grid on the image"""
    h, w = img.shape[:2]
    vis = img.copy()
    
    # Draw vertical lines
    for x in range(0, w, step):
        color = (200, 200, 200) if x % 100 != 0 else (255, 255, 255)
        thickness = 1 if x % 100 != 0 else 2
        cv2.line(vis, (x, 0), (x, h), color, thickness)
        if x % 100 == 0:
            cv2.putText(vis, str(x), (x+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    # Draw horizontal lines
    for y in range(0, h, step):
        color = (200, 200, 200) if y % 100 != 0 else (255, 255, 255)
        thickness = 1 if y % 100 != 0 else 2
        cv2.line(vis, (0, y), (w, y), color, thickness)
        if y % 100 == 0:
            cv2.putText(vis, str(y), (2, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    return vis

def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images_dir = os.path.join(project_root, 'n1')
    output_dir = os.path.join(project_root, 'tests', 'helpers')
    
    os.makedirs(output_dir, exist_ok=True)
    
    target_frames = [0, 30, 70]
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Generating helper frames in {output_dir}...")
    
    for frame_idx in target_frames:
        if frame_idx < len(image_files):
            img_path = image_files[frame_idx]
            img = cv2.imread(img_path)
            
            # Draw grid
            grid_img = draw_grid(img, step=50)
            
            # Save
            out_name = f"helper_frame_{frame_idx:04d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, grid_img)
            print(f"Saved {out_path}")
        else:
            print(f"Frame {frame_idx} out of range.")

if __name__ == "__main__":
    main()
