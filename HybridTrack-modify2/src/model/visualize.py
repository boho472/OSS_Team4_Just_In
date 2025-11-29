import numpy as np
import cv2
import os
import re
import torch
import glob
import math

def compute_3d_box_points(h, w, l, x, y, z, ry):
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h] 
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    corners_3d = R @ corners_3d
    
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    
    return corners_3d.T

def project_3d_to_image(P2, pts_3d):
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d_hom = P2 @ pts_3d_hom.T
    pts_2d = (pts_2d_hom[:2] / pts_2d_hom[2]).T
    return pts_2d.astype(np.int32)

def visualize2video(P2, config, seq_id):
    results = {}
    seq_name = str(seq_id).zfill(4)
    tracking_result = os.path.join(config.save_path, seq_name+'.txt')
    
    with open(tracking_result) as f:
        for line in f.readlines():
            data = re.split(" ", line.strip())
            if len(data) < 5:
                continue
            
            frame_id = int(data[0])
            track_id = int(data[1])
            dims_loc_ry = np.array(data[10:17], dtype=np.float32)
            
            if frame_id not in results:
                results[frame_id] = []
            
            results[frame_id].append({
                'id': track_id, 
                'data': dims_loc_ry
            })
            
    image_path = os.path.join(config.dataset_path, 'image_02/'+seq_name)
    image_files = sorted(glob.glob(os.path.join(image_path, '*.png')))
    if not image_files:
        print(f"Error: No images found in {image_path}")
        return

    height, width, _ = cv2.imread(image_files[0]).shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = os.path.join(config.save_path, seq_name+'.avi')
    video_writer = cv2.VideoWriter(save_path, fourcc, 15.0, (width, height))
    
    for image_file_path in image_files:
        frame_name = os.path.basename(image_file_path)
        frame_id = int(frame_name[:6])
        
        img = cv2.imread(image_file_path)
        if img is None: continue
            
        if frame_id in results:
            for ob in results[frame_id]:
                h, w, l, x, y, z, ry = ob['data']
                track_id = ob['id']
                
                corners_3d = compute_3d_box_points(h, w, l, x, y, z, ry)
                corners_2d = project_3d_to_image(P2, corners_3d)
                
                box_color = (0, 255, 0) # Green
                
                lines = [
                    (0, 1), (1, 2), (2, 3), (3, 0), 
                    (4, 5), (5, 6), (6, 7), (7, 4), 
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                
                for start, end in lines:
                    cv2.line(img, tuple(corners_2d[start]), tuple(corners_2d[end]), box_color, 1)
                
                cv2.putText(img, f'ID:{track_id}', tuple(corners_2d[4]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
        video_writer.write(img)
        
    video_writer.release()