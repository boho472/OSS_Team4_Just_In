import torch
import numpy as np
import cv2
import sys

def use_ZoeDepth(image_path,depth_model,device):
    if depth_model is None:
        return None
    
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    inp = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.
    inp = inp.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = depth_model(inp)
    
    if isinstance(out,dict):
        depth = out["metric_depth"]
    else:
        depth = out
        
    return depth.squeeze().cpu().numpy()