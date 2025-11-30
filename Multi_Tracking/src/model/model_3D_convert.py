import numpy as np
import cv2

def middle_40_mask(mask_full):
    ys, xs = np.where(mask_full > 0)
    if len(xs) == 0:
        return np.zeros_like(mask_full)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    H = y_max - y_min
    W = x_max - x_min

    y1 = y_min + int(H * 0.30)
    y2 = y_min + int(H * 0.70)

    x1 = x_min + int(W * 0.30)
    x2 = x_min + int(W * 0.70)

    torso_mask = np.zeros_like(mask_full)
    torso_mask[y1:y2, x1:x2] = mask_full[y1:y2, x1:x2]

    return torso_mask


def compute_Z_from_mask(depth_map, mask_full):
    if mask_full is None:
        return None
    
    mask_mid = middle_40_mask(mask_full)
    
    ys, xs = np.where(mask_mid > 0)
    if len(xs) == 0:
        return None
    
    H, W = depth_map.shape
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    
    depth_vals = depth_map[ys, xs]
    
    Z = np.percentile(depth_vals, 20)
    
    return float(Z)


def convert_to_3D(txt_path,boxes,masks,depth_map,save_json_path,file_name):
    out = open(txt_path, "w")
    used_Z = set()
    
    yolo_dict = {}
    norm3d_dict = {}
    
    for idx, (bbox, mask_full) in enumerate(zip(boxes, masks)):
        x1, y1, x2, y2 = bbox
        
        yolo_dict[det_id] = {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class": "person",
            "score": 1.0   # score 필요하면 run_det_then_seg에서 가져와서 넣으면 됨
        }

        Z = compute_Z_from_mask(depth_map, mask_full)
        if Z is None:
            print(f"if Z is None")
            continue

        rounded_Z = round(Z, 3)

        if rounded_Z in used_Z:
            print(f"if rounded_Z in used_Z")
            continue

        used_Z.add(rounded_Z)

        ys, xs = np.where(mask_full > 0)
        u = float(np.mean(xs))
        v = float(np.mean(ys))

        H, W = depth_map.shape
        FOV_x = np.deg2rad(70)
        fx = W / (2 * np.tan(FOV_x / 2))
        fy = fx

        X = (u - W/2) * Z / fx
        Y = (v - H/2) * Z / fy

        out.write(
            f"Pedestrian 0 0 0 {x1} {y1} {x2} {y2} "
            f"1.7571 0.7342 0.8948 {X:.4f} {Y:.4f} {Z:.4f} 0.0 1.0\n"
        )
        
        norm3d_dict[det_id] = {
            "h": 1.7571,
            "w": 0.7342,
            "l": 0.8948,
            "x": float(X),
            "y": float(Y),
            "z": float(Z),
            "theta": 0.0
        }
    
    frame_json_path = os.path.join(save_json_path, file_name+'.json')
    frame_json_data = {
        "yolo_detection": yolo_dict,
        "normalization_3d": norm3d_dict,
        "dam4sam": None
    }
    with open(frame_json_path, "w") as f:
        json.dump(frame_json_data, f, indent=2)
    
    out.close()