
import json
import os

def load_frame_db(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)

def save_frame_db(json_path,db):
    with open(json_path, "w") as f:
        json.dump(db, f, indent=2)

def update_frame_db(json_path,frame_key, yolo_data, norm3d, hybridtrack_data, dam_data):
    db = load_frame_db(json_path)
    db[frame_key] = {
        "yolo_detection": yolo_data,       # YOLO bbox + mask id
        "normalization_3d": norm3d,        # X,Y,Z,H,W,L,theta
        "hybridtrack": hybridtrack_data,   # 🔥 새로 추가된 HT 정보
        "dam4sam": dam_data                # 최종 segmentation 결과
    }
    save_frame_db(json_path,db)
