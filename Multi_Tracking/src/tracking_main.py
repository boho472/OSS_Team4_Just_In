from model.video_frame_exchange import video_to_frame, frame_to_video
from model.model_YOLO import use_YOLO
from model.model_ZoeDepth import use_ZoeDepth
from model.model_3D_convert import convert_to_3D
from model.model_DAM4SAM import DAM4SAMIntegration
from dataset.tracking_dataset import KittiTrackingDataset
from tracker.hybridtrack import HYBRIDTRACK
from configs.config_utils import cfg, cfg_from_yaml_file
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os
import argparse
import torch
from PIL import Image
from json_system.frame_db import update_frame_db
from json_system.tracker_log import update_tracker_log


def track_one_seq(seq_id,config,video_path,save_frame,save_txt,used_frame,result_file_name):
    saved_frame = [f for f in os.listdir(save_frame) if f.endswith('.jpg') or f.endswith('.png')]
    saved_frame.sort()
    
    dataset_path = config.dataset_path
    detections_path = config.detections_path
    tracking_type = config.tracking_type
    detections_path += "/" + str(seq_id).zfill(4)
    save_json_path = config.save_json_path + "/" + str(seq_id).zfill(4)
    os.makedirs(save_json_path, exist_ok=True)

    tracker = HYBRIDTRACK(box_type="Kitti", tracking_features=False, config = config)
    dataset = KittiTrackingDataset(dataset_path,save_frame,seq_id=seq_id,ob_path=detections_path,type=[tracking_type])
    dam4sam = DAM4SAMIntegration(
        model_size=config.d4sm_model_size if hasattr(
            config, 'd4sm_model_size') else 'tiny',
        checkpoint_dir=config.checkpoint_dir if hasattr(
            config, 'checkpoint_dir') else './checkpoints'
    )
    
    new_info = []
    new_info_dict = {}
    dict_key = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_det = YOLO("yolo11n.pt")
    yolo_seg = YOLO("yolo11n-seg.pt")
    depth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(device).eval()
    
    for i in range(len(dataset)):
        image_path = os.path.join(save_frame, saved_frame[i])
        txt_path = os.path.join(save_txt, saved_frame[i][:-4]+'.txt')
        
        boxes, masks = use_YOLO(image_path,yolo_det, yolo_seg)
        depth_map = use_ZoeDepth(image_path,depth_model,device)
        convert_to_3D(txt_path,boxes,masks,depth_map, save_json_path,saved_frame[i][:-4])
        
        
        _, _, _, _, objects, det_scores, _ = dataset[i]
        mask = det_scores>config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        tracker.tracking(objects[:,:7],
                             features=None,
                             scores=torch.tensor(det_scores),
                             timestamp=i)
        
        ##########################
        #if new_info:
        #    new_info = [j for j in new_info if j["gap_btw_18_nseen"] != 18]
        ##########################
        
        new_obj = tracker.new_obj
        
        with open(dataset.ob_txt_path) as f:
            obj_info = f.readlines()
            whole_txt_file = np.array([item.strip().split(' ') for item in obj_info])
        for (obj_num,obj_id) in new_obj:
            obj_id_str = "tracks_" + str(obj_id)
            tracker.current_frame_ids.remove(obj_id)
            if obj_id in tracker.previous_frame_ids:
                tracker.previous_frame_ids.remove(obj_id)
            for key, value in new_info_dict.items():
                if obj_id_str == key:
                    value["last_detected_frame"] = int(saved_frame[i][:-4])
                    value["undetected_num"] = 0
                    value["det_bbox"]["x"] = float(whole_txt_file[obj_num, 4])
                    value["det_bbox"]["y"] = float(whole_txt_file[obj_num, 5])
                    value["det_bbox"]["w"] = float(whole_txt_file[obj_num, 6])
                    value["det_bbox"]["h"] = float(whole_txt_file[obj_num, 7])
                    value["status"] = "detected"
            if obj_id_str not in dict_key:
                new_dict = {}
                new_dict[obj_id_str] = {}
                new_dict[obj_id_str]["created_frame"] = int(saved_frame[i][:-4])
                new_dict[obj_id_str]["last_detected_frame"] = int(saved_frame[i][:-4])
                new_dict[obj_id_str]["undetected_num"] = 0
                new_dict[obj_id_str]["det_bbox"] = {}
                new_dict[obj_id_str]["det_bbox"]["x"] = float(whole_txt_file[obj_num, 4])
                new_dict[obj_id_str]["det_bbox"]["y"] = float(whole_txt_file[obj_num, 5])
                new_dict[obj_id_str]["det_bbox"]["w"] = float(whole_txt_file[obj_num, 6])
                new_dict[obj_id_str]["det_bbox"]["h"] = float(whole_txt_file[obj_num, 7])
                new_dict[obj_id_str]["status"] = "detected"
                new_info.append(new_dict)
        
        for j in tracker.previous_frame_ids:
            id_str = "tracks_" + str(j)
            for info in new_info:
                if id_str in info:
                    info[id_str]["undetected_num"] += 1
                    info[id_str]["det_bbox"]["x"] = 0.0
                    info[id_str]["det_bbox"]["y"] = 0.0
                    info[id_str]["det_bbox"]["w"] = 0.0
                    info[id_str]["det_bbox"]["h"] = 0.0
                    info[id_str]["status"] = "undetected"
        
        new_info_dict = {}
        for item_dict in new_info:
            for key, value in item_dict.items():
                new_info_dict[key] = value
        dict_key = list(new_info_dict.keys())
        
        new_info_dict["dead"] = tracker.dead_list
        result_dict = {}
        frame_num = "frame_" + saved_frame[i][:-4]
        result_dict[frame_num] = new_info_dict

        #=======================
        # tracker_log 업데이트(HT 전체 로그)
        #=======================
        save_json_log_path = os.path.join(save_json_path, "hybrid_track_log.json")  
        update_tracker_log(save_json_log_path,frame_num,result_dict)
        
        print(result_dict, "\n")

        #===========================
        # HT 결과에서 필요한 정보만 추출
        #===========================
        hybridtrack_data = extract_ht_for_dam4sam(result_dict[frame_num])

        #===========================
        # frame_db에 HT 데이터 업데이트(id, bbox)
        #===========================
        frame_json_path = os.path.join(save_json_path, f"{saved_frame[i][:-4]}.json")

        #===========================
        #DAM4SAM 처리
        #===========================

        # 현재 프레임 이미지 로드
        image = Image.open(image_path)

        # DAM4SAM 처리
        dam_outputs = dam4sam.process_frame(
            frame_idx=i,
            frame_json_path=frame_json_path,
            hybridtrack_data=hybridtrack_data,
            image=image
        )
        print(f"✅ DAM4SAM processed frame {i}: {len(dam_outputs['masks'])} objects tracked")
        

def tracking_val_seq(arg):

    yaml_file = arg.cfg_file
    config = cfg_from_yaml_file(yaml_file,cfg)
    videos_path = config.dataset_path
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4') or f. endswith('.avi')]
    save_path = config.save_frame_path                       # the results saving path
    save_txt_path = config.save_txt_path
    used_frame_path = config.used_frame_path
    result_path = config.save_video_path
    os.makedirs(save_path,exist_ok=True)
    #seq_list = config.tracking_seqs    # the tracking sequences

    for id in range(len(video_files)):
        file_name = video_files[id][:-4]
        video_path = os.path.join(videos_path, video_files[id])
        save_frame = os.path.join(save_path, file_name)
        save_txt = os.path.join(save_txt_path, file_name)
        used_frame = os.path.join(used_frame_path, file_name)
        result_file_name = os.path.join(result_path, video_files[id])
        file_name = int(file_name)
        
        video_to_frame(video_path, save_frame)
        
        track_one_seq(file_name,config,video_path,save_frame,save_txt,used_frame,result_file_name)
        
        frame_to_video(used_frame, result_file_name)


def extract_ht_for_dam4sam(frame_tracks):
    """
    HybridTrack 결과에서 DAM4SAM에 필요한 정보만 추출
    
    Args:
        frame_tracks: result_dict[frame_key] 내용
            {
                "tracks_1": {"created_frame": 38, "det_bbox": {"x":..., "y":..., "w":..., "h":...}, ...},
                "tracks_2": {...},
                "dead": []
            }
    
    Returns:
        [
            {"object_id": 1, "bbox": [x, y, w, h]},
            {"object_id": 2, "bbox": [x, y, w, h]},
            ...
        ]
    """
    ht_results = []
    
    for track_key, track_info in frame_tracks.items():
        # "dead" 키는 스킵
        if not track_key.startswith("tracks_"):
            continue
        
        # tracks_1 → 1
        track_id = int(track_key.split("_")[1])
        
        # det_bbox 추출
        det_bbox = track_info.get("det_bbox")
        if det_bbox is None:
            continue
        
        # status가 "undetected"이면 bbox가 0,0,0,0일 수 있음 → 스킵
        if track_info.get("status") == "undetected":
            continue
        
        ht_results.append({
            "object_id": track_id,
            "bbox": [
                det_bbox["x"],
                det_bbox["y"],
                det_bbox["w"],
                det_bbox["h"]
            ]
        })
    
    return ht_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="",
                        help='specify the config for tracking')
    args = parser.parse_args()
    tracking_val_seq(args)