from model.video_frame_exchange import video_to_frame, frame_to_video
from model.model_YOLO import use_YOLO
from model.model_ZoeDepth import use_ZoeDepth
from model.model_3D_convert import convert_to_3D
from model.model_DAM4SAM import DAM4SAMIntegration, use_DAM4SAM, extract_ht_for_dam4sam
from model.print_ht_result import ht_result
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
import cv2
from PIL import Image
from json_system.frame_db import update_frame_db
from json_system.tracker_log import update_tracker_log


def track_one_seq(seq_id,config,video_path,save_frame,save_txt,used_frame,result_file_name):  #,point_cloud_frame):
    saved_frame = [f for f in os.listdir(save_frame) if f.endswith('.jpg') or f.endswith('.png')]
    saved_frame.sort()
    
    dataset_path = config.dataset_path
    detections_path = config.detections_path
    tracking_type = config.tracking_type
    detections_path += "/" + str(seq_id).zfill(4)
    save_json_path = config.save_json_path + "/" + str(seq_id).zfill(4)
    
    os.makedirs(save_json_path, exist_ok=True)
    os.makedirs(save_txt, exist_ok=True)
    os.makedirs(used_frame, exist_ok=True)
    #os.makedirs(point_cloud_frame, exist_ok=True)

    # ===== MOT format 출력 파일 생성 =====
    mot_result_dir = 'src/data/pipeline_mot_result'
    mot_result_file = os.path.join(mot_result_dir, f"{str(seq_id).zfill(4)}.txt")
    mot_file = open(mot_result_file, 'w')
    print(f"📝 MOT format 결과 저장 시작: {mot_result_file}\n")
    
    save_json_log_path = os.path.join(save_json_path, "hybrid_track_log.json")
    if os.path.exists(save_json_log_path):
        os.remove(save_json_log_path)

    tracker = HYBRIDTRACK(box_type="Kitti", tracking_features=False, config = config)
    dataset = KittiTrackingDataset(dataset_path,save_frame,seq_id=seq_id,ob_path=detections_path,type=[tracking_type])
    
    print(os.path.abspath(__file__))
    
    dam4sam = DAM4SAMIntegration(
        model_size=config.d4sm_model_size if hasattr(
            config, 'd4sm_model_size') else 'tiny',
        checkpoint_dir=config.checkpoint_dir if hasattr(
            config, 'checkpoint_dir') else 'src/checkpoints'
    )
    
    new_info = []
    new_info_dict = {}
    dict_key = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_det = YOLO("yolo11n.pt")
    yolo_seg = YOLO("yolo11n-seg.pt")
    depth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(device).eval()
    
    print(f"\n프레임 수 : {len(saved_frame)}")
    
    for i in range(len(dataset)):
        print(f"\n현재 프레임 : [{i + 1} / {len(saved_frame)}]")
        image_path = os.path.join(save_frame, saved_frame[i])
        txt_path = os.path.join(save_txt, saved_frame[i][:-4]+'.txt')
        #point_frame = os.path.join(point_cloud_frame, saved_frame[i])
        
        boxes, masks, scores = use_YOLO(image_path,yolo_det, yolo_seg)
        depth_map = use_ZoeDepth(image_path,depth_model,device)
        convert_to_3D(txt_path,boxes,masks,depth_map,scores,save_json_path,saved_frame[i][:-4])
        
        _, _, _, _, objects, det_scores, _ = dataset[i]
        mask = det_scores>config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        tracker.tracking(objects[:,:7],
                             features=None,
                             scores=torch.tensor(det_scores),
                             timestamp=i)
        
        new_info, new_info_dict, dict_key = ht_result(tracker,dataset,i,saved_frame,new_info,new_info_dict,dict_key,save_json_log_path)

        #===========================
        # HT 결과에서 필요한 정보만 추출
        #===========================
        print(f"\n=== Frame {i} Debug ===")
        frame_key = f"frame_{int(saved_frame[i][:-4]):06d}"
        if frame_key in new_info_dict:
            for track_key, track_info in new_info_dict[frame_key].items():
                if track_key.startswith("tracks_") and track_info.get('created_frame') == int(saved_frame[i][:-4]):
                    det_bbox = track_info.get("det_bbox")
                    print(f"Track {track_key}:")
                    print(f"  Type: {type(det_bbox)}")
                    print(f"  Value: {det_bbox}")
                    
                    # 값 분석
                    if isinstance(det_bbox, dict):
                        x, y, w, h = det_bbox.get("x"), det_bbox.get("y"), det_bbox.get("w"), det_bbox.get("h")
                        print(f"  Interpreted as dict: x={x}, y={y}, w={w}, h={h}")
                        if w > x + 50:
                            print(f"  → This looks like xyxy! (w={w} > x={x})")
                    elif isinstance(det_bbox, (list, tuple)):
                        v1, v2, v3, v4 = det_bbox
                        print(f"  Interpreted as list: v1={v1}, v2={v2}, v3={v3}, v4={v4}")
                        if v3 > v1 + 50:
                            print(f"  → This looks like xyxy! (v3={v3} > v1={v1})")
        hybridtrack_data = extract_ht_for_dam4sam(new_info_dict,int(saved_frame[i][:-4]))

        #===========================
        # frame_db에 HT 데이터 업데이트(id, bbox)
        #===========================
        frame_json_path = os.path.join(save_json_path, f"{saved_frame[i][:-4]}.json")

        use_DAM4SAM(image_path,dam4sam,i,frame_json_path,hybridtrack_data,used_frame,saved_frame, mot_file)

    mot_file.close()
    print(f"\n✅ MOT format 결과 저장 완료: {mot_result_file}")
    print(f"   저장된 총 프레임 수: {len(dataset)}\n")

def tracking_val_seq(arg):

    yaml_file = arg.cfg_file
    config = cfg_from_yaml_file(yaml_file,cfg)
    videos_path = config.dataset_path
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4') or f. endswith('.avi')]
    save_path = config.save_frame_path                       # the results saving path
    save_txt_path = config.save_txt_path
    used_frame_path = config.used_frame_path
    result_path = config.save_video_path
    #save_point_cloud_frame = config.save_point_cloud_frame
    #save_point_cloud_video = config.save_point_cloud_video
    
    os.makedirs(save_path,exist_ok=True)
    #seq_list = config.tracking_seqs    # the tracking sequences

    for id in range(len(video_files)):
        file_name = video_files[id][:-4]
        video_path = os.path.join(videos_path, video_files[id])
        save_frame = os.path.join(save_path, file_name)
        save_txt = os.path.join(save_txt_path, file_name)
        used_frame = os.path.join(used_frame_path, file_name)
        result_file_name = os.path.join(result_path, video_files[id])
        #point_cloud_frame = os.path.join(save_point_cloud_frame, file_name)
        #point_cloud_video = os.path.join(save_point_cloud_video, video_files[id])
        file_name = int(file_name)
        
        print(f"{id + 1}번째 영상 처리 시작. [{id + 1} / {len(video_files)}]\n")
        
        #video_to_frame(video_path, save_frame)
        
        track_one_seq(file_name,config,video_path,save_frame,save_txt,used_frame,result_file_name)  #,point_cloud_frame)
        
        print("객체 추적 영상 생성\n")
        frame_to_video(used_frame, result_file_name)
        #print("포인트 클라우드 영상 생성\n")
        #frame_to_video(point_cloud_frame, point_cloud_video)
        
        print(f"{id + 1}번째 영상 처리 완료. [{id + 1} / {len(video_files)}]\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="",
                        help='specify the config for tracking')
    args = parser.parse_args()
    tracking_val_seq(args)