"""
최적화된 tracking_main.py
- 체크포인트 시스템 (중단 시 재개 가능)
- 배치 처리 및 메모리 최적화
- 진행 상황 저장
"""

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
import gc
from tqdm import tqdm


class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, "tracking_checkpoint.json")
    
    def save_checkpoint(self, frame_idx, tracker_state=None):
        """체크포인트 저장"""
        checkpoint = {
            "last_processed_frame": frame_idx,
            "timestamp": str(np.datetime64('now'))
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"💾 Checkpoint saved at frame {frame_idx}")
    
    def load_checkpoint(self):
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"✅ Resuming from frame {checkpoint['last_processed_frame']}")
            return checkpoint['last_processed_frame']
        return -1
    
    def clear_checkpoint(self):
        """체크포인트 삭제"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print("🗑️ Checkpoint cleared")


def clear_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def process_frames_batch(frames_batch, yolo_det, yolo_seg, depth_model, device, 
                         save_txt, save_json_path):
    """배치 단위로 프레임 처리 (YOLO + ZoeDepth)"""
    
    batch_results = []
    
    for frame_info in frames_batch:
        image_path = frame_info['image_path']
        txt_path = frame_info['txt_path']
        frame_name = frame_info['frame_name']
        
        # YOLO 처리
        boxes, masks, scores = use_YOLO(image_path, yolo_det, yolo_seg)
        
        # Depth 처리
        depth_map = use_ZoeDepth(image_path, depth_model, device)
        
        # 3D 변환
        convert_to_3D(txt_path, boxes, masks, depth_map, scores, 
                     save_json_path, frame_name)
        
        batch_results.append({
            'boxes': boxes,
            'masks': masks,
            'scores': scores,
            'depth_map': depth_map
        })
        
        # 배치 내 메모리 정리
        del boxes, masks, scores, depth_map
    
    # 배치 완료 후 메모리 정리
    clear_memory()
    
    return batch_results


def track_one_seq_optimized(seq_id, config, video_path, save_frame, save_txt, 
                            used_frame, result_file_name, batch_size=10, 
                            checkpoint_interval=50):
    """
    최적화된 시퀀스 추적 (프로파일링 버전)
    
    Args:
        batch_size: 배치 크기 (메모리에 따라 조정)
        checkpoint_interval: 체크포인트 저장 간격
    """
    
    import time  # ✅ 추가
    
    # 체크포인트 매니저 초기화
    checkpoint_dir = os.path.join(config.save_json_path, str(seq_id).zfill(4), "checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
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

    # MOT format 출력 파일
    mot_result_dir = 'src/data/pipeline_mot_result'
    os.makedirs(mot_result_dir, exist_ok=True)
    mot_result_file = os.path.join(mot_result_dir, f"{str(seq_id).zfill(4)}.txt")
    
    # 체크포인트 로드
    start_frame = checkpoint_manager.load_checkpoint() + 1
    
    # MOT 파일 모드 결정
    mot_mode = 'a' if start_frame > 0 else 'w'
    mot_file = open(mot_result_file, mot_mode)
    
    if start_frame > 0:
        print(f"🔄 Resuming from frame {start_frame}/{len(saved_frame)}")
    else:
        print(f"🚀 Starting fresh processing: {len(saved_frame)} frames")
    
    save_json_log_path = os.path.join(save_json_path, "hybrid_track_log.json")
    
    # Tracker 초기화
    tracker = HYBRIDTRACK(box_type="Kitti", tracking_features=False, config=config)
    dataset = KittiTrackingDataset(dataset_path, save_frame, seq_id=seq_id, 
                                   ob_path=detections_path, type=[tracking_type])
    
    # DAM4SAM 초기화
    dam4sam = DAM4SAMIntegration(
        model_size=config.d4sm_model_size if hasattr(config, 'd4sm_model_size') else 'tiny',
        checkpoint_dir=config.checkpoint_dir if hasattr(config, 'checkpoint_dir') else 'src/checkpoints'
    )
    
    new_info = []
    new_info_dict = {}
    dict_key = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 초기화
    print("🔧 Initializing models...")
    yolo_det = YOLO("yolo11n.pt")
    yolo_seg = YOLO("yolo11n-seg.pt")
    depth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(device).eval()
    
    # ✅ 프로파일링 데이터 저장
    profiling_data = {
        'yolo_times': [],
        'depth_times': [],
        'convert_3d_times': [],
        'tracking_times': [],
        'ht_result_times': [],
        'dam4sam_times': [],
        'total_times': []
    }
    
    # @torch.no_grad() 데코레이터 효과
    with torch.no_grad():
        # 프레임 처리
        for i in tqdm(range(start_frame, len(dataset)), desc="Processing frames"):
            try:
                frame_start = time.time()  # ✅ 프레임 시작 시간
                
                print(f"\n{'='*60}")
                print(f"현재 프레임: [{i + 1} / {len(saved_frame)}]")
                print(f"{'='*60}")
                
                image_path = os.path.join(save_frame, saved_frame[i])
                txt_path = os.path.join(save_txt, saved_frame[i][:-4]+'.txt')
                
                # ✅ YOLO + Depth 처리 (시간 측정)
                t_yolo = time.time()
                boxes, masks, scores = use_YOLO(image_path, yolo_det, yolo_seg)
                yolo_time = time.time() - t_yolo
                profiling_data['yolo_times'].append(yolo_time)

                
                t_depth = time.time()
                depth_map = use_ZoeDepth(image_path, depth_model, device)
                depth_time = time.time() - t_depth
                profiling_data['depth_times'].append(depth_time)

                
                t_convert = time.time()
                convert_to_3D(txt_path, boxes, masks, depth_map, scores, 
                            save_json_path, saved_frame[i][:-4])
                convert_time = time.time() - t_convert
                profiling_data['convert_3d_times'].append(convert_time)

                
                # 메모리 정리
                del boxes, masks, scores, depth_map
                
                # ✅ Tracking 처리 (시간 측정)
                t_tracking = time.time()
                _, _, _, _, objects, det_scores, _ = dataset[i]
                mask = det_scores > config.input_score
                objects = objects[mask]
                det_scores = det_scores[mask]

                tracker.tracking(objects[:,:7],
                               features=None,
                               scores=torch.tensor(det_scores),
                               timestamp=i)
                tracking_time = time.time() - t_tracking
                profiling_data['tracking_times'].append(tracking_time)

                
                t_ht_result = time.time()
                new_info, new_info_dict, dict_key = ht_result(
                    tracker, dataset, i, saved_frame, new_info, 
                    new_info_dict, dict_key, save_json_log_path
                )

                # HT 전체 상태 추출
                ht_status_dict = {}
                for track_key, track_info in new_info_dict.items():
                    if not track_key.startswith("tracks_"):
                        continue
                    track_id = int(track_key.split("_")[1])
                    ht_status_dict[track_id] = {
                        'status': track_info.get('status'),
                        'undetected_num': track_info.get('undetected_num', 0)
                    }

                ht_result_time = time.time() - t_ht_result
                profiling_data['ht_result_times'].append(ht_result_time)


                # 주기적 체크포인트 저장
                if (i + 1) % checkpoint_interval == 0:
                    checkpoint_manager.save_checkpoint(i)
                    clear_memory()
                    print(f"💾 Checkpoint saved & memory cleared at frame {i+1}")
                
                # 매 프레임마다 가벼운 메모리 정리
                if (i + 1) % 10 == 0:
                    clear_memory()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️ OOM at frame {i}. Saving checkpoint and cleaning memory...")
                    checkpoint_manager.save_checkpoint(i - 1)
                    clear_memory()
                    print("💡 Restart the script to resume from last checkpoint")
                    raise
                else:
                    raise

    mot_file.close()
    
    # 완료 후 체크포인트 삭제
    checkpoint_manager.clear_checkpoint()
    
    print(f"\n✅ MOT format 결과 저장 완료: {mot_result_file}")
    print(f"   저장된 총 프레임 수: {len(dataset)}\n")



def tracking_val_seq(arg):
    """메인 실행 함수"""
    
    yaml_file = arg.cfg_file
    config = cfg_from_yaml_file(yaml_file, cfg)
    videos_path = config.dataset_path
    video_files = [f for f in os.listdir(videos_path) 
                   if f.endswith('.mp4') or f.endswith('.avi')]
    save_path = config.save_frame_path
    save_txt_path = config.save_txt_path
    used_frame_path = config.used_frame_path
    result_path = config.save_video_path
    
    os.makedirs(save_path, exist_ok=True)

    for id in range(len(video_files)):
        file_name = video_files[id][:-4]
        video_path = os.path.join(videos_path, video_files[id])
        save_frame = os.path.join(save_path, file_name)
        save_txt = os.path.join(save_txt_path, file_name)
        used_frame = os.path.join(used_frame_path, file_name)
        result_file_name = os.path.join(result_path, video_files[id])
        file_name = int(file_name)
        
        print(f"\n{'='*60}")
        print(f"📹 Processing video {id + 1}/{len(video_files)}: {video_files[id]}")
        print(f"{'='*60}\n")
        
        # 최적화된 추적 실행
        track_one_seq_optimized(
            file_name, config, video_path, save_frame, save_txt, 
            used_frame, result_file_name,
            batch_size=10,  # 메모리 상황에 따라 조정
            checkpoint_interval=50  # 50 프레임마다 체크포인트
        )
        
        print("🎬 객체 추적 영상 생성\n")
        frame_to_video(used_frame, result_file_name)
        
        print(f"\n✅ Video {id + 1}/{len(video_files)} completed\n")
        
        # 비디오 간 메모리 정리
        clear_memory()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized tracking with checkpoints')
    parser.add_argument('--cfg_file', type=str, default="",
                        help='specify the config for tracking')
    parser.add_argument('--resume', action='store_true',
                        help='resume from last checkpoint')
    args = parser.parse_args()
    
    tracking_val_seq(args)