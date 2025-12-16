"""
DAM4SAM 통합 모듈
HybridTrack과 연동하여 동적 객체 추가 기능 제공
"""

import os
import json
from PIL import Image
from .tracking_wrapper_mot import DAM4SAMMOT
import numpy as np
import cv2
import shutil


def convert_to_serializable(obj):
    """NumPy ndarray를 재귀적으로 리스트로 변환합니다."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        # 배열은 파이썬 리스트로 변환합니다.
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        # NumPy 정수 타입을 파이썬 기본 int로 변환합니다. (추가적인 안정성 확보)
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        # NumPy 실수 타입을 파이썬 기본 float로 변환합니다. (추가적인 안정성 확보)
        return float(obj)
    else:
        return obj

class DAM4SAMIntegration:
    """HybridTrack과 DAM4SAM 통합 클래스"""

    def __init__(self, model_size='tiny', checkpoint_dir='./checkpoints'):
        """
        Args:
            model_size: SAM2 모델 크기 ('tiny', 'small', 'base', 'large')
            checkpoint_dir: 체크포인트 디렉토리 경로
        """
        self.tracker = DAM4SAMMOT(
            model_size=model_size,
            checkpoint_dir=checkpoint_dir
        )
        print(f"✅ DAM4SAM initialized with model size: {model_size}")

    def process_frame(self, frame_idx, frame_json_path, hybridtrack_data, image):
        """
        프레임 처리: DAM4SAM 추적 -> 프레임 JSON 업데이트
    
        Args: 
            frame_idx: 현재 프레임 번호
            frame_json_path: 프레임 JSON 파일 경로 (예: 000038.json)
            hybridtrack_data: [{"object_id": 1, "bbox": [x, y, w, h]}, ...]
            image: PIL Image
    
        Returns:
            dam_outputs: DAM4SAM 추적 결과 {'masks': [...], 'mask_arrays': [...]}
        """
        #=======================
        # 1. 기존 프레임 JSON 로드
        #=======================
        if os.path.exists(frame_json_path):
            with open(frame_json_path, 'r') as f:
                frame_data = json.load(f)
        else:
            # JSON이 없으면 기본 구조 생성
            frame_data = {
                "yolo_detection": None,
                "normalization_3d": None,
                "dam4sam": None
            }
    
        #=======================
        # 2. DAM4SAM 처리
        #=======================
        dam_outputs = self.tracker.process_frame_with_ht_data(
            frame_idx=frame_idx,
            ht_data=hybridtrack_data,  # [{"object_id": 1, "bbox": [x,y,w,h]}, ...]
            image=image
        )
        # dam_outputs = {
        #     "masks": [...],        # 메타데이터 (bbox, mask_pixels 등)
        #     "mask_arrays": [...]   # 실제 마스크 numpy 배열
        # }
    
        #=======================
        # 3. ✅ JSON 저장용 데이터 생성 (mask_arrays 제외!)
        #=======================
        dam_outputs_for_json = {
            "masks": dam_outputs["masks"]  # 메타데이터만 저장!
            # mask_arrays는 제외!
        }
        
        #=======================
        # 4. 프레임 JSON에 DAM4SAM 결과 업데이트
        #=======================
        frame_data["hybridtrack"] = hybridtrack_data  # HT 결과도 추가
        frame_data["dam4sam"] = dam_outputs_for_json  # ✅ mask_arrays 없는 버전!
        
        serializable_frame_data = convert_to_serializable(frame_data)
        
        with open(frame_json_path, 'w') as f:
            json.dump(serializable_frame_data, f, indent=2)
        
        print(f"✅ Frame {frame_idx}: DAM4SAM tracked {len(dam_outputs['masks'])} objects")
        print(f"   Saved to: {frame_json_path}")
    
        return dam_outputs  # ✅ 시각화용으로는 mask_arrays 포함된 전체 반환

    def create_frame_json(self, frame_idx, new_info, json_dir):
        """
        HybridTrack의 new_info를 DAM4SAM JSON 형식으로 변환하여 저장

        Args:
            frame_idx: 현재 프레임 번호
            new_info: HybridTrack의 new_info 딕셔너리
                    {
                        "frame_000038: {
                            "tracks_1": {"created_frame": 38, "det_bbox": [x,y,w,h], ...},
                            "tracks_2": {...},
                            "dead": []
                        }
                    }
                    [{object_id, bbox: {x,y,w,h}, gap_btw_18_nseen}, ...]
            json_dir: JSON 저장 디렉토리

        Returns:
            json_path: 생성된 JSON 파일 경로
        """
        import json

        # JSON 구조 생성
        frame_data = {
            "frame_number": frame_idx,
            "dam4sam_tracking": {
                "HybridTrack_results": [],
                "DAM4SAM_results": []
            }
        }

        frame_key = f"frame_{frame_idx:06d}"

        if frame_key not in new_info:
            print(f"Warning: {frame_key} not found in new_info")
            #빈 JSON 저장
            json_path = os.path.join(json_dir, f"frame_{frame_idx:06d}.json")
            with open(json_path, 'w', encoding='utf_8') as f:
                json.dump(frame_data, f, indent=2, ensure_ascii=False)
            return json_path

        frame_tracks = new_info[frame_key]

        for track_key, track_info in frame_tracks.items():
            if not track_key.startswith("tracks_"):
                continue
                
            #track_id 추출: "tracks_1 -> 1"
            track_id = int(track_key.split("_")[1])

            #det_bbox 추출: [x, y, w, h]
            det_bbox = track_info.get("det_bbox")
            if det_bbox is None:
                print(f"Warning: det_bbox missing for {track_key}")
                continue
            
            ht_result = {
                "object_id": track_id,
                "bbox": {
                    "x": int(det_bbox[0]),
                    "y": int(det_bbox[1]),
                    "w": int(det_bbox[2]),
                    "h": int(det_bbox[3])
                }
            }
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append(ht_result)

        # JSON 파일 저장
        json_path = os.path.join(json_dir, f"frame_{frame_idx:06d}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)

        return json_path


def create_dam4sam_tracker(model_size='tiny', checkpoint_dir='./checkpoints'):
    """
    DAM4SAM 트래커 생성

    Args:
        model_size: 모델 크기
        checkpoint_dir: 체크포인트 경로

    Returns:
        DAM4SAMIntegration 인스턴스
    """
    return DAM4SAMIntegration(model_size, checkpoint_dir)


def visualize_segmentation(image, masks, obj_ids, scores=None, alpha=0.5):
    """
    Segmentation 마스크를 원본 이미지 위에 오버레이하여 시각화
    
    Args:
        image: 원본 이미지 (numpy array, RGB)
        masks: 마스크 리스트 (각 마스크는 binary numpy array)
        obj_ids: 객체 ID 리스트
        scores: 신뢰도 점수 리스트 (optional)
        alpha: 마스크 투명도 (0~1)
    
    Returns:
        시각화된 이미지 (numpy array, RGB)
    """
    
    # 이미지 복사
    vis_image = image.copy()
    
    # 각 객체별로 고유한 색상 생성
    np.random.seed(42)  # 일관된 색상을 위해
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(masks))]
    
    # 각 마스크를 오버레이
    for idx, (mask, obj_id) in enumerate(zip(masks, obj_ids)):
        if mask.sum() == 0 or None:  # 빈 마스크는 건너뜀
            continue
            
        color = colors[idx % len(colors)]
        
        # 마스크 영역을 색상으로 채움
        colored_mask = np.zeros_like(vis_image)
        colored_mask[mask > 0] = color
        
        # 알파 블렌딩으로 오버레이
        vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
        
        # 마스크 윤곽선 그리기
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # 객체 ID와 점수 표시
        if len(contours) > 0:
            # 마스크의 중심점 계산
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 텍스트 생성
                if scores is not None:
                    text = f"ID:{obj_id} ({scores[idx]:.2f})"
                else:
                    text = f"ID:{obj_id}"
                
                # 텍스트 배경 그리기
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    vis_image, 
                    (cx - 5, cy - text_height - 5), 
                    (cx + text_width + 5, cy + 5), 
                    color, 
                    -1
                )
                
                # 텍스트 그리기
                cv2.putText(
                    vis_image, 
                    text, 
                    (cx, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
    
    return vis_image


def extract_ht_for_dam4sam(frame_tracks,index):
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
        
        if track_info['created_frame'] != index:
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

        # ✅ xyxy → xywh 변환
        if isinstance(det_bbox, (list, tuple)) and len(det_bbox) == 4:
            x1, y1, x2, y2 = det_bbox
            bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            print(f"[Extract] Track {track_id}: xyxy {det_bbox} → xywh {bbox_xywh}")
        elif isinstance(det_bbox, dict):
            x1 = det_bbox.get("x")
            y1 = det_bbox.get("y")
            x2 = det_bbox.get("w")  # 실제로는 x2
            y2 = det_bbox.get("h")  # 실제로는 y2
            bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            print(f"[Extract] Track {track_id}: dict xyxy→xywh → {bbox_xywh}")
        else:
            continue

        
        ht_results.append({
            "object_id": track_id,
            "bbox": bbox_xywh
        })
    
    return ht_results


def use_DAM4SAM(image_path,dam4sam,i,frame_json_path,hybridtrack_data,used_frame,saved_frame, mot_file=None):
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
    
  # ===== 디버그: DAM4SAM 출력 구조 확인 =====
    if dam_outputs.get('masks'):
        print(f"\n🔍 DAM4SAM masks structure (Frame {i}):")
        print(f"   Number of masks: {len(dam_outputs['masks'])}")
        if len(dam_outputs['masks']) > 0:
            print(f"   First mask keys: {dam_outputs['masks'][0].keys()}")
            print(f"   First mask sample: {dam_outputs['masks'][0]}")

    # ✅ 여기에서 mask_arrays의 타입/shape을 확인
    if dam_outputs.get('mask_arrays'):
        print("type(mask_arrays[0]) =", type(dam_outputs['mask_arrays'][0]))
        print("mask_arrays[0].shape =", dam_outputs['mask_arrays'][0].shape)

    # ===========================
    # ✅ MOT format 저장 (DAM4SAM 최종 결과)
    # ===========================
    if mot_file is not None and dam_outputs.get('masks'):
        frame_id = i + 1  # MOT format은 1부터 시작
        
        for mask_meta in dam_outputs['masks']:
            obj_id = mask_meta.get('internal_id')
            bbox = mask_meta.get('bbox')  # [x, y, w, h]
            
            if obj_id is not None and bbox is not None and len(bbox) == 4:
                x, y, w, h = bbox
                # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                mot_file.write(f"{frame_id},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1\n")

    # ===========================
    # Segmentation 시각화 이미지 생성 및 used_frame 폴더에 저장
    # ===========================
    if dam_outputs.get('mask_arrays'):
        masks = dam_outputs['mask_arrays']

        meta_list = dam_outputs.get('masks', [])
        if meta_list and isinstance(meta_list[0], dict) and 'internal_id' in meta_list[0]:
            obj_ids = [m['internal_id'] for m in meta_list]
        else:
            obj_ids = list(range(len(masks)))
        
        vis_image = visualize_segmentation(
            image=np.array(image),
            masks=masks,
            obj_ids=obj_ids,
            scores=None
        )
        
        # used_frame 폴더에 시각화 이미지 저장
        vis_save_path = os.path.join(used_frame, saved_frame[i])
        cv2.imwrite(vis_save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"💾 Saved visualization to used_frame: {vis_save_path}")
    else:
        # 마스크가 없는 경우 원본 이미지를 복사
        import shutil
        vis_save_path = os.path.join(used_frame, saved_frame[i])
        shutil.copy(image_path, vis_save_path)
        print(f"⚠️ No masks detected, copied original image: {vis_save_path}")