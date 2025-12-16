"""
메모리 최적화된 DAM4SAM 통합 모듈
- CPU 오프로딩
- 메모리 효율적인 마스크 저장
- 배치 처리 지원
"""

import os
import json
from PIL import Image
from .tracking_wrapper_mot import DAM4SAMMOT
import numpy as np
import cv2
import torch
import gc


def convert_to_serializable(obj):
    """NumPy ndarray를 재귀적으로 리스트로 변환"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj


class DAM4SAMIntegration:
    """메모리 최적화된 HybridTrack-DAM4SAM 통합 클래스"""

    def __init__(self, model_size='tiny', checkpoint_dir='./checkpoints', 
                 cpu_offload=True):
        """
        Args:
            model_size: SAM2 모델 크기 ('tiny', 'small', 'base', 'large')
            checkpoint_dir: 체크포인트 디렉토리 경로
            cpu_offload: CPU 메모리 오프로딩 사용 여부
        """
        self.tracker = DAM4SAMMOT(
            model_size=model_size,
            checkpoint_dir=checkpoint_dir
        )

        # ✅ 추가: 객체별 미탐지 카운터
        self.object_not_seen_count = {}
        self.removal_threshold = 75  # 75 프레임 이상 안 보이면 제거
        
        self.cpu_offload = cpu_offload
        print(f"✅ DAM4SAM initialized with model size: {model_size}")
        print(f"   CPU offloading: {cpu_offload}")

    @torch.no_grad()
    def process_frame(self, frame_idx, frame_json_path, hybridtrack_data, image, ht_status_dict=None):
        """
        메모리 최적화된 프레임 처리
        
        Args: 
            frame_idx: 현재 프레임 번호
            frame_json_path: 프레임 JSON 파일 경로
            hybridtrack_data: [{"object_id": 1, "bbox": [x, y, w, h]}, ...]
            image: PIL Image
            ht_status_dict: HT의 전체 객체 상태 정보
        
        Returns:
            dam_outputs: DAM4SAM 추적 결과
        """
        # 기존 프레임 JSON 로드
        if os.path.exists(frame_json_path):
            with open(frame_json_path, 'r') as f:
                frame_data = json.load(f)
        else:
            frame_data = {
                "yolo_detection": None,
                "normalization_3d": None,
                "dam4sam": None
            }
    
        # DAM4SAM 처리
        dam_outputs = self.tracker.process_frame_with_ht_data(
            frame_idx=frame_idx,
            ht_data=hybridtrack_data,
            image=image
        )

        # ✅ 추가: 추적 실패 카운트 업데이트
        tracked_ids = [m.get('internal_id') for m in dam_outputs.get('masks', [])]

        for obj_id in list(self.tracker.all_obj_ids):
            if obj_id not in self.object_not_seen_count:
                self.object_not_seen_count[obj_id] = 0
            
            # ✅ 수정: DAM4SAM이 마스크를 생성했는지 먼저 확인
            if obj_id in tracked_ids:
                # DAM4SAM이 추적 중이면 무조건 리셋
                self.object_not_seen_count[obj_id] = 0
                continue

            # HT 상태 확인 (우선순위 1)
            if ht_status_dict and obj_id in ht_status_dict:
                ht_status = ht_status_dict[obj_id]
                if ht_status['status'] == 'undetected':
                    # HT가 undetected라고 판단하면 무조건 카운트 증가
                    self.object_not_seen_count[obj_id] = ht_status.get('undetected_num', 0)
                    continue
            
            # HT 정보도 없으면 카운트 증가
            self.object_not_seen_count[obj_id] += 1
        
        # CPU 오프로딩 (메모리 절약)
        if self.cpu_offload and dam_outputs.get('mask_arrays'):
            # GPU 텐서를 CPU로 이동하고 numpy로 변환
            dam_outputs['mask_arrays'] = [
                mask.cpu().numpy() if torch.is_tensor(mask) else mask
                for mask in dam_outputs['mask_arrays']
            ]
    
        # JSON 저장용 데이터 생성 (mask_arrays 제외)
        dam_outputs_for_json = {
            "masks": dam_outputs["masks"]
        }
        
        # 프레임 JSON 업데이트
        frame_data["hybridtrack"] = hybridtrack_data
        frame_data["dam4sam"] = dam_outputs_for_json
        
        serializable_frame_data = convert_to_serializable(frame_data)
        
        with open(frame_json_path, 'w') as f:
            json.dump(serializable_frame_data, f, indent=2)
        
        print(f"✅ Frame {frame_idx}: DAM4SAM tracked {len(dam_outputs['masks'])} objects")
        print(f"   Saved to: {frame_json_path}")
    
        return dam_outputs

    def cleanup_dead_objects(self, dead_list_from_ht):
        """
        안전한 객체 제거: HT dead + 충분히 오래 안 보임
        
        Args:
            dead_list_from_ht: HT에서 죽었다고 판단한 객체 ID 리스트
        
        Returns:
            removed: 실제로 제거된 객체 ID 리스트
        """

        removed = []

        # ✅ dead 중에서 all_obj_ids에 있는 것만 확인
        dead_in_dam = [obj_id for obj_id in dead_list_from_ht if obj_id in self.tracker.all_obj_ids]
        print(f"   dead & in all_obj_ids: {len(dead_in_dam)} objects → {dead_in_dam}")

        for obj_id in dead_list_from_ht:
            # all_obj_ids에 없으면 skip (DAM4SAM이 추적 안 하는 객체)
            if obj_id not in self.tracker.all_obj_ids:
                continue
                
            # ✅ 조건: object_not_seen_count >= threshold AND HT dead
            if obj_id in self.object_not_seen_count:
                count = self.object_not_seen_count[obj_id]
                
                if count >= self.removal_threshold:
                    # DAM4SAM도 충분히 오래 못 봄 + HT도 dead
                    self.tracker.all_obj_ids.remove(obj_id)
                    if obj_id in self.tracker.per_object_outputs_all:
                        del self.tracker.per_object_outputs_all[obj_id]
                    if obj_id in self.tracker.per_object_obj_ptr:
                        del self.tracker.per_object_obj_ptr[obj_id]
                    if obj_id in self.tracker.add_to_drm_next:
                        del self.tracker.add_to_drm_next[obj_id]
                    if obj_id in self.object_not_seen_count:
                        del self.object_not_seen_count[obj_id]
                    
                    removed.append(obj_id)
                    print(f"   🗑️ Removed obj_id={obj_id} (DAM count={count}, HT dead)")
                else:
                    print(f"   ⏳ Keep obj_id={obj_id} (DAM count={count} < {self.removal_threshold})")
            else:
                # object_not_seen_count에 없음 → 카운트가 안 되고 있다는 의미
                print(f"   ⚠️ obj_id={obj_id}: in all_obj_ids but NOT in object_not_seen_count!")

        print(f"   all_obj_ids after cleanup: {sorted(list(self.tracker.all_obj_ids))}")
        print(f"   Total removed: {len(removed)}/{len(dead_list_from_ht)}\n")

        return removed

def visualize_segmentation(image, masks, obj_ids, scores=None, alpha=0.5):
    """
    메모리 효율적인 시각화
    """
    vis_image = image.copy()
    
    np.random.seed(42)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(masks))]
    
    for idx, (mask, obj_id) in enumerate(zip(masks, obj_ids)):
        if mask is None or (isinstance(mask, np.ndarray) and mask.sum() == 0):
            continue
        
        # 텐서를 numpy로 변환
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
            
        color = colors[idx % len(colors)]
        
        # 마스크 영역 색상 채우기
        colored_mask = np.zeros_like(vis_image)
        colored_mask[mask > 0] = color
        
        vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
        
        # 윤곽선 그리기
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # 객체 ID 표시
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                text = f"ID:{obj_id} ({scores[idx]:.2f})" if scores else f"ID:{obj_id}"
                
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
                
                cv2.putText(
                    vis_image, text, (cx, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        
        # 메모리 정리
        del mask, colored_mask, contours
    
    return vis_image



def extract_ht_for_dam4sam(new_info_dict, index):
    """
    HybridTrack 결과에서 DAM4SAM에 필요한 정보만 추출
    
    Args:
        new_info_dict: {"tracks_0": {...}, "tracks_1": {...}, "dead": [...]}
        index: 현재 프레임 번호
    """
    ht_results = []
    
    for track_key, track_info in new_info_dict.items():
        if not track_key.startswith("tracks_"):
            continue
        
        # ✅ 원래 로직: created_frame 체크
        if track_info.get('created_frame') != index:
            continue
        
        if track_info.get("status") == "undetected":
            continue
        
        track_id = int(track_key.split("_")[1])
        det_bbox = track_info.get("det_bbox")
        
        if det_bbox is None:
            continue
        
        # dict 형태의 bbox 변환
        if isinstance(det_bbox, dict):
            x1 = det_bbox.get("x")
            y1 = det_bbox.get("y")
            x2 = det_bbox.get("w")
            y2 = det_bbox.get("h")
            bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        else:
            continue
        
        ht_results.append({
            "object_id": track_id,
            "bbox": bbox_xywh
        })
    
    return ht_results


@torch.no_grad()
def use_DAM4SAM(image_path, dam4sam, i, frame_json_path, hybridtrack_data, 
                used_frame, saved_frame, mot_file=None, dead_list=None, ht_status_dict=None):
    """
    메모리 최적화된 DAM4SAM 처리
    """
    # 이미지 로드
    image = Image.open(image_path)

    # ✅ 추가: dead 객체 정리 (프레임 처리 전)
    if dead_list:
        dam4sam.cleanup_dead_objects(dead_list)

    # DAM4SAM 처리
    dam_outputs = dam4sam.process_frame(
        frame_idx=i,
        frame_json_path=frame_json_path,
        hybridtrack_data=hybridtrack_data,
        image=image,
        ht_status_dict=ht_status_dict
    )
    print(f"✅ DAM4SAM processed frame {i}: {len(dam_outputs['masks'])} objects tracked")

    # MOT format 저장
    if mot_file is not None and dam_outputs.get('masks'):
        frame_id = i + 1
        
        for mask_meta in dam_outputs['masks']:
            obj_id = mask_meta.get('internal_id')
            bbox = mask_meta.get('bbox')
            
            if obj_id is not None and bbox is not None and len(bbox) == 4:
                x, y, w, h = bbox
                mot_file.write(f"{frame_id},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1\n")

    # 시각화
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
        
        vis_save_path = os.path.join(used_frame, saved_frame[i])
        cv2.imwrite(vis_save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"💾 Saved visualization to: {vis_save_path}")
        
        # 메모리 정리
        del vis_image, masks
    else:
        import shutil
        vis_save_path = os.path.join(used_frame, saved_frame[i])
        shutil.copy(image_path, vis_save_path)
        print(f"⚠️ No masks, copied original image: {vis_save_path}")
    
    # 이미지 메모리 정리
    del image
    gc.collect()


def create_dam4sam_tracker(model_size='tiny', checkpoint_dir='./checkpoints', 
                           cpu_offload=True):
    """
    메모리 최적화된 DAM4SAM 트래커 생성
    """
    return DAM4SAMIntegration(model_size, checkpoint_dir, cpu_offload)