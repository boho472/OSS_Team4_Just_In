"""
DAM4SAM 통합 모듈
HybridTrack과 연동하여 동적 객체 추가 기능 제공
"""

import os
import json
from PIL import Image
from .tracking_wrapper_mot import DAM4SAMMOT


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
            frame_json_path: 프레임 JSON 파일 경로 (예: frame_000038.json)
            hybridtrack_data: [{"object_id": 1, "bbox": [x, y, w, h]}, ...]
            image: PIL Image

        Returns:
            dam_outputs: DAM4SAM 추적 결과 {'masks': [...]}
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

        #=======================
        # 3. 프레임 JSON에  DAM4SAM 결과 업데이트
        #=======================
        frame_data["hybridtrack"] = hybridtrack_data  # HT 결과도 추가
        frame_data["dam4sam"] = dam_outputs
        
        with open(frame_json_path, 'w') as f:
            json.dump(frame_data, f, indent=2)
        
        print(f"✅ Frame {frame_idx}: DAM4SAM tracked {len(dam_outputs['masks'])} objects")
        print(f"   Saved to: {frame_json_path}")

        return dam_outputs

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
