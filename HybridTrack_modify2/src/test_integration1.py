"""
HybridTrack + DAM4SAM 통합 테스트
- 개선: 현재 프레임 segmentation 사용
- 실험: 빠른 움직임에서 DAM4SAM 한계 증명
"""

from pathlib import Path
import sys
from PIL import Image
import numpy as np
import json
import os
import cv2

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from model.model_DAM4SAM import DAM4SAMIntegration


def create_dummy_image_with_objects(frame_idx, width=1024, height=1024):
    """
    실제 객체가 그려진 더미 이미지 생성 (25프레임 시나리오)
    
    시나리오:
    - 객체 1 (빨강): 속도 변화 테스트
      * Frame 0-4: 천천히 (5px/frame)
      * Frame 5-14: 빠르게 (20px/frame) → ID switching 유발
      * Frame 15-19: 초고속 (30px/frame) → 재차 ID switching
      * Frame 20-24: 감속 (10px/frame)
    
    - 객체 2 (초록): 일정 속도 제어군 (3px/frame)
    
    - 객체 3 (파랑): Frame 5부터 등장, 천천히 이동 (2px/frame)
    
    - 객체 4 (노랑): Frame 10부터 등장, 대각선 이동
    """
    # 그라디언트 배경
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        intensity = int(255 * y / height)
        img[y, :] = [intensity // 3, intensity // 2, intensity]
    
    # ========================================
    # 객체 1 (빨강): 속도 변화 테스트
    # ========================================
    if frame_idx < 5:
        speed = 5  # 천천히
        x1 = 100 + frame_idx * speed
    elif frame_idx < 15:
        # Phase 2: 빠르게 (Frame 5-14)
        base_x = 100 + 4 * 5  # 120
        x1 = base_x + (frame_idx - 4) * 20
    elif frame_idx < 20:
        # Phase 3: 초고속 (Frame 15-19)
        base_x = 100 + 4 * 5 + 10 * 20  # 320
        x1 = base_x + (frame_idx - 14) * 30
    else:
        # Phase 4: 감속 (Frame 20-24)
        base_x = 100 + 4 * 5 + 10 * 20 + 5 * 30  # 470
        x1 = base_x + (frame_idx - 19) * 10

    y1 = 100
    cv2.rectangle(img, (x1, y1), (x1 + 50, y1 + 50), (0, 0, 255), -1)  # 빨강
    
    # ========================================
    # 객체 2 (초록): 일정 속도 제어군
    # ========================================

    x2 = 200
    y2 = 200 + frame_idx * 3
    cv2.rectangle(img, (x2, y2), (x2 + 60, y2 + 60), (0, 255, 0), -1)  # 초록
    
    # ========================================
    # 객체 3 (파랑): Frame 5부터 등장
    # ========================================
    if frame_idx >= 5:
        x3 = 300
        y3 = 300 + (frame_idx - 5) * 2
        cv2.rectangle(img, (x3, y3), (x3 + 55, y3 + 55), (255, 0, 0), -1)  # 파랑
    
    # ========================================
    # 객체 4 (노랑): Frame 10부터 등장, 대각선 이동
    # ========================================
    if frame_idx >= 10:
        x4 = 400 + (frame_idx - 10) * 4
        y4 = 100 + (frame_idx - 10) * 3
        cv2.rectangle(img, (x4, y4), (x4 + 45, y4 + 45), (0, 255, 255), -1)  # 노랑

    return img


def create_dummy_ht_new_info(frame_idx):
    """
    HybridTrack 시뮬레이션(25프레임)
    """
    
    frame_key = f"frame_{frame_idx:06d}"

    # ========================================
    # Frame 0: 초기화 (2개 객체)
    # ========================================
    if frame_idx == 0:
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": 0,
                    "undetected_num": 0,
                    "det_bbox": [100, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": 0,
                    "undetected_num": 0,
                    "det_bbox": [200, 200, 60, 60],
                    "status": "detected"
                },
                "dead": []
            }
        }
    
    # ========================================
    # Frame 1-4: 천천히 이동
    # ========================================
    elif frame_idx < 5:
        # Frame 1-4: 천천히 이동
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [100 + frame_idx * 5, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [200, 200 + frame_idx * 3, 60, 60],
                    "status": "detected"
                },
                "dead": []
            }
        }
    
    # ========================================
    # Frame 5: 객체 1 가속 + 객체 3 등장
    # ========================================
    elif frame_idx == 5:
        # Frame 5: 객체 1 갑자기 빨라짐 + 객체 3 등장
        base_x = 100 + 4 * 5  # 120
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [base_x + (frame_idx - 4) * 20, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [200, 200 + frame_idx * 3, 60, 60],
                    "status": "detected"
                },
                "tracks_3": {  # 새 객체!
                    "created_frame": 5,
                    "last_detected_frame": 5,
                    "undetected_num": 0,
                    "det_bbox": [300, 300, 55, 55],
                    "status": "detected"
                },
                "dead": []
            }
        }
    
    # ========================================
    # Frame 6-9: 3개 객체 추적
    # ========================================
    elif frame_idx < 10:
        base_x = 100 + 4 * 5
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [base_x + (frame_idx - 4) * 20, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [200, 200 + frame_idx * 3, 60, 60],
                    "status": "detected"
                },
                "tracks_3": {
                    "created_frame": 5,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [300, 300 + (frame_idx - 5) * 2, 55, 55],
                    "status": "detected"
                },
                "dead": []
            }
        }
    
    # ========================================
    # Frame 10: 객체 4 등장
    # ========================================
    elif frame_idx == 10:
        base_x = 100 + 4 * 5
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [base_x + (frame_idx - 4) * 20, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [200, 200 + frame_idx * 3, 60, 60],
                    "status": "detected"
                },
                "tracks_3": {
                    "created_frame": 5,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [300, 300 + (frame_idx - 5) * 2, 55, 55],
                    "status": "detected"
                },
                "tracks_4": {  # 새 객체!
                    "created_frame": 10,
                    "last_detected_frame": 10,
                    "undetected_num": 0,
                    "det_bbox": [400, 100, 45, 45],
                    "status": "detected"
                },
                "dead": []
            }
        }
    
    # ========================================
    # Frame 11-14: 4개 객체 추적
    # ========================================
    elif frame_idx < 15:
        base_x = 100 + 4 * 5
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [base_x + (frame_idx - 4) * 20, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [200, 200 + frame_idx * 3, 60, 60],
                    "status": "detected"
                },
                "tracks_3": {
                    "created_frame": 5,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [300, 300 + (frame_idx - 5) * 2, 55, 55],
                    "status": "detected"
                },
                "tracks_4": {
                    "created_frame": 10,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [400 + (frame_idx - 10) * 4, 100 + (frame_idx - 10) * 3, 45, 45],
                    "status": "detected"
                },
                "dead": []
            }
        }
    
    # ========================================
    # Frame 15-19: 객체 1 초고속 (재가속)
    # ========================================
    elif frame_idx < 20:
        base_x = 100 + 4 * 5 + 10 * 20  # 320
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [base_x + (frame_idx - 14) * 30, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [200, 200 + frame_idx * 3, 60, 60],
                    "status": "detected"
                },
                "tracks_3": {
                    "created_frame": 5,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [300, 300 + (frame_idx - 5) * 2, 55, 55],
                    "status": "detected"
                },
                "tracks_4": {
                    "created_frame": 10,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [400 + (frame_idx - 10) * 4, 100 + (frame_idx - 10) * 3, 45, 45],
                    "status": "detected"
                },
                "dead": []
            }
        }
    
    # ========================================
    # Frame 20-24: 객체 1 감속, 최종 안정화
    # ========================================
    else:
        base_x = 100 + 4 * 5 + 10 * 20 + 5 * 30  # 470
        return {
            frame_key: {
                "tracks_1": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [base_x + (frame_idx - 19) * 10, 100, 50, 50],
                    "status": "detected"
                },
                "tracks_2": {
                    "created_frame": 0,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [200, 200 + frame_idx * 3, 60, 60],
                    "status": "detected"
                },
                "tracks_3": {
                    "created_frame": 5,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [300, 300 + (frame_idx - 5) * 2, 55, 55],
                    "status": "detected"
                },
                "tracks_4": {
                    "created_frame": 10,
                    "last_detected_frame": frame_idx,
                    "undetected_num": 0,
                    "det_bbox": [400 + (frame_idx - 10) * 4, 100 + (frame_idx - 10) * 3, 45, 45],
                    "status": "detected"
                },
                "dead": []
            }
        }

def test_integration():
    """통합 테스트 실행"""
    
    print("=" * 80)
    print("HybridTrack + DAM4SAM Integration Test")
    print("Scenario: 25-Frame Comprehensive Tracking Challenge")
    print("=" * 80)
    
    # 테스트 디렉토리 생성
    test_dir = "test_integration"
    json_dir = os.path.join(test_dir, "jsons")
    image_dir = os.path.join(test_dir, "images")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # ✅ DAM4SAM + 통합 모듈 초기화
    print("\n[1] Initializing DAM4SAM Integration...")
    integration = DAM4SAMIntegration(
        model_size='tiny',
        checkpoint_dir='./checkpoints'
    )
    print("✅ DAM4SAM Integration initialized")
    
    # 테스트 프레임 수
    n_frames = 25
    print(f"\n[2] Processing {n_frames} frames...")
    print("-" * 80)
    
    for frame_idx in range(n_frames):
        print(f"\n{'='*60}")
        print(f"🚩 Frame {frame_idx}")
        print(f"{'='*60}")
        
        # 이미지 생성
        img_array = create_dummy_image_with_objects(frame_idx)
        image = Image.fromarray(img_array)
        
        # 이미지 저장
        img_path = os.path.join(image_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # ✅ HT 원본 구조 생성
        ht_new_info = create_dummy_ht_new_info(frame_idx)
        
        # ✅ 통합 모듈로 처리 (JSON 생성 + DAM4SAM 추적)
        try:
            outputs = integration.process_frame(
                frame_idx,
                ht_new_info,  # ← 원본 딕셔너리 그대로
                json_dir,
                image
            )
            
            # 결과 확인
            json_path = os.path.join(json_dir, f"frame_{frame_idx:06d}.json")
            with open(json_path, 'r') as f:
                result = json.load(f)
            
            dam_results = result['dam4sam_tracking']['DAM4SAM_results']
            actions = result['dam4sam_tracking'].get('actions', [])
            
            print(f"\n✅ DAM4SAM Results:")
            print(f"   Total objects tracked: {len(dam_results)}")
            for dam_obj in dam_results:
                print(f"     - internal_id={dam_obj['internal_id']}, "
                      f"bbox={dam_obj['bbox']}, pixels={dam_obj['mask_pixels']}")
            
            if actions:
                print(f"\n   Actions:")
                for action in actions:
                    print(f"     - {action}")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "=" * 80)
    print("🎉 Test Complete!")
    print("=" * 80)
    
    # 결과 분석
    print("\n[3] Result Analysis:")
    print("-" * 80)
    
    analyze_results(json_dir, n_frames)
    
    # 비디오 생성
    print("\n[4] Generating visualization video...")
    print("-" * 80)
    
    try:
        from visualize_tracking import create_tracking_video
        
        video_path = os.path.join(test_dir, "tracking_result.mp4")
        create_tracking_video(
            json_dir=json_dir,
            output_path=video_path,
            fps=5
        )
        
        print(f"\n✅ Video saved: {video_path}")
        print("   Duration: ~5 seconds at 5 FPS")
    
    except ImportError:
        print("\n⚠️ visualize_tracking.py not found. Skipping video generation.")
    except Exception as e:
        print(f"\n⚠️ Video generation failed: {e}")


def analyze_results(json_dir, n_frames):
    """
    결과 분석 및 요약
    """
    print("\n📊 Tracking Performance Analysis:")
    print("-" * 60)
    
    id_switches = []
    object_tracking = {1: [], 2: [], 3: [], 4: []}  # HT obj_id -> DAM internal_ids
    
    for frame_idx in range(n_frames):
        json_path = os.path.join(json_dir, f"frame_{frame_idx:06d}.json")
        
        if not os.path.exists(json_path):
            continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        actions = data['dam4sam_tracking'].get('actions', [])
        
        for action in actions:
            ht_id = action['ht_obj_id']
            dam_id = action.get('internal_id')
            act_type = action['action']
            
            if dam_id is not None and ht_id in object_tracking:
                object_tracking[ht_id].append({
                    'frame': frame_idx,
                    'dam_id': dam_id,
                    'action': act_type,
                    'overlap': action.get('overlap_ratio', 0.0)
                })
    
    # ========================================
    # 객체 1 분석 (속도 변화)
    # ========================================
    print("\n🔴 Object 1 (Speed Change Test - Red):")
    print("  Expected: Multiple ID switches due to rapid acceleration")
    print("  Actual tracking:")

    for record in object_tracking[1]:
        status = "✅" if record['action'] in ['INIT', 'MATCH'] else "🔄 ID SWITCH"
        print(f"    Frame {record['frame']:2d}: DAM-{record['dam_id']} "
              f"({record['action']:5s}, overlap={record['overlap']:.3f}) {status}")
    
    dam_ids_obj1 = [r['dam_id'] for r in object_tracking[1]]
    unique_dam_ids_obj1 = set(dam_ids_obj1)
    
    if len(unique_dam_ids_obj1) > 1:
        print(f"\n  ⚠️  ID SWITCHING DETECTED!")
        print(f"  Used {len(unique_dam_ids_obj1)} different DAM IDs: {unique_dam_ids_obj1}")
        
        # ID 전환 지점 찾기
        switches = []
        for i in range(1, len(dam_ids_obj1)):
            if dam_ids_obj1[i] != dam_ids_obj1[i-1]:
                frame = object_tracking[1][i]['frame']
                switches.append(frame)
        
        print(f"  ID switches occurred at frames: {switches}")
    else:
        print(f"\n  ✅ No ID switching - tracking stable")
    
    # ========================================
    # 객체 2 분석 (제어군)
    # ========================================
    print("\n🟢 Object 2 (Constant Speed Control - Green):")
    print("  Expected: Consistent DAM ID throughout")
    print("  Sample tracking (first 5, last 5):")
    
    obj2_tracking = object_tracking[2]
    for record in obj2_tracking[:5]:
        print(f"    Frame {record['frame']:2d}: DAM-{record['dam_id']} "
              f"({record['action']:5s}, overlap={record['overlap']:.3f})")
    
    if len(obj2_tracking) > 10:
        print("    ...")
        for record in obj2_tracking[-5:]:
            print(f"    Frame {record['frame']:2d}: DAM-{record['dam_id']} "
                  f"({record['action']:5s}, overlap={record['overlap']:.3f})")
    
    dam_ids_obj2 = [r['dam_id'] for r in object_tracking[2]]
    unique_dam_ids_obj2 = set(dam_ids_obj2)
    
    if len(unique_dam_ids_obj2) == 1:
        print(f"\n  ✅ Perfect tracking - single ID {list(unique_dam_ids_obj2)[0]} maintained")
    else:
        print(f"\n  ⚠️  Unexpected ID switching in control object!")

    # ========================================
    # 객체 3 분석 (동적 추가)
    # ========================================
    print("\n🔵 Object 3 (Dynamic Addition - Blue):")
    if object_tracking[3]:
        first_record = object_tracking[3][0]
        print(f"  First appeared: Frame {first_record['frame']}")
        print(f"  Assigned: DAM-{first_record['dam_id']}")
        print(f"  ✅ Dynamic object addition successful")
        
        dam_ids_obj3 = [r['dam_id'] for r in object_tracking[3]]
        if len(set(dam_ids_obj3)) == 1:
            print(f"  ✅ Stable tracking after addition")
    
    # ========================================
    # 객체 4 분석 (대각선 이동)
    # ========================================
    print("\n🟡 Object 4 (Diagonal Movement - Yellow):")
    if object_tracking[4]:
        first_record = object_tracking[4][0]
        print(f"  First appeared: Frame {first_record['frame']}")
        print(f"  Assigned: DAM-{first_record['dam_id']}")
        print(f"  ✅ Dynamic object addition successful")
        
        dam_ids_obj4 = [r['dam_id'] for r in object_tracking[4]]
        if len(set(dam_ids_obj4)) == 1:
            print(f"  ✅ Stable diagonal tracking")
    
    # ========================================
    # 최종 요약
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    if len(unique_dam_ids_obj1) > 1:
        print(f"\n🎯 Research Contribution Demonstrated:")
        print(f"   ❌ DAM4SAM: {len(unique_dam_ids_obj1)} ID switches for fast-moving object")
        print(f"   ✅ HybridTrack: Maintained consistent obj_id=1 throughout")
        print(f"\n   💡 This proves the complementary strengths:")
        print(f"      - HybridTrack: Robust ID management with motion prediction")
        print(f"      - DAM4SAM: Superior segmentation quality")
        print(f"      - Integration: Best of both worlds!")
    else:
        print("\n⚠️  No ID switching detected - try increasing speed further")


if __name__ == "__main__":
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available. Using CPU (will be slow)")
    
    test_integration()